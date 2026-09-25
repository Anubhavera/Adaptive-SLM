#!/usr/bin/env python3
# AdaptiveSLM Training Pipeline v2

"""
Training pipeline for AdaptiveSLM v2 (deep-thin dense SLM + elastic depth).

Architecture (modern dense SLM recipe, MobileLLM-style deep-thin + Qwen3 tricks):
- 30 layers x 768 hidden (deep & thin), GQA 12 Q-heads / 4 KV-heads (head_dim 64)
- RoPE (theta 1e6), RMSNorm, SwiGLU, tied input/output embeddings
- Optional immediate block-wise weight sharing (MobileLLM-LS), optional QK-norm
- Elastic depth (MatFormer-style): aux LM heads at layers 15/22 trained jointly,
  so depth-30/22/15 prefixes are all valid standalone models -> export a family.

Why no MoE: sub-512MB budget means ~500M params total; MoE needs >=1B active
params to beat dense (Qwen3 ships dense below 4B). Elastic depth gives the
"adaptive inference" story instead, and PAKD user profiles pick the depth.

Phases (each auto-resumes the latest checkpoint, so per-phase processes chain):
- pretrain : raw-corpus LM training (pretrain_corpus.jsonl, {"text": ...})
- distill  : 4-bit teacher KD (distill_data.jsonl, {"messages": [...]})
- pakd     : profile-aware KD (pakd_data.jsonl, {"messages": [...], "profile": ...})
- export   : HF qwen3-compatible safetensors per depth (converts via llama.cpp)

Chat data is wrapped in ChatML (<|im_start|>role\\ncontent<|im_end|>) here.
"""

import argparse
import contextlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from safetensors.torch import save_file as save_safetensors
except ImportError:
    save_safetensors = None

PROFILE_ORDER = ["beginner", "intermediate", "expert", "general"]


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training + architecture configuration for AdaptiveSLM v2"""

    # Architecture (deep & thin)
    vocab_size: int = 151936            # matches Qwen3 embedding exactly
    hidden_size: int = 768
    num_layers: int = 30
    num_attention_heads: int = 12       # head_dim = 768 / 12 = 64
    num_key_value_heads: int = 4        # GQA groups of 3
    intermediate_size: int = 2048
    head_dim: int = 64
    max_position_embeddings: int = 4096
    rope_theta: float = 1_000_000.0     # Qwen2.5/3-style long-context base
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    use_qk_norm: bool = True             # Qwen3-family convention (required for export)
    pair_share: bool = False            # MobileLLM-LS immediate block-wise sharing

    # Elastic depth (MatFormer-style nested model)
    elastic_exit_layers: tuple = (15, 22)
    aux_loss_weight: float = 0.3

    # Sequence/batching (defaults sized for a free-tier T4 16GB)
    max_length: int = 512
    batch_size: int = 16                # pretrain
    distill_batch_size: int = 4         # teacher+student logits don't fit at 16
    gradient_accumulation_steps: int = 8

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200             # optimizer steps
    max_steps: int = 100_000            # optimizer steps

    # Distillation
    teacher_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer_model: str = "Qwen/Qwen3-0.6B"
    distillation_alpha: float = 0.5     # weight of KD vs CE
    temperature: float = 2.0
    pakd_profile_weights: dict = field(
        default_factory=lambda: {
            "beginner": 1.2,
            "intermediate": 1.0,
            "expert": 0.8,
            "general": 1.0,
        }
    )

    # Memory
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True    # bf16 if supported, else fp16

    # Paths
    output_dir: str = "./checkpoints"
    data_path: str = "./data"


class AdaptiveSLMConfig:
    """Runtime config handed to the model modules"""

    def __init__(self, config: TrainingConfig):
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.intermediate_size = config.intermediate_size
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rms_norm_eps = config.rms_norm_eps
        self.tie_word_embeddings = config.tie_word_embeddings
        self.use_qk_norm = config.use_qk_norm
        self.elastic_exit_layers = config.elastic_exit_layers


# ============================================================================
# Model
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 1_000_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.to(q.dtype)[None, None, :, :]
    sin = sin.to(q.dtype)[None, None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: AdaptiveSLMConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor],
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if hasattr(self, "q_norm"):
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = apply_rotary(q, k, cos, sin)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(v.dtype)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: AdaptiveSLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AdaptiveSLMBlock(nn.Module):
    def __init__(self, config: AdaptiveSLMConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor],
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, cos, sin)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


def causal_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted CE; returns 0-graph loss if a batch has no valid targets."""
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    if not (shift_labels != -100).any():
        return logits.sum() * 0.0
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class AdaptiveSLMModel(nn.Module):
    """
    Deep-thin dense SLM with elastic-depth aux exits.

    forward returns:
      loss      - shifted CE (if labels given)
      logits    - full-depth LM head output
      aux_logits- {exit_depth: logits} from intermediate exit norms (elastic family)
    """

    def __init__(self, config: AdaptiveSLMConfig, pair_share: bool = False):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta)

        blocks = [AdaptiveSLMBlock(config) for _ in range(config.num_hidden_layers)]
        if pair_share:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i - 1]
        self.layers = nn.ModuleList(blocks)

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.exit_norms = nn.ModuleDict({
            str(l): RMSNorm(config.hidden_size, config.rms_norm_eps)
            for l in config.elastic_exit_layers
        })

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        hidden = self.embed_tokens(input_ids)

        causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        mask = causal[None, None]
        if attention_mask is not None:
            pad = ~attention_mask.bool()
            mask = mask | pad[:, None, None, :]

        cos, sin = self.rotary(seq_len, device, hidden.dtype)

        aux_hidden = {}
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    layer, hidden, mask, cos, sin, use_reentrant=False)
            else:
                hidden = layer(hidden, mask, cos, sin)
            depth = i + 1
            if str(depth) in self.exit_norms:
                aux_hidden[depth] = hidden

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        aux_logits = {}
        for depth, h in aux_hidden.items():
            aux_logits[depth] = self.lm_head(self.exit_norms[str(depth)](h))

        loss = None
        if labels is not None:
            loss = causal_cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits, "aux_logits": aux_logits}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Distillation losses
# ============================================================================

class DistillationLoss(nn.Module):
    """CE(ground truth) + KL(teacher soft targets), both over shifted positions."""

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        temp = self.temperature
        # Teachers can have a larger padded vocab; slice to the student's.
        teacher_logits = teacher_logits[..., : student_logits.size(-1)]

        s = student_logits[:, :-1].float()
        t = teacher_logits[:, :-1].float()
        y = labels[:, 1:].contiguous()

        ce = causal_cross_entropy(student_logits, labels)

        student_lp = F.log_softmax(s / temp, dim=-1)
        teacher_p = F.softmax(t / temp, dim=-1)
        kd = F.kl_div(student_lp, teacher_p, reduction="batchmean") * (temp * temp)

        return (1 - self.alpha) * ce + self.alpha * kd


class PAKDLoss(DistillationLoss):
    """
    Profile-Aware Knowledge Distillation (novel contribution):
    per-sample loss weighted by the PAKD user profile (beginner/expert/etc.).
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0,
                 profile_weights: Optional[dict] = None):
        super().__init__(alpha, temperature)
        self.profile_weights = profile_weights or {"beginner": 1.2, "intermediate": 1.0, "expert": 0.8, "general": 1.0}

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                labels: torch.Tensor, profile_ids: torch.Tensor) -> torch.Tensor:
        temp = self.temperature
        teacher_logits = teacher_logits[..., : student_logits.size(-1)]

        bsz = student_logits.size(0)
        s = student_logits[:, :-1].float()
        t = teacher_logits[:, :-1].float()
        y = labels[:, 1:].contiguous()
        vocab = s.size(-1)

        flat_ce = F.cross_entropy(s.reshape(-1, vocab), y.reshape(-1), ignore_index=-100, reduction="none")
        valid = (y != -100).reshape(-1)
        ce_b = (flat_ce * valid).reshape(bsz, -1).sum(1) / valid.reshape(bsz, -1).sum(1).clamp(min=1)

        student_lp = F.log_softmax(s / temp, dim=-1)
        teacher_p = F.softmax(t / temp, dim=-1)
        kd_b = F.kl_div(student_lp, teacher_p, reduction="none").sum(-1).mean(1) * (temp * temp)

        per_sample = (1 - self.alpha) * ce_b + self.alpha * kd_b

        weights = torch.ones(bsz, device=student_logits.device)
        for pid, name in enumerate(PROFILE_ORDER):
            weights[profile_ids == pid] = float(self.profile_weights.get(name, 1.0))

        return (per_sample * weights).mean()


# ============================================================================
# Data
# ============================================================================

class AdaptiveSLMDataset(Dataset):
    """
    JSONL dataset. Row formats:
      {"text": str}
      {"messages": [{"role": ..., "content": ...}, ...]}           -> ChatML
      {"messages": [...], "profile": "beginner|...|general"}       -> ChatML + profile id
    """

    PROFILE_MAP = {"beginner": 0, "intermediate": 1, "expert": 2, "general": 3}

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        if not self.data:
            raise ValueError(f"No samples loaded from {data_path}")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def to_chatml(messages) -> str:
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)

    def __getitem__(self, idx):
        item = self.data[idx]
        if "messages" in item:
            text = self.to_chatml(item["messages"])
        else:
            text = item.get("text", "")
        ids = self.tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"]
        out = {"input_ids": ids}
        if "profile" in item:
            out["profile_id"] = self.PROFILE_MAP.get(item["profile"], 3)
        return out


def collate_fn(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attention, labels = [], [], []
    for b in batch:
        ids = b["input_ids"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        labels.append(list(ids) + [-100] * pad)
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if "profile_id" in batch[0]:
        out["profile_id"] = torch.tensor([b["profile_id"] for b in batch], dtype=torch.long)
    return out


# ============================================================================
# Trainer
# ============================================================================

class AdaptiveSLMTrainer:
    CHECKPOINT_PRIORITY = ["pakd_finetuned", "distilled", "pretrained"]

    def __init__(self, config: TrainingConfig, resume: bool = True):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        self.model = AdaptiveSLMModel(
            AdaptiveSLMConfig(config), pair_share=config.pair_share
        ).to(self.device)
        self.model.gradient_checkpointing = config.use_gradient_checkpointing

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.teacher_model = None

        print(f"Device: {self.device} | AMP dtype: {self.amp_dtype if config.use_mixed_precision else 'fp32'}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Elastic exit layers: {config.elastic_exit_layers}")

        if resume:
            path = self._latest_checkpoint()
            if path:
                self.load_checkpoint(path)
                print(f"Resumed checkpoint: {path}")
            else:
                print("No checkpoint found; training from scratch")

    # ---- checkpoints -----------------------------------------------------

    def _latest_checkpoint(self) -> Optional[str]:
        for name in self.CHECKPOINT_PRIORITY:
            path = os.path.join(self.config.output_dir, f"{name}.pt")
            if os.path.exists(path):
                return path
        return None

    def load_checkpoint(self, path: str):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            print(f"  load_checkpoint: missing keys: {missing[:4]}{'...' if len(missing) > 4 else ''}")
        if unexpected:
            print(f"  load_checkpoint: unexpected keys: {unexpected[:4]}{'...' if len(unexpected) > 4 else ''}")

    def save_checkpoint(self, name: str):
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, f"{name}.pt")
        torch.save(
            {"model_state_dict": self.model.state_dict(), "config": asdict(self.config)},
            path,
        )
        print(f"Saved checkpoint: {path}")

    # ---- teacher ----------------------------------------------------------

    def load_teacher(self):
        if self.teacher_model is not None:
            return
        print(f"Loading teacher: {self.config.teacher_model} (4-bit NF4)")
        compute_dtype = self.amp_dtype if self.device.type == "cuda" else torch.float32
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        try:
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.config.teacher_model,
                quantization_config=bnb_config,
                device_map={"": 0} if self.device.type == "cuda" else None,
                torch_dtype=compute_dtype,
            )
        except Exception as e:
            print(f"  bitsandbytes 4-bit load failed ({e}); falling back to fp16 teacher")
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.config.teacher_model,
                torch_dtype=torch.float16,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            if self.device.type == "cuda":
                self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        print("Teacher loaded")

    # ---- training loop ----------------------------------------------------

    def _autocast(self):
        if self.device.type == "cuda" and self.config.use_mixed_precision:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def _scaler(self):
        if (self.device.type == "cuda" and self.config.use_mixed_precision
                and self.amp_dtype == torch.float16):
            try:
                return torch.amp.GradScaler("cuda")
            except (TypeError, AttributeError):
                return torch.cuda.amp.GradScaler()
        return None

    def _make_optimizer_scheduler(self, lr: float):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=self.config.weight_decay,
        )
        warmup = max(1, self.config.warmup_steps)
        total = max(warmup + 1, self.config.max_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / (total - warmup)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _make_loader(self, dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=self.device.type == "cuda",
            collate_fn=partial(collate_fn, pad_id=self.tokenizer.pad_token_id),
            drop_last=len(dataset) >= batch_size,
        )

    def _train_loop(self, phase: str, loader: DataLoader, compute_loss, lr: float,
                    num_epochs: int, log_every: int = 20):
        self.model.train()
        optimizer, scheduler = self._make_optimizer_scheduler(lr)
        scaler = self._scaler()
        accum = self.config.gradient_accumulation_steps
        global_step = 0
        micro = 0
        done = False

        for _ in range(num_epochs):
            if done:
                break
            for batch in loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                with self._autocast():
                    loss = compute_loss(batch)
                loss = loss / accum
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                micro += 1

                if micro % accum == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    if global_step % log_every == 0:
                        print(f"[{phase}] step {global_step}/{self.config.max_steps} "
                              f"loss {loss.item() * accum:.4f} "
                              f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
                    if global_step >= self.config.max_steps:
                        done = True
                        break

        print(f"[{phase}] finished at optimizer step {global_step}")

    # ---- phases ------------------------------------------------------------

    def pretrain(self, data_path: str, num_epochs: int = 1):
        print(f"\n=== Phase 1: Pre-training ({data_path}) ===")
        dataset = AdaptiveSLMDataset(data_path, self.tokenizer, self.config.max_length)
        print(f"  {len(dataset):,} samples")
        loader = self._make_loader(dataset, self.config.batch_size, shuffle=True)
        aux_w = self.config.aux_loss_weight

        def compute_loss(batch):
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out["loss"]
            for logits in out["aux_logits"].values():
                loss = loss + aux_w * causal_cross_entropy(logits, batch["labels"])
            return loss

        self._train_loop("pretrain", loader, compute_loss, self.config.learning_rate, num_epochs)
        self.save_checkpoint("pretrained")

    def distill(self, data_path: str, num_epochs: int = 1):
        print(f"\n=== Phase 2: Knowledge Distillation ({data_path}) ===")
        self.load_teacher()
        dataset = AdaptiveSLMDataset(data_path, self.tokenizer, self.config.max_length)
        print(f"  {len(dataset):,} samples")
        loader = self._make_loader(dataset, self.config.distill_batch_size, shuffle=True)
        kd_loss = DistillationLoss(self.config.distillation_alpha, self.config.temperature)
        aux_w = self.config.aux_loss_weight

        def compute_loss(batch):
            with torch.no_grad():
                t_out = self.teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
            teacher_logits = t_out.logits.float()

            s_out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = kd_loss(s_out["logits"], teacher_logits, batch["labels"])
            for logits in s_out["aux_logits"].values():
                loss = loss + aux_w * kd_loss(logits, teacher_logits, batch["labels"])
            return loss

        self._train_loop("distill", loader, compute_loss, self.config.learning_rate * 0.1, num_epochs)
        self.save_checkpoint("distilled")

    def pakd_finetune(self, data_path: str, num_epochs: int = 1):
        print(f"\n=== Phase 3: PAKD Fine-tuning ({data_path}) ===")
        self.load_teacher()
        dataset = AdaptiveSLMDataset(data_path, self.tokenizer, self.config.max_length)
        print(f"  {len(dataset):,} samples")
        loader = self._make_loader(dataset, self.config.distill_batch_size, shuffle=True)
        pakd_loss = PAKDLoss(
            self.config.distillation_alpha,
            self.config.temperature,
            self.config.pakd_profile_weights,
        )
        aux_w = self.config.aux_loss_weight

        def compute_loss(batch):
            with torch.no_grad():
                t_out = self.teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
            teacher_logits = t_out.logits.float()

            s_out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = pakd_loss(s_out["logits"], teacher_logits, batch["labels"], batch["profile_id"])
            for logits in s_out["aux_logits"].values():
                loss = loss + aux_w * pakd_loss(logits, teacher_logits, batch["labels"], batch["profile_id"])
            return loss

        self._train_loop("pakd", loader, compute_loss, self.config.learning_rate * 0.05, num_epochs)
        self.save_checkpoint("pakd_finetuned")

    # ---- export -------------------------------------------------------------

    def export_gguf(self, hf_root: Optional[str] = None, depths: Tuple[int, ...] = (30, 22, 15)):
        """
        Export elastic-depth family as HuggingFace qwen3-compatible directories
        (model.safetensors + config.json + tokenizer). llama.cpp (>= b5260)
        converts these via convert_hf_to_gguf.py; llama-quantize then makes Q4_K_M.
        """
        print("\n=== Phase 4: Export (HF format) ===")
        if save_safetensors is None:
            raise RuntimeError("safetensors is required for export: pip install safetensors")
        if not self.config.use_qk_norm:
            raise RuntimeError("qwen3 export requires use_qk_norm=True (Qwen3-family convention)")

        hf_root = hf_root or os.path.join(self.config.output_dir, "hf_export")
        state = self.model.state_dict()
        full_depth = self.config.num_layers

        for depth in depths:
            if not (0 < depth <= full_depth):
                raise ValueError(f"depth {depth} out of range (1..{full_depth})")
            out_dir = os.path.join(hf_root, f"depth{depth}")
            os.makedirs(out_dir, exist_ok=True)

            sd = {
                "model.embed_tokens.weight":
                    state["embed_tokens.weight"].detach().to(torch.float16).contiguous(),
            }
            for i in range(depth):
                for key in (
                    "input_layernorm.weight",
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "self_attn.q_norm.weight",
                    "self_attn.k_norm.weight",
                    "post_attention_layernorm.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ):
                    sd[f"model.layers.{i}.{key}"] = \
                        state[f"layers.{i}.{key}"].detach().to(torch.float16).contiguous()

            norm_key = "norm.weight" if depth == full_depth else f"exit_norms.{depth}.weight"
            sd["model.norm.weight"] = state[norm_key].detach().to(torch.float16).contiguous()

            if not self.config.tie_word_embeddings:
                sd["lm_head.weight"] = state["lm_head.weight"].detach().to(torch.float16).contiguous()

            save_safetensors(sd, os.path.join(out_dir, "model.safetensors"))

            model_cfg = {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": self.config.hidden_size,
                "intermediate_size": self.config.intermediate_size,
                "num_attention_heads": self.config.num_attention_heads,
                "num_key_value_heads": self.config.num_key_value_heads,
                "num_hidden_layers": depth,
                "head_dim": self.config.head_dim,
                "vocab_size": self.config.vocab_size,
                "max_position_embeddings": self.config.max_position_embeddings,
                "rms_norm_eps": self.config.rms_norm_eps,
                "rope_theta": self.config.rope_theta,
                "tie_word_embeddings": self.config.tie_word_embeddings,
                "qk_norm": True,
                "attention_bias": False,
                "bias": False,
                "hidden_act": "silu",
                "use_cache": True,
                "torch_dtype": "float16",
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "pad_token_id": 151643,
            }
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(model_cfg, f, indent=2)

            generation_cfg = {
                "bos_token_id": 151643,
                "eos_token_id": [151645, 151643],
                "pad_token_id": 151643,
                "do_sample": True,
                "top_p": 0.9,
                "temperature": 0.7,
            }
            with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
                json.dump(generation_cfg, f, indent=2)

            self.tokenizer.save_pretrained(out_dir)

            n_params = sum(t.numel() for t in sd.values())
            print(f"  depth{depth}: {n_params:,} params -> {out_dir}")

        print("\nConvert + quantize with llama.cpp:")
        print("  python llama.cpp/convert_hf_to_gguf.py checkpoints/hf_export/depth30 "
              "--outfile ../models/adaptive_slm-30l-f16.gguf")
        print("  ./llama.cpp/build/bin/llama-quantize ../models/adaptive_slm-30l-f16.gguf "
              "../models/adaptive_slm-30l-q4_k_m.gguf Q4_K_M")
        print("(repeat for depth22/depth15; or keep only the PAKD profile depths you need)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AdaptiveSLM v2 training pipeline")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["pretrain", "distill", "pakd", "export", "all"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--distill-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--teacher", type=str, default=None,
                        help="Teacher model for distill/pakd (default: Qwen3-4B-Instruct-2507)")
    parser.add_argument("--export-depths", type=str, default="30,22,15")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints")
    parser.add_argument("--fp32", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    config = TrainingConfig()
    config.output_dir = args.output_dir
    config.data_path = args.data_dir
    config.batch_size = args.batch_size
    config.distill_batch_size = args.distill_batch_size
    config.max_length = args.max_length
    config.max_steps = args.max_steps
    config.learning_rate = args.lr
    if args.teacher:
        config.teacher_model = args.teacher
    if args.fp32:
        config.use_mixed_precision = False

    print("AdaptiveSLM v2 Training Pipeline")
    print("=" * 50)

    trainer = AdaptiveSLMTrainer(config, resume=not args.fresh)

    def data_file(name):
        return os.path.join(args.data_dir, name)

    if args.phase in ("pretrain", "all"):
        path = data_file("pretrain_corpus.jsonl")
        if os.path.exists(path):
            trainer.pretrain(path, num_epochs=args.num_epochs)
        else:
            print(f"Skipping pretrain: {path} not found")

    if args.phase in ("distill", "all"):
        path = data_file("distill_data.jsonl")
        if os.path.exists(path):
            trainer.distill(path, num_epochs=args.num_epochs)
        else:
            print(f"Skipping distill: {path} not found")

    if args.phase in ("pakd", "all"):
        path = data_file("pakd_data.jsonl")
        if os.path.exists(path):
            trainer.pakd_finetune(path, num_epochs=args.num_epochs)
        else:
            print(f"Skipping pakd: {path} not found")

    if args.phase in ("export", "all"):
        depths = tuple(int(d) for d in args.export_depths.split(","))
        trainer.export_gguf(depths=depths)

    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()
