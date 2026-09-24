# AdaptiveSLM Training Pipeline

"""
Training pipeline for AdaptiveSLM that beats competitors.

Key Strategy:
1. Custom architecture (MobileLLM-style: deep & thin)
2. Knowledge distillation from Qwen-7B teacher
3. High-quality data curation
4. Profile-Aware Knowledge Distillation (PAKD) - our novel contribution

Training Phases:
- Phase 1: Pre-training on filtered web corpus
- Phase 2: Knowledge distillation from larger teacher
- Phase 3: Domain-specific fine-tuning with PAKD
- Phase 4: Quantization-aware training + GGUF export
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from datasets import load_dataset
from typing import Optional, Dict, Any
import json
import os

# ============================================================================
# Configuration
# ============================================================================

class TrainingConfig:
    """Training configuration for AdaptiveSLM"""
    
    # Model Architecture (MobileLLM-style: deep and thin)
    vocab_size: int = 151665  # Qwen2.5 tokenizer vocabulary size
    hidden_size: int = 576          # Thin: smaller hidden dim
    num_layers: int = 30            # Deep: more layers (vs typical 12)
    num_attention_heads: int = 9
    num_key_value_heads: int = 3    # Grouped-Query Attention (GQA)
    intermediate_size: int = 1536
    max_position_embeddings: int = 2048
    
    # Training Hyperparameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    
    # Knowledge Distillation
    teacher_model: str = "Qwen/Qwen2.5-7B-Instruct"
    distillation_alpha: float = 0.5     # Balance between CE and KD loss
    temperature: float = 2.0
    
    # PAKD (Profile-Aware KD) - Our Novel Contribution
    pakd_enabled: bool = True
    pakd_profiles: list = ["beginner", "intermediate", "expert"]
    pakd_domain_weights: dict = None
    
    # Memory Optimization
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True    # bfloat16
    use_flash_attention: bool = True
    
    # Quantization-Aware Training
    qat_enabled: bool = True
    qat_bits: int = 4
    
    # Paths
    output_dir: str = "./checkpoints"
    data_path: str = "./data"
    
    def __post_init__(self):
        if self.pakd_domain_weights is None:
            self.pakd_domain_weights = {
                "general": 1.0,
                "coding": 1.2,
                "science": 1.1,
                "creative": 0.9
            }

# ============================================================================
# Custom Architecture (MobileLLM-style)
# ============================================================================

class AdaptiveSLMConfig:
    """Configuration matching MobileLLM architecture"""
    
    def __init__(self, config: TrainingConfig):
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.intermediate_size = config.intermediate_size
        self.max_position_embeddings = config.max_position_embeddings
        self.rms_norm_eps = 1e-6
        self.use_embedding_sharing = True  # MobileLLM innovation
        self.use_block_wise_sharing = True  # MobileLLM-LS innovation

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) for memory efficiency"""
    
    def __init__(self, config: AdaptiveSLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat KV for grouped attention
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

class SwiGLU(nn.Module):
    """SwiGLU activation (better than ReLU for transformers)"""
    
    def __init__(self, config: AdaptiveSLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class AdaptiveSLMBlock(nn.Module):
    """Single transformer block with GQA and SwiGLU"""
    
    def __init__(self, config: AdaptiveSLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class AdaptiveSLMModel(nn.Module):
    """
    AdaptiveSLM: Custom architecture for ultra-low-resource devices
    
    Novel features:
    - MobileLLM-style deep & thin architecture (30 layers, 576 hidden)
    - Grouped-Query Attention (GQA) for memory efficiency
    - Embedding sharing (input/output)
    - Block-wise weight sharing option
    """
    
    def __init__(self, config: AdaptiveSLMConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            AdaptiveSLMBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Language model head (shares weights with embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.use_embedding_sharing:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
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
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask
        if attention_mask is not None:
            causal_mask = self._make_causal_mask(batch_size, seq_len, hidden_states.device)
            attention_mask = causal_mask + attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            attention_mask = self._make_causal_mask(batch_size, seq_len, hidden_states.device)
        
        # Forward through transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final norm and LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {"loss": loss, "logits": logits}
    
    def _make_causal_mask(self, batch_size: int, seq_len: int, device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# Knowledge Distillation
# ============================================================================

class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation loss combining:
    - Cross-entropy with ground truth
    - KL divergence with teacher logits
    """
    
    def __init__(self, alpha: float = 0.5, temperature: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int
    ) -> torch.Tensor:
        # Cross-entropy loss with ground truth
        ce_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100
        )
        
        # KL divergence with teacher (soft targets)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kd_loss = F.kl_div(
            student_probs.view(-1, vocab_size),
            teacher_probs.view(-1, vocab_size),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        
        # Combined loss
        return (1 - self.alpha) * ce_loss + self.alpha * kd_loss

# ============================================================================
# Profile-Aware Knowledge Distillation (PAKD) - Our Novel Contribution
# ============================================================================

class PAKDLoss(nn.Module):
    """
    Profile-Aware Knowledge Distillation (PAKD)
    
    NOVEL CONTRIBUTION:
    Weights the distillation loss based on user profile relevance.
    Different profile types receive different emphasis during training.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 2.0,
        profile_weights: Dict[str, float] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.profile_weights = profile_weights or {
            "beginner": 1.2,      # Boost simple explanations
            "intermediate": 1.0,   # Standard weight
            "expert": 0.8,        # Less focus on jargon
            "general": 1.0
        }
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        profile_ids: torch.Tensor,  # Which profile each sample belongs to
        vocab_size: int
    ) -> torch.Tensor:
        batch_size = student_logits.size(0)
        
        # Base cross-entropy loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction="none"
        ).view(batch_size, -1).mean(dim=1)
        
        # KL divergence with teacher
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kd_loss = F.kl_div(
            student_probs.view(batch_size, -1, vocab_size),
            teacher_probs.view(batch_size, -1, vocab_size),
            reduction="none"
        ).sum(dim=-1).mean(dim=1) * (self.temperature ** 2)
        
        # Apply profile-based weighting
        weights = torch.ones(batch_size, device=student_logits.device)
        for profile_id, weight in enumerate(self.profile_weights.values()):
            mask = profile_ids == profile_id
            weights[mask] = weight
        
        # Weighted combination
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
        weighted_loss = (total_loss * weights).mean()
        
        return weighted_loss

# ============================================================================
# Data Preparation
# ============================================================================

class AdaptiveSLMDataset(Dataset):
    """Dataset with profile annotations for PAKD"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        profile_enabled: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.profile_enabled = profile_enabled
        
        # Load data
        self.data = self._load_data(data_path)
    
    def _load_data(self, path: str):
        # Support multiple formats
        if path.endswith(".jsonl"):
            data = []
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        elif os.path.isdir(path):
            # Load from HuggingFace datasets cache
            return load_dataset("json", data_dir=path)["train"]
        else:
            raise ValueError(f"Unsupported data format: {path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        text = item.get("text", item.get("content", ""))
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone()
        }
        
        # Add profile ID for PAKD
        if self.profile_enabled:
            profile = item.get("profile", "general")
            profile_map = {"beginner": 0, "intermediate": 1, "expert": 2, "general": 3}
            result["profile_id"] = profile_map.get(profile, 3)
        
        return result

# ============================================================================
# Training Loop
# ============================================================================

class AdaptiveSLMTrainer:
    """
    Complete training pipeline for AdaptiveSLM
    
    Stages:
    1. Pre-training from scratch
    2. Knowledge distillation from Qwen-7B
    3. PAKD fine-tuning
    4. Quantization-aware training
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        model_config = AdaptiveSLMConfig(config)
        self.model = AdaptiveSLMModel(model_config).to(self.device)
        
        # Initialize tokenizer (from base model)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        
        # Initialize teacher model for distillation
        self.teacher_model = None
        
        print(f"Model Parameters: {self.model.count_parameters():,}")
        print(f"Trainable Parameters: {self.model.count_trainable_parameters():,}")
    
    def load_teacher(self):
        """Load teacher model for knowledge distillation"""
        print(f"Loading teacher model: {self.config.teacher_model}")
        
        # Use 4-bit quantization for teacher to save memory
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model,
            quantization_config=bnb_config,
            device_map={"": self.device},  # Force to single device
            torch_dtype=torch.bfloat16
        )
        self.teacher_model.eval()
        
        # Disable gradient for teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        print("Teacher model loaded")
    
    def pretrain(self, data_path: str, num_epochs: int = 1):
        """Phase 1: Pre-training from scratch"""
        print("\n=== Phase 1: Pre-training ===")
        
        dataset = AdaptiveSLMDataset(data_path, self.tokenizer, profile_enabled=False)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(dataloader) * num_epochs
        )
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.config.use_mixed_precision else None
        
        self.model.train()
        global_step = 0
        
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = outputs["loss"]
                
                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    scheduler.step()
                
                global_step += 1
                
                if global_step % 100 == 0:
                    print(f"Step {global_step}, Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f}")
                
                if global_step >= self.config.max_steps:
                    break
        
        self.save_checkpoint("pretrained")
    
    def distill(self, data_path: str, num_epochs: int = 1):
        """Phase 2: Knowledge Distillation from teacher"""
        print("\n=== Phase 2: Knowledge Distillation ===")
        
        if self.teacher_model is None:
            self.load_teacher()
        
        dataset = AdaptiveSLMDataset(data_path, self.tokenizer, profile_enabled=False)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size // 2)
        
        kd_loss_fn = KnowledgeDistillationLoss(
            alpha=self.config.distillation_alpha,
            temperature=self.config.temperature
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate * 0.1,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(dataloader) * num_epochs
        )
        
        scaler = torch.cuda.amp.GradScaler() if self.config.use_mixed_precision else None
        
        self.model.train()
        global_step = 0
        
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get teacher logits (no grad)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )
                    teacher_logits = teacher_outputs.logits
                
                # Get student logits
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    student_outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )
                    student_logits = student_outputs["logits"]
                    
                    # Compute distillation loss
                    loss = kd_loss_fn(
                        student_logits[:, :-1],
                        teacher_logits[:, :-1],
                        batch["labels"][:, 1:],
                        self.config.vocab_size
                    )
                
                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    scheduler.step()
                
                global_step += 1
                
                if global_step % 50 == 0:
                    print(f"Distillation Step {global_step}, Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f}")
                
                if global_step >= self.config.max_steps:
                    break
        
        self.save_checkpoint("distilled")
    
    def pakd_finetune(self, data_path: str, num_epochs: int = 1):
        """Phase 3: Profile-Aware Knowledge Distillation (PAKD)"""
        print("\n=== Phase 3: PAKD Fine-tuning ===")
        
        if self.teacher_model is None:
            self.load_teacher()
        
        dataset = AdaptiveSLMDataset(data_path, self.tokenizer, profile_enabled=True)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size // 2)
        
        pakd_loss_fn = PAKDLoss(
            alpha=self.config.distillation_alpha,
            temperature=self.config.temperature,
            profile_weights=self.config.pakd_domain_weights
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate * 0.05,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(dataloader) * num_epochs
        )
        
        scaler = torch.cuda.amp.GradScaler() if self.config.use_mixed_precision else None
        
        self.model.train()
        global_step = 0
        
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                profile_ids = batch["profile_id"].to(self.device)
                
                # Get teacher logits
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    teacher_logits = teacher_outputs.logits
                
                # Get student logits
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    student_outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    student_logits = student_outputs["logits"]
                    
                    # PAKD loss with profile weighting
                    loss = pakd_loss_fn(
                        student_logits[:, :-1],
                        teacher_logits[:, :-1],
                        labels[:, 1:],
                        profile_ids,
                        self.config.vocab_size
                    )
                
                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    scheduler.step()
                
                global_step += 1
                
                if global_step % 50 == 0:
                    print(f"PAKD Step {global_step}, Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f}")
                
                if global_step >= self.config.max_steps:
                    break
        
        self.save_checkpoint("pakd_finetuned")
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        path = os.path.join(self.config.output_dir, f"{name}.pt")
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__
        }, path)
        
        print(f"Saved checkpoint: {path}")
    
    def export_gguf(self, output_path: str):
        """Export model to GGUF format for inference"""
        print("\n=== Exporting to GGUF ===")

        # Save as HuggingFace format for conversion
        hf_path = os.path.join(self.config.output_dir, "hf_export")
        os.makedirs(hf_path, exist_ok=True)

        # Save model weights in HF format
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(hf_path, "pytorch_model.bin"))

        # Save config.json for convert script
        model_config = {
            "architectures": ["AdaptiveSLMForCausalLM"],
            "hidden_size": self.config.hidden_size,
            "intermediate_size": self.config.intermediate_size,
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": self.config.num_key_value_heads,
            "num_hidden_layers": self.config.num_layers,
            "vocab_size": self.config.vocab_size,
            "max_position_embeddings": self.config.max_position_embeddings,
            "model_type": "adaptive_slm",
            "torch_dtype": "float16",
        }
        with open(os.path.join(hf_path, "config.json"), "w") as f:
            json.dump(model_config, f, indent=2)

        print(f"HuggingFace export saved to: {hf_path}")
        print(f"Convert to GGUF with:")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {hf_path} --outfile {output_path}")
        print(f"Quantize with:")
        print(f"  ./llama.cpp/build/bin/llama-quantize {output_path} {output_path.replace('.gguf', '-q4_k_m.gguf')} Q4_K_M")

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training entry point"""
    
    # Configuration
    config = TrainingConfig()
    
    # Initialize trainer
    trainer = AdaptiveSLMTrainer(config)
    
    # Training phases
    print("Starting AdaptiveSLM Training Pipeline")
    print("=" * 50)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all",
                        choices=["pretrain", "distill", "pakd", "export", "all"],
                        help="Which training phase to run")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-model", type=str, default="./models/adaptive_slm.gguf")
    args = parser.parse_args()

    data_dir = args.data_dir

    if args.phase in ("pretrain", "all"):
        print("\n--- Phase 1: Pre-training ---")
        corpus_path = os.path.join(data_dir, "pretrain_corpus.jsonl")
        if os.path.exists(corpus_path):
            trainer.pretrain(corpus_path, num_epochs=1)
        else:
            print(f"  Skipping: {corpus_path} not found")

    if args.phase in ("distill", "all"):
        print("\n--- Phase 2: Knowledge Distillation ---")
        distill_path = os.path.join(data_dir, "distill_data.jsonl")
        if os.path.exists(distill_path):
            trainer.distill(distill_path, num_epochs=1)
        else:
            print(f"  Skipping: {distill_path} not found")

    if args.phase in ("pakd", "all"):
        print("\n--- Phase 3: PAKD Fine-tuning (novel contribution) ---")
        pakd_path = os.path.join(data_dir, "pakd_data.jsonl")
        if os.path.exists(pakd_path):
            trainer.pakd_finetune(pakd_path, num_epochs=1)
        else:
            print(f"  Skipping: {pakd_path} not found")

    if args.phase in ("export", "all"):
        print("\n--- Phase 4: Export to GGUF ---")
        trainer.export_gguf(args.output_model)

    print("\nTraining pipeline complete.")

if __name__ == "__main__":
    main()
