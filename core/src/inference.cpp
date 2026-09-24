/**
 * AdaptiveSLM Inference Engine
 * Real model inference via llama.cpp with ACC integration
 */

#include "adaptive_slm.h"
#include "context_adapt.h"
#include "user_profile.h"

#include <llama.h>
#include <ggml.h>

#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <thread>

namespace aslm {

struct Context {
    llama_model* model = nullptr;
    llama_context* llama_ctx = nullptr;
    llama_sampler* sampler = nullptr;

    aslm_init_params init_params;

    std::unique_ptr<ContextAdapter> acc;
    int32_t current_context_size = 512;

    const UserProfile* user_profile = nullptr;

    uint64_t peak_memory = 0;
    bool is_initialized = false;
};

static llama_sampler* create_sampler(const aslm_gen_params* params) {
    auto* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
        64,                       // penalty_last_n: look back 64 tokens
        params->repeat_penalty,   // penalty_repeat
        0.0f,                     // penalty_freq
        0.0f                      // penalty_present
    ));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params->top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params->top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(params->temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    return smpl;
}

} // namespace aslm

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

aslm_init_params aslm_default_init_params(void) {
    return aslm_init_params{
        .model_path = nullptr,
        .n_threads = 0,
        .max_context = 2048,
        .use_mmap = true,
        .verbose = false
    };
}

aslm_context* aslm_init(const aslm_init_params* params) {
    if (!params || !params->model_path) {
        std::cerr << "[ASLM] Error: model_path is required\n";
        return nullptr;
    }

    auto* ctx = new aslm::Context();
    ctx->init_params = *params;

    // Initialize ACC
    aslm::ACCConfig acc_config;
    acc_config.max_context = params->max_context;
    ctx->acc = std::make_unique<aslm::ContextAdapter>(acc_config);
    ctx->current_context_size = acc_config.base_context;

    // Load model via llama.cpp
    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = params->use_mmap;

    if (params->verbose) {
        std::cout << "[ASLM] Loading model: " << params->model_path << "\n";
    }

    ctx->model = llama_model_load_from_file(params->model_path, mparams);
    if (!ctx->model) {
        std::cerr << "[ASLM] Error: failed to load model from " << params->model_path << "\n";
        delete ctx;
        return nullptr;
    }

    // Create inference context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = params->max_context;
    cparams.n_threads = params->n_threads > 0
        ? params->n_threads
        : static_cast<int32_t>(std::thread::hardware_concurrency());
    cparams.n_threads_batch = cparams.n_threads;

    ctx->llama_ctx = llama_init_from_model(ctx->model, cparams);
    if (!ctx->llama_ctx) {
        std::cerr << "[ASLM] Error: failed to create llama context\n";
        llama_model_free(ctx->model);
        delete ctx;
        return nullptr;
    }

    ctx->is_initialized = true;

    if (params->verbose) {
        std::cout << "[ASLM] Model loaded successfully\n";
        std::cout << "[ASLM] Context size: " << params->max_context << " tokens\n";
        std::cout << "[ASLM] Threads: " << cparams.n_threads << "\n";
        std::cout << "[ASLM] Vocab size: " << llama_vocab_n_tokens(llama_model_get_vocab(ctx->model)) << "\n";
    }

    return reinterpret_cast<aslm_context*>(ctx);
}

void aslm_free(aslm_context* handle) {
    if (!handle) return;

    auto* ctx = reinterpret_cast<aslm::Context*>(handle);

    if (ctx->sampler) {
        llama_sampler_free(ctx->sampler);
    }
    if (ctx->llama_ctx) {
        llama_free(ctx->llama_ctx);
    }
    if (ctx->model) {
        llama_model_free(ctx->model);
    }

    delete ctx;
}

int32_t aslm_generate(
    aslm_context* handle,
    const char* prompt,
    const aslm_gen_params* params,
    char* output,
    size_t output_size
) {
    if (!handle || !prompt || !output || output_size == 0 || !params) {
        return -1;
    }

    auto* ctx = reinterpret_cast<aslm::Context*>(handle);
    if (!ctx->is_initialized || !ctx->model || !ctx->llama_ctx) {
        return -1;
    }

    // Build prompt in ChatML format (Qwen2/Qwen2.5 native format)
    // System message carries profile modifier; user message carries the actual prompt
    std::string full_prompt;
    if (ctx->user_profile) {
        std::string modifier = ctx->user_profile->getPromptModifier();
        if (!modifier.empty()) {
            full_prompt += "<|im_start|>system\n" + modifier + "<|im_end|>\n";
        }
    }
    full_prompt += "<|im_start|>user\n";
    full_prompt += prompt;
    full_prompt += "<|im_end|>\n<|im_start|>assistant\n";

    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);

    // Tokenize the prompt
    int n_prompt_max = full_prompt.size() + 128;
    std::vector<llama_token> prompt_tokens(n_prompt_max);
    int n_prompt_tokens = llama_tokenize(
        vocab,
        full_prompt.c_str(),
        full_prompt.size(),
        prompt_tokens.data(),
        n_prompt_max,
        true,   // add_special (BOS)
        true    // parse_special
    );

    if (n_prompt_tokens < 0) {
        std::cerr << "[ASLM] Error: tokenization failed\n";
        return -1;
    }
    prompt_tokens.resize(n_prompt_tokens);

    // Check prompt fits in context
    int n_ctx = llama_n_ctx(ctx->llama_ctx);
    int effective_ctx = std::min(n_ctx, ctx->current_context_size);
    if (n_prompt_tokens >= effective_ctx) {
        std::cerr << "[ASLM] Error: prompt too long (" << n_prompt_tokens
                  << " tokens) for context (" << effective_ctx << ")\n";
        return -1;
    }

    // Clear KV cache for fresh generation
    llama_kv_self_clear(ctx->llama_ctx);

    // Free previous sampler if any, create new one with current params
    if (ctx->sampler) {
        llama_sampler_free(ctx->sampler);
    }
    ctx->sampler = aslm::create_sampler(params);

    // Process prompt tokens in a single batch
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt_tokens);
    if (llama_decode(ctx->llama_ctx, batch) != 0) {
        std::cerr << "[ASLM] Error: prompt decode failed\n";
        return -1;
    }

    // Autoregressive generation
    std::string result;
    int32_t n_generated = 0;
    int32_t max_tokens = std::min(params->max_tokens, effective_ctx - n_prompt_tokens);

    for (int32_t i = 0; i < max_tokens; ++i) {
        llama_token new_token = llama_sampler_sample(ctx->sampler, ctx->llama_ctx, -1);

        // Check for end of generation
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        // Detokenize the token
        char piece[128];
        int n_piece = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, true);
        if (n_piece > 0) {
            result.append(piece, n_piece);
        }

        n_generated++;

        // Decode the new token for next iteration
        llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx->llama_ctx, next_batch) != 0) {
            std::cerr << "[ASLM] Error: decode failed at token " << i << "\n";
            break;
        }
    }

    // Copy result to output buffer
    size_t copy_len = std::min(result.size(), output_size - 1);
    std::memcpy(output, result.c_str(), copy_len);
    output[copy_len] = '\0';

    return n_generated;
}

void aslm_update_device_state(aslm_context* handle, const aslm_device_state* state) {
    if (!handle || !state) return;

    auto* ctx = reinterpret_cast<aslm::Context*>(handle);

    if (ctx->acc) {
        ctx->current_context_size = ctx->acc->getSmoothedContextSize(*state);
    }
}

int32_t aslm_get_effective_context_size(const aslm_context* handle) {
    if (!handle) return 512;

    const auto* ctx = reinterpret_cast<const aslm::Context*>(handle);
    return ctx->current_context_size;
}

void aslm_set_user_profile(aslm_context* handle, const aslm_user_profile* profile) {
    if (!handle) return;

    auto* ctx = reinterpret_cast<aslm::Context*>(handle);
    ctx->user_profile = reinterpret_cast<const aslm::UserProfile*>(profile);
}

uint64_t aslm_get_memory_usage(const aslm_context* handle) {
    if (!handle) return 0;

    const auto* ctx = reinterpret_cast<const aslm::Context*>(handle);
    if (!ctx->llama_ctx) return 0;

    // Report actual memory used by llama.cpp context
    return llama_state_get_size(ctx->llama_ctx);
}

uint64_t aslm_get_peak_memory_usage(const aslm_context* handle) {
    if (!handle) return 0;

    const auto* ctx = reinterpret_cast<const aslm::Context*>(handle);
    if (!ctx->llama_ctx) return 0;

    size_t state_size = llama_state_get_size(ctx->llama_ctx);
    size_t model_size = llama_model_size(ctx->model);
    return state_size + model_size;
}

int32_t aslm_generate_stream(
    aslm_context* handle,
    const char* prompt,
    const aslm_gen_params* params,
    aslm_token_callback callback,
    void* user_data
) {
    if (!handle || !prompt || !params || !callback) {
        return -1;
    }

    auto* ctx = reinterpret_cast<aslm::Context*>(handle);
    if (!ctx->is_initialized || !ctx->model || !ctx->llama_ctx) {
        return -1;
    }

    std::string full_prompt;
    if (ctx->user_profile) {
        std::string modifier = ctx->user_profile->getPromptModifier();
        if (!modifier.empty()) {
            full_prompt += "<|im_start|>system\n" + modifier + "<|im_end|>\n";
        }
    }
    full_prompt += "<|im_start|>user\n";
    full_prompt += prompt;
    full_prompt += "<|im_end|>\n<|im_start|>assistant\n";

    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);

    int n_prompt_max = full_prompt.size() + 128;
    std::vector<llama_token> prompt_tokens(n_prompt_max);
    int n_prompt_tokens = llama_tokenize(
        vocab, full_prompt.c_str(), full_prompt.size(),
        prompt_tokens.data(), n_prompt_max, true, true
    );

    if (n_prompt_tokens < 0) return -1;
    prompt_tokens.resize(n_prompt_tokens);

    int n_ctx = llama_n_ctx(ctx->llama_ctx);
    int effective_ctx = std::min(n_ctx, ctx->current_context_size);
    if (n_prompt_tokens >= effective_ctx) return -1;

    llama_kv_self_clear(ctx->llama_ctx);

    if (ctx->sampler) llama_sampler_free(ctx->sampler);
    ctx->sampler = aslm::create_sampler(params);

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt_tokens);
    if (llama_decode(ctx->llama_ctx, batch) != 0) return -1;

    int32_t n_generated = 0;
    int32_t max_tokens = std::min(params->max_tokens, effective_ctx - n_prompt_tokens);

    for (int32_t i = 0; i < max_tokens; ++i) {
        llama_token new_token = llama_sampler_sample(ctx->sampler, ctx->llama_ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) break;

        char piece[128];
        int n_piece = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, true);
        if (n_piece > 0) {
            piece[n_piece] = '\0';
            n_generated++;

            if (!callback(piece, n_generated, user_data)) {
                break;  // User requested early stop
            }
        }

        llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx->llama_ctx, next_batch) != 0) break;
    }

    return n_generated;
}

}
