# API Reference

## Rust API (Recommended)

Add to `Cargo.toml`:
```toml
[dependencies]
adaptive_slm = { path = "../adaptive-slm/rust-wrapper" }
```

### `AdaptiveSLM`

The main client struct.

#### `new`
```rust
pub async fn new(
    model_path: &str, 
    db_path: &str, 
    tavily_api_key: Option<&str>
) -> Result<Self>
```
Initializes the model, cache database, and optional search client.

#### `generate`
```rust
pub async fn generate(
    &mut self, 
    prompt: &str, 
    params: &GenerationParams, 
    use_search: bool
) -> Result<String>
```
Generates a response.
- Checks **SCPD Cache** first.
- If `use_search` is true, performs **Tavily** search and augments context.
- Runs inference via C++ core.
- Stores result in cache.

#### `set_profile`
```rust
pub fn set_profile(&mut self, profile: UserProfile)
```
Sets the active user profile for personalization.

---

## C++ Core API

Header: `core/include/adaptive_slm.h`

### `aslm_init`
```c
aslm_context* aslm_init(const aslm_init_params* params);
```
Initializes the GGML context and loads the model.

### `aslm_generate`
```c
int32_t aslm_generate(
    aslm_context* ctx,
    const char* prompt,
    const aslm_gen_params* params,
    char* output,
    size_t output_size
);
```
Blocking generation call. Returns cached token count.

### `aslm_update_device_state`
```c
void aslm_update_device_state(aslm_context* ctx, const aslm_device_state* state);
```
Feeds telemetry data (RAM, Battery) to the **ACC** module to adjust internal buffers.
