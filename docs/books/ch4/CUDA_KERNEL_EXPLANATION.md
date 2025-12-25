# How `vllm serve --model gpt2` Translates to CUDA Kernels: Complete Guide

## Overview

This document explains the complete journey from a user's `vllm serve` command and client prompt requests all the way down to CUDA kernel execution on the GPU.

---

## Part 1: Server Startup (`vllm serve --model gpt2`)

### The 7-Stage Server Startup

```
vllm serve --model gpt2
    ↓ (Stage 1)
Parse CLI arguments
    ↓ (Stage 2)
ServeSubcommand.cmd()
    ↓ (Stage 3)
run_server(args) - Setup HTTP socket
    ↓ (Stage 4)
AsyncLLMEngine initialization
    ↓ (Stage 5)
Load model & allocate GPU memory
    ↓ (Stage 6)
Initialize attention backends (FlashAttention)
    ↓ (Stage 7)
Start Uvicorn HTTP server on 0.0.0.0:8000
    ↓
Ready to accept client requests
```

### Stage-by-Stage Breakdown

#### **Stage 1: CLI Parsing** (< 1ms)
**Files:** `vllm/entrypoints/cli/main.py`

```python
def main():
    # Parse "serve" subcommand
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers()
    
    # Register all CLI subcommands
    for cmd_module in [cli.serve, cli.benchmark, ...]:
        cmd = cmd_module.cmd_init()
        cmd.subparser_init(subparsers)
    
    args = parser.parse_args()  # args.subparser = "serve", args.model = "gpt2"
    args.dispatch_function(args)  # → ServeSubcommand.cmd(args)
```

#### **Stage 2: Serve Subcommand Routing**
**Files:** `vllm/entrypoints/cli/serve.py`

```python
class ServeSubcommand(CLISubcommand):
    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # Handle model positional argument
        if hasattr(args, "model_tag"):
            args.model = args.model_tag  # "gpt2"
        
        # Decide execution mode
        if args.headless or args.api_server_count < 1:
            run_headless(args)  # Engine-only (no HTTP)
        else:
            if args.api_server_count > 1:
                run_multi_api_server(args)  # Multiple servers
            else:
                # Standard case: single API server
                uvloop.run(run_server(args))
```

#### **Stage 3: HTTP Socket Setup** (< 10ms)
**Files:** `vllm/entrypoints/openai/api_server.py`

```python
async def run_server(args: AsyncEngineArgs):
    # 1. Create TCP socket
    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket.bind(("0.0.0.0", 8000))  # Listen on port 8000
    socket.listen(128)  # Queue up to 128 connections
    
    # 2. Prepare async HTTP server
    app = build_app(args)  # FastAPI app with OpenAI-compatible endpoints
    
    # 3. Initialize engine (async)
    engine_client = await build_async_engine_client(args)
    
    # 4. Start Uvicorn server
    serve_http(app, host="0.0.0.0", port=8000)
```

#### **Stage 4-5: AsyncLLMEngine Initialization**
**Files:** `vllm/engine/async_llm_engine.py`, `vllm/v1/engine/async_llm.py`

```python
class AsyncLLMEngine:
    async def __init__(self, args: AsyncEngineArgs):
        # 1. Create model config from HuggingFace
        model_config = ModelConfig.from_pretrained("gpt2")
        # Shape: [vocab_size=50257, hidden_size=768, num_layers=12, ...]
        
        # 2. Create cache config
        cache_config = CacheConfig(
            block_size=16,           # 16 tokens per block
            num_gpu_blocks=8000,     # GPU KV cache blocks
            num_cpu_blocks=0         # CPU offloading
        )
        
        # 3. Load model onto GPU
        self.model = GPT2Model.from_pretrained("gpt2")
        self.model.to("cuda:0")  # ← GPU memory allocation happens here
```

#### **Stage 6: GPU Memory Allocation**
**Files:** `vllm/v1/attention/backends/flash_attn.py`

```python
class FlashAttentionBackend:
    @staticmethod
    def create_kv_cache(
        num_gpu_blocks: int,      # 8000 blocks
        block_size: int,           # 16 tokens/block
        num_heads: int,            # 12 heads (GPT2)
        head_size: int,            # 64 dimensions
    ) -> torch.Tensor:
        # Shape: [num_blocks, block_size, num_heads, head_size]
        #      = [8000, 16, 12, 64]
        #      = ~98 MB per layer
        
        kv_cache = torch.zeros(
            (num_gpu_blocks, block_size, num_heads, head_size),
            dtype=torch.float16,  # 2 bytes per value
            device="cuda:0"
        )
        return kv_cache
```

#### **Stage 7: Server Ready**

The server is now listening on `http://0.0.0.0:8000` and ready to accept requests!

---

## Part 2: Client Request Flow

### Typical Client Request

```bash
# User sends a completion request via HTTP
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "Today is a",
    "max_tokens": 10,
    "temperature": 0.7
  }'
```

### Request Processing Pipeline

```
HTTP POST /v1/completions
    ↓
Request Handler (openai/serving_completion.py)
    ↓
Parse JSON → CompletionRequest object
    ↓
Engine.add_request()
    → Tokenize prompt: "Today is a" → [2911, 318, 257]
    → Create RequestGroup with metadata
    → Add to request queue
    ↓
Engine loops (generate_request_output):
    ↓ (While not finished)
    Process batch & schedule requests
    ↓
Execute model forward pass (on GPU)
    ↓
Sample tokens
    ↓
Stream response to client
```

---

## Part 3: The Forward Pass (GPU Execution)

### Model Forward Pass for GPT2

When the engine processes a batch, it executes:

```python
# vllm/model_executor/models/gpt2.py (or v1 variant)

class GPT2ForCausalLM(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,           # [batch_size, seq_len]
        positions: torch.Tensor,           # [batch_size, seq_len]
        kv_caches: list[torch.Tensor],     # Pre-allocated GPU buffers
        attn_metadata: FlashAttentionMetadata,  # ← CRITICAL
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs from the prompt
                       Example: [[2911, 318, 257]] for "Today is a"
            positions: Absolute positions in sequence
                       Example: [[0, 1, 2]]
            kv_caches: List of 2 tensors per layer (K-cache, V-cache)
                       Shape: [num_blocks, block_size, num_heads, head_size]
            attn_metadata: Contains:
                - block_tables: [batch_size, max_blocks_per_seq]
                - seq_lens: [batch_size]
                - cu_seqlens: Cumulative sequence lengths
                - causal: Whether to apply causal masking
        """
        
        # Step 1: Embed tokens
        hidden_states = self.embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        # Shape: [1, 3, 768] for our example
        
        # Step 2: Pass through 12 transformer layers
        for layer_idx in range(12):
            # 2a. Layer norm
            attn_in = self.norm(hidden_states)
            
            # 2b. ATTENTION LAYER (← Most compute-intensive)
            attn_out = self.attn_layers[layer_idx].forward(
                hidden_states=attn_in,
                kv_cache=kv_caches[layer_idx],  # ← Pre-allocated GPU buffers
                attn_metadata=attn_metadata,    # ← Block table & sequence info
            )
            
            # 2c. Residual connection
            hidden_states = attn_out + hidden_states
            
            # 2d. MLP (Feed Forward)
            mlp_out = self.mlp_layers[layer_idx](hidden_states)
            hidden_states = mlp_out + hidden_states
        
        # Step 3: Final layer norm
        hidden_states = self.final_norm(hidden_states)
        
        # Step 4: Project to vocabulary
        logits = hidden_states @ self.lm_head.weight  # [batch_size, seq_len, vocab_size]
        # Shape: [1, 3, 50257] for our example
        
        return logits  # [1, 3, 50257]
```

### The Critical Part: Attention Layer

The **attention layer** consumes ~50% of the compute for LLMs!

```python
# vllm/v1/attention/backends/flash_attn.py

class FlashAttentionImpl(AttentionImpl):
    def forward(
        self,
        layer,
        query: torch.Tensor,           # [batch_size * seq_len, num_heads, head_size]
        key: torch.Tensor,             # [batch_size * seq_len, num_heads, head_size]
        value: torch.Tensor,           # [batch_size * seq_len, num_heads, head_size]
        kv_cache: torch.Tensor,        # [num_blocks, block_size, num_heads, head_size]
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        The entire attention computation happens here!
        
        For GPT2 example:
            query.shape = [3, 12, 64]   # 3 tokens, 12 heads, 64 dims per head
            key.shape = [3, 12, 64]
            value.shape = [3, 12, 64]
            kv_cache.shape = [8000, 16, 12, 64]  # Pre-allocated GPU memory
        """
        
        # Step 1: Reshape query for kernel
        # Flatten to [batch_size * seq_len, num_heads, head_size]
        # = [3, 12, 64]
        
        # Step 2: Project KV cache to contiguous format
        # vLLM uses PAGED MEMORY: KV cache is scattered across GPU memory
        # in blocks, not contiguous!
        
        # Step 3: Reshape key and value similarly
        
        # ═══════════════════════════════════════════════════════════
        # STEP 4: CALL THE CUDA KERNEL!!! 
        # ═══════════════════════════════════════════════════════════
        
        attn_output = flash_attn_varlen_func(
            q=query,                           # [3, 12, 64]
            k=key_cache,                       # Scattered across blocks
            v=value_cache,                     # Scattered across blocks
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            causal=attn_metadata.causal,
            block_table=attn_metadata.block_tables,  # ← PAGED ATTENTION!
            # ... other parameters ...
        )
        
        # Step 5: Update KV cache for future tokens
        # Store current query (as key) and value in cache for next iteration
        
        return attn_output  # [batch_size * seq_len, num_heads, head_size]
```

---

## Part 4: CUDA Kernels for LLM Inference

### CUDA Kernel Files Used

Here's the complete list of CUDA kernel files and their purposes:

| Kernel File | Purpose | Lines |
|---|---|---|
| **attention/paged_attention_v1.cu** | PagedAttention v1 - basic efficient attention | ~187 |
| **attention/paged_attention_v2.cu** | PagedAttention v2 - optimized variant | ~400+ |
| **attention/merge_attn_states.cu** | Merge attention outputs across splits | - |
| **attention/vertical_slash_index.cu** | Utility for paged KV cache indexing | - |
| **activation_kernels.cu** | GELU, ReLU, Swish, etc. | ~300 |
| **layernorm_kernels.cu** | LayerNorm (residual + norm) | ~200 |
| **pos_encoding_kernels.cu** | RoPE position encodings | ~150 |
| **cuda_utils_kernels.cu** | Memory utilities, scatter/gather | ~100 |
| **sampler.cu** | Token sampling kernels | ~200 |

### Most Important: PagedAttention Kernels

**Why are these special?**

1. **Paged Attention** = Scattered KV cache in GPU memory
   - Traditional: Contiguous [batch, seq_len, heads, dim] blocks
   - vLLM: Non-contiguous blocks scattered across GPU memory
   - Advantage: Dynamic batching without fragmentation

2. **Efficient Attention Computation**
   - Fused kernel: Q·K→softmax→P·V all in one GPU kernel
   - No intermediate materializations (saves memory bandwidth)
   - Minimizes data movement

### Inside PagedAttention Kernel

**File:** `csrc/attention/paged_attention_v1.cu`

```cuda-cpp
template <typename T, typename CACHE_T, int HEAD_SIZE, int BLOCK_SIZE, 
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE>
__global__ void paged_attention_v1_kernel(
    T* out,                           // Output: [num_tokens, num_heads, head_size]
    T* query,                         // Input query: [num_tokens, num_heads, head_size]
    CACHE_T* key_cache,               // All K cache blocks (scattered)
    CACHE_T* value_cache,             // All V cache blocks (scattered)
    
    int num_kv_heads,                 // Number of KV heads (12 for GPT2)
    float scale,                      // Attention scale: 1/sqrt(head_size)
    
    // ← PAGED ATTENTION MAGIC: Block indirection
    int* block_tables,                // [num_seqs, max_blocks_per_seq]
    int* seq_lens,                    // [num_seqs] - actual sequence lengths
    int max_num_blocks_per_seq,       // Max blocks for any sequence
    
    // Optional features
    const float* alibi_slopes,        // ALiBi position bias (optional)
    int q_stride, int kv_block_stride, int kv_head_stride,
    const float* k_scale, const float* v_scale,  // FP8 quantization scales
    // ... other params ...
) {
    // GPU THREAD ORGANIZATION:
    // blockIdx.x  = head index (0 to num_heads - 1)
    // blockIdx.y  = sequence index (0 to num_seqs - 1)
    // threadIdx.x = thread within block (0 to NUM_THREADS - 1)
    
    // Each thread block handles ONE sequence × ONE head
    int head_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 1: Load Query & Compute Q·K
    // ═══════════════════════════════════════════════════════════
    
    // Step 1: Get pointer to this token's query
    T* query_ptr = query + seq_idx * stride + head_idx * head_size;
    T q[HEAD_SIZE];  // Load query into registers
    
    for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
        q[i] = query_ptr[i];
    }
    __syncthreads();  // Wait for all threads to load
    
    // Step 2: Iterate over all KV blocks in this sequence
    float max_logit = -FLT_MAX;
    float logits[BLOCK_SIZE];  // Logits for this block
    
    // Get the block table for this sequence
    int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
    int seq_len = seq_lens[seq_idx];
    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // For each block of keys in memory
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        
        // ← THIS IS THE KEY: Page indirection
        // Find physical block number from logical index
        int physical_block_idx = block_table[block_idx];  // ← INDIRECTION!
        
        // Get pointer to this block's key cache
        CACHE_T* key_cache_block = 
            key_cache + physical_block_idx * BLOCK_SIZE * num_heads * HEAD_SIZE
                      + head_idx * HEAD_SIZE;
        
        // Step 3: Load K from cache and compute Q·K for each position
        for (int pos_idx = 0; pos_idx < BLOCK_SIZE; ++pos_idx) {
            CACHE_T* key = key_cache_block + pos_idx * stride;
            
            // Compute dot product (Q · K^T)
            float logit = 0.0f;
            for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
                logit += q[i] * key[i];
            }
            
            // Reduce across threads
            logit = warp_reduce_sum(logit);  // Sum logits across 32 threads
            logits[pos_idx] = logit * scale;  // Scale
            
            // Track maximum for numerical stability
            max_logit = max(max_logit, logits[pos_idx]);
        }
    }
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 2: Apply Softmax(Q·K)
    // ═══════════════════════════════════════════════════════════
    
    // Compute exp(logit - max_logit) for numerical stability
    for (int pos_idx = 0; pos_idx < BLOCK_SIZE; ++pos_idx) {
        logits[pos_idx] = exp(logits[pos_idx] - max_logit);
    }
    
    // Sum of exponentials
    float sum_exp = 0.0f;
    for (int pos_idx = 0; pos_idx < BLOCK_SIZE; ++pos_idx) {
        sum_exp += logits[pos_idx];
    }
    sum_exp = warp_reduce_sum(sum_exp);
    
    // Normalize: softmax = exp(...) / sum(exp(...))
    for (int pos_idx = 0; pos_idx < BLOCK_SIZE; ++pos_idx) {
        logits[pos_idx] = logits[pos_idx] / sum_exp;  // ← Attention weights
    }
    __syncthreads();
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 3: Weighted Sum of Values (P·V)
    // ═══════════════════════════════════════════════════════════
    
    T accum[HEAD_SIZE] = {0};  // Accumulator for output
    
    // For each block again
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        
        int physical_block_idx = block_table[block_idx];
        CACHE_T* value_cache_block = 
            value_cache + physical_block_idx * BLOCK_SIZE * num_heads * HEAD_SIZE
                        + head_idx * HEAD_SIZE;
        
        // For each position in block
        for (int pos_idx = 0; pos_idx < BLOCK_SIZE; ++pos_idx) {
            float weight = logits[pos_idx];  // Attention weight (0.0 to 1.0)
            CACHE_T* value = value_cache_block + pos_idx * stride;
            
            // Weighted sum: output += attention_weight * V
            for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
                accum[i] += weight * value[i];
            }
        }
    }
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 4: Store Output
    // ═══════════════════════════════════════════════════════════
    
    T* output = out + seq_idx * stride + head_idx * head_size;
    for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
        output[i] = accum[i];
    }
}
```

### Kernel Launch from Python

**File:** `csrc/attention/paged_attention_v1.cu` (launcher part)

```cuda-cpp
template <typename T, typename CACHE_T, int BLOCK_SIZE, int NUM_THREADS>
void paged_attention_v1_launcher(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int max_seq_len,
    // ... params ...
) {
    // 1. Validate inputs
    TORCH_CHECK(query.is_cuda(), "Query must be on CUDA");
    TORCH_CHECK(query.dtype() == torch::kFloat16 || query.dtype() == torch::kBFloat16);
    
    // 2. Extract raw pointers for GPU memory
    T* out_ptr = out.data_ptr<T>();
    T* query_ptr = query.data_ptr<T>();
    CACHE_T* key_cache_ptr = key_cache.data_ptr<CACHE_T>();
    CACHE_T* value_cache_ptr = value_cache.data_ptr<CACHE_T>();
    int* block_tables_ptr = block_tables.data_ptr<int>();
    int* seq_lens_ptr = seq_lens.data_ptr<int>();
    
    // 3. Calculate grid and block dimensions
    int num_seqs = query.size(0);           // Number of sequences in batch
    int num_heads = query.size(1);          // Number of attention heads
    int head_size = query.size(2);          // Dimensions per head
    
    dim3 grid(num_heads, num_seqs);         // Grid = [num_heads, num_seqs]
    dim3 block(NUM_THREADS);                // Block = 128 threads per block
    
    // For GPT2 example:
    //   grid = [12, 1]           (12 heads, 1 sequence in batch)
    //   block = [128]            (128 threads)
    //   Total GPU threads = 12 * 1 * 128 = 1536 threads
    
    // 4. Calculate shared memory size
    int shared_mem_size = ...;  // For storing intermediate data
    
    // ═══════════════════════════════════════════════════════════
    // 5. LAUNCH THE CUDA KERNEL!!!
    // ═══════════════════════════════════════════════════════════
    
    VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
        ((void*)paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>),
        shared_mem_size
    );
    
    paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
        <<<grid, block, shared_mem_size, stream>>>(
            out_ptr, query_ptr, key_cache_ptr, value_cache_ptr,
            num_kv_heads, scale,
            block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,
            // ... all other parameters ...
        );
    
    // 6. Check for CUDA errors
    TORCH_CHECK_CUDA_ERROR();
}
```

---

## Part 5: Compilation/Build Process

### How vLLM CUDA Code Gets Compiled

#### Stage 1: Build Configuration (setup.py)

**File:** `setup.py` (simplified)

```python
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name='vllm._custom_ops',  # Python module name
        sources=[
            'csrc/torch_bindings.cpp',           # Pybind11 bindings
            'csrc/attention/paged_attention_v1.cu',
            'csrc/attention/paged_attention_v2.cu',
            'csrc/attention/merge_attn_states.cu',
            'csrc/activation_kernels.cu',
            'csrc/layernorm_kernels.cu',
            'csrc/pos_encoding_kernels.cu',
            'csrc/sampler.cu',
            'csrc/cuda_utils_kernels.cu',
            # ... more files ...
        ],
        include_dirs=['csrc/'],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': [
                '-O3',
                # Compile for Ampere (A100, RTX30xx)
                '-gencode', 'arch=compute_80,code=sm_80',
                # Also compile for Hopper (H100, H200)
                '-gencode', 'arch=compute_90,code=sm_90',
                # And Ada (RTX40xx)
                '-gencode', 'arch=compute_89,code=sm_89',
            ]
        },
    ),
    # ... more extensions ...
]

setup(
    name='vllm',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
```

**Build Command:**
```bash
python setup.py build_ext --inplace
```

#### Stage 2: Compilation Pipeline

```
setup.py build_ext
    ↓
detect_compute_capabilities()
    ↓
Find CUDA toolkit (nvcc, libcudart)
    ↓
For each .cu file:
    ↓
nvcc compiler:
    1. Parse CUDA syntax
    2. Generate GPU machine code for each arch (sm_80, sm_90, etc.)
    3. Generate CPU wrapper code (C++)
    4. Compile to object files (.o)
    ↓
C++ compiler (g++/clang):
    1. Compile Pybind11 bindings (torch_bindings.cpp)
    2. Combine with object files
    3. Link with PyTorch/CUDA runtime libraries
    ↓
Linker:
    1. Create shared library (.so on Linux, .pyd on Windows)
    ↓
Result: vllm/_custom_ops.so (Python extension)
    Contains:
    - paged_attention_v1_kernel (GPU executable)
    - paged_attention_v2_kernel (GPU executable)
    - activation_kernels (GPU executable)
    - ... all other kernels ...
```

#### Stage 3: Pybind11 Bridge

**File:** `csrc/torch_bindings.cpp` (simplified)

```cpp
#include <torch/extension.h>
#include "attention_kernels.cuh"

// C++ wrapper that Pybind11 will expose to Python
torch::Tensor paged_attention_v1_wrapper(
    torch::Tensor out,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    int64_t max_seq_len,
    // ... more params ...
) {
    // 1. Input validation
    TORCH_CHECK(query.is_cuda(), "Query must be on CUDA device");
    TORCH_CHECK(query.dtype() == torch::kFloat16 || query.dtype() == torch::kBFloat16,
                "Unsupported dtype");
    
    // 2. Extract tensor properties
    int batch_size = query.size(0);
    int num_heads = query.size(1);
    int head_size = query.size(2);
    
    // 3. Call the actual CUDA kernel launcher
    paged_attention_v1_launcher(
        out, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens,
        max_seq_len, // ... all other params ...
    );
    
    return out;
}

// Register with PyTorch/Pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_v1", &paged_attention_v1_wrapper);
    m.def("paged_attention_v2", &paged_attention_v2_wrapper);
    m.def("rms_norm", &rms_norm_wrapper);
    // ... register all kernel wrappers ...
}
```

#### Stage 4: PyTorch Extension Loading

**File:** `vllm/attention/utils/fa_utils.py`

```python
# When vLLM imports, PyTorch loads the compiled .so
try:
    import vllm._custom_ops as ops  # ← This is the compiled vllm/_custom_ops.so
except ImportError:
    raise RuntimeError("vLLM CUDA extensions not built. Run: pip install -e .")

# Now we can call the kernels from Python!
def flash_attn_varlen_func(q, k, v, ...):
    # This calls the compiled CUDA kernel
    return ops.flash_attn_varlen(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        # ... all parameters ...
    )
```

---

## Part 6: Putting It All Together - Complete Request Journey

### Example: Client Request for 1 Token

```
USER SENDS:
curl -X POST http://localhost:8000/v1/completions \
  -d '{"prompt": "Today is a", "max_tokens": 1}'

↓ ↓ ↓

1. HTTP Request Handler (api_server.py)
   - Receives request
   - Tokenizes: "Today is a" → [2911, 318, 257]
   
2. Engine.add_request()
   - Creates RequestGroup
   - Adds to waiting queue
   
3. Engine Generation Loop (every 10-50ms)
   a. Scheduler selects requests to process
   b. Batches 1 or more requests
   
4. Processor.prepare_inputs()
   - Creates metadata:
     block_tables: [[0, 1, 2, 3, 4, ...]]  # Which KV cache blocks contain data
     seq_lens: [3]                          # 3 tokens in sequence
     cu_seqlens: [0, 3]                     # Cumulative positions
   
5. Model.forward()
   - Input: input_ids=[2911, 318, 257], block_tables, seq_lens
   
   a. Embedding: [3, 768]
   
   b. For layer 1-12:
      i. Attention Layer:
         - Compute Q from hidden_states: [3, 12, 64]
         - Load K, V from KV cache
         - Call: flash_attn_varlen_func(Q, K_cache, V_cache, block_tables)
         
         GPU EXECUTION:
         ```
         paged_attention_v1_kernel<<<[12, 1], [128]>>>(...)
            ↑ blocks            ↑ threads per block
            12 attention heads, 1 sequence
            Total threads: 1536
         
         Each thread block handles: 1 head × 1 sequence
         
         Timeline (RTX 3090):
         - Load Q from registers: ~1 ns
         - Q·K computation loop: ~100 ns
         - Softmax: ~50 ns
         - P·V: ~100 ns
         - Total kernel time: ~250 ns per thread
         - Whole kernel: ~5 µs
         ```
         
         - Return attention output: [3, 12, 64]
         - Update KV cache with new K, V
      
      ii. FFN Layer
      iii. Residuals
   
   c. Output: logits [3, 50257]
   
6. Take last token: logits_last = logits[-1]  # [50257]

7. Sampler.forward()
   - Apply temperature: logits_last / 0.7
   - Apply top-p filtering (if enabled)
   - Call torch.multinomial() ← GPU sampling kernel
   - Sample next token: e.g., token_id = 407
   
8. Stream response to client
   - "The next token is: 'is'"

TOTAL TIME: 5-15ms on modern GPU
```

### GPU Memory Layout During Forward Pass

```
GPU Memory Map:
┌─────────────────────────────────────────┐
│ Model Parameters (~2.7 GB for GPT2-xl) │  ← Loaded once at startup
│  - Embeddings: 50257 × 768              │
│  - 12 Transformer Layers                │
│    - Attention: Q, K, V projections     │
│    - FFN weights                        │
├─────────────────────────────────────────┤
│ KV Cache (~98 MB for 1 layer × 8000 blocks) │  ← Allocated once, reused
│  - Shape: [8000, 16, 12, 64]            │
│  - For all 12 layers: ~1.2 GB           │
├─────────────────────────────────────────┤
│ Temporary Buffers (activations)         │  ← Allocated per forward pass
│  - Hidden states: [3, 768]              │
│  - Attention Q,K,V: [3, 12, 64]         │
│  - FFN intermediate: [3, 3072]          │
│  - Total: ~500 MB for full batch        │
└─────────────────────────────────────────┘

TOTAL: ~4-5 GB for GPT2-xl inference
```

---

## Part 7: Key CUDA Optimization Techniques Used

### 1. **PagedAttention (Block Indirection)**

Instead of contiguous memory:
```
Traditional:
Sequence 1: [Block 0] [Block 1] [Block 2] ...
Sequence 2: [Block 3] [Block 4] [Block 5] ...
            ↑ Contiguous, but fragments GPU memory

vLLM Paged:
Sequence 1: Block 23 → Block 47 → Block 191 → ...
Sequence 2: Block 8 → Block 52 → Block 9 → ...
            ↑ Scattered, but NO external fragmentation!
```

**Benefit:** Dynamic batching without memory reallocation

### 2. **Fused Kernels**

Instead of separate kernels:
```
Traditional:
Q·K → [temp] → Softmax([temp]) → [temp2] → P·V → [output]
      4 kernel launches, 4× memory bandwidth

vLLM:
Q·K → Softmax → P·V (all in 1 kernel)
      1 kernel launch, optimized memory pattern
```

### 3. **Shared Memory & Registers**

```cuda
// Load data into fast memory (not slow GPU DRAM)
__shared__ float shared_q[BLOCK_SIZE];      // ~32 KB per block
float q_reg[HEAD_SIZE];                     // ~256 bytes per thread
```

**Speed:** Registers: ~1 ns, Shared: ~10 ns, DRAM: ~100 ns

### 4. **Thread Cooperation**

```cuda
// All 128 threads work together on same attention computation
for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    logit += q[i] * k[i];  // Different threads compute different dimensions
}
// Synchronize when done
__syncthreads();
// Reduce across all threads
logit = warp_reduce_sum(logit);
```

---

## Summary: Request to CUDA Kernel Mapping

```
┌─────────────────────────────────────────┐
│ curl /v1/completions?prompt="..."       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ HTTP Request Handler                    │
│ (openai/serving_completion.py)          │
└────────────┬────────────────────────────┘
             │ Tokenize prompt
             ▼
┌─────────────────────────────────────────┐
│ LLMEngine.add_request()                 │
│ (v1/engine/async_llm.py)                │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Scheduler.schedule()                    │
│ (v1/engine/scheduler.py)                │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ ModelRunner.execute_model()             │
│ (executor/model_runner.py)              │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ GPT2Model.forward()                     │
│ (model_executor/models/gpt2.py)         │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ FlashAttentionImpl.forward()             │
│ (v1/attention/backends/flash_attn.py)   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ flash_attn_varlen_func()                │
│ (from vllm._custom_ops)                 │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Pybind11 Bridge (torch_bindings.cpp)    │
│ (csrc/torch_bindings.cpp)               │
└────────────┬────────────────────────────┘
             │ Extract tensor pointers
             ▼
┌─────────────────────────────────────────┐
│ paged_attention_v1_launcher<T, CACHE_T> │
│ (csrc/attention/paged_attention_v1.cu)  │
└────────────┬────────────────────────────┘
             │ Calculate grid/block, set parameters
             ▼
╔═════════════════════════════════════════╗
║ CUDA KERNEL LAUNCH                      ║
║ paged_attention_v1_kernel<T,...>        ║
║ <<<grid=[num_heads, num_seqs],          ║
║    block=[128], shared_mem>>>(...)      ║
║                                         ║
║ GPU Execution (5-10 µs):                ║
║  1. Load Q from GPU memory              ║
║  2. Q·K computation with page indirection
║  3. Softmax(Q·K)                        ║
║  4. Softmax(Q·K) · V                    ║
║  5. Store attention output              ║
║  6. Synchronize threads                 ║
╚═════════════════════════════════════════╝
             │
             ▼
┌─────────────────────────────────────────┐
│ Sampler.forward() - Sample next token   │
│ (model_executor/layers/sampler.py)      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Return result to client                 │
│ Stream token in response                │
└─────────────────────────────────────────┘
```

---

## Files Reference

### Key Python Entry Points
- `vllm/entrypoints/cli/main.py` - CLI entry point
- `vllm/entrypoints/cli/serve.py` - Serve subcommand
- `vllm/entrypoints/openai/api_server.py` - HTTP server
- `vllm/v1/engine/async_llm.py` - Async inference engine
- `vllm/v1/attention/backends/flash_attn.py` - Attention implementation

### Critical CUDA Files
- `csrc/attention/paged_attention_v1.cu` - Main attention kernel
- `csrc/attention/paged_attention_v2.cu` - Optimized variant
- `csrc/torch_bindings.cpp` - Pybind11 bindings
- `csrc/attention/attention_kernels.cuh` - Header with kernel definitions
- `setup.py` - Build configuration

### Model & Forward Pass
- `vllm/model_executor/models/` - Model implementations
- `vllm/v1/executor/` - Executor implementations
- `vllm/model_executor/layers/sampler.py` - Token sampling

---

## Conclusion

The journey from `vllm serve --model gpt2` and client requests to CUDA kernels involves:

1. **CLI → Server Setup** (< 100ms) - Parse arguments, start HTTP server
2. **Request Reception** - Accept /v1/completions POST
3. **Batching & Scheduling** - Group requests for efficient GPU utilization
4. **Model Forward Pass** - Execute transformer layers on GPU
5. **Attention Computation** - Largest component
   - Python → Pybind11 → CUDA kernel
   - PagedAttention with block indirection
   - Fused kernel optimization
6. **Token Sampling** - Use GPU-accelerated sampling
7. **Response Stream** - Return tokens to client

Each step leverages GPU acceleration, with PagedAttention kernels being the most critical and optimized part of the system. The entire forward pass for one token takes only 5-15ms on modern GPUs!

