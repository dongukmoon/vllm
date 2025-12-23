# Python to CUDA Kernels: How vLLM's Python Code Translates to FlashAttention CUDA Kernels

## Overview: The Translation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│           vLLM Python Code (High Level)                         │
│                                                                 │
│  flash_attn_varlen_func(                                        │
│      q, k, v, block_table, seq_lens, ...)                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │  Python Function Call Resolution    │
        │  (Import/Dispatch)                  │
        │                                     │
        │  IF CUDA:                          │
        │    from vllm.vllm_flash_attn       │
        │    (Pre-compiled .so module)       │
        │                                     │
        │  ELIF XPU (Intel):                 │
        │    from vllm._ipex_ops             │
        │    (Intel optimized)               │
        │                                     │
        │  ELIF ROCm:                        │
        │    from aiter                      │
        │    (AMD library)                   │
        └──────────────┬──────────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
       ┌────v────┐         ┌─────v─────┐
       │  CUDA   │         │  Non-CUDA │
       │  Path   │         │  Paths    │
       └────┬────┘         └─────┬─────┘
            │                    │
            ▼                    ▼
┌─────────────────────┐  ┌──────────────────┐
│ vllm_flash_attn     │  │ Intel IPEX Ops   │
│ (Pre-compiled)      │  │ AMD Aiter        │
│                     │  │ PyTorch Custom   │
│ ├─ .so library      │  └──────────────────┘
│ ├─ CUDA kernels     │
│ └─ Pybind11 bindings
└─────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│             CUDA Runtime Execution                              │
│                                                                 │
│  GPU Kernel Launch:                                             │
│  ├─ Device Memory Setup                                         │
│  │  (Q, K, V, Block Table on GPU)                              │
│  │                                                              │
│  ├─ Kernel Configuration                                        │
│  │  (Grid/Block dimensions)                                     │
│  │                                                              │
│  ├─ Kernel Execution                                            │
│  │  (FlashAttention algorithm)                                  │
│  │                                                              │
│  └─ Result Transfer (Output back to host)                      │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  Attention Output Tensor            │
│  [num_tokens, num_heads, head_size] │
└─────────────────────────────────────┘
```

---

## The Python Call Path

### 1. High-Level Python Call

**File:** [vllm/v1/attention/backends/flash_attn.py](vllm/v1/attention/backends/flash_attn.py)

```python
class FlashAttentionImpl(AttentionImpl):
    def forward(
        self,
        layer,
        query: torch.Tensor,           # On GPU
        key: torch.Tensor,             # On GPU
        value: torch.Tensor,           # On GPU
        kv_cache: torch.Tensor,        # On GPU (scattered blocks)
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor = None,
    ) -> torch.Tensor:
        
        # ... KV cache update ...
        
        # CRITICAL: Call external FlashAttention kernel
        attn_output = flash_attn_varlen_func(
            q=query,                      # Contiguous tensor
            k=key_cache,                  # Non-contiguous (scattered blocks)
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            block_table=block_table,      # ← PagedAttention indirection!
            softmax_scale=self.scale,
            sliding_window=self.sliding_window,
            alibi_slopes=self.alibi_slopes,
            logits_soft_cap=self.logits_soft_cap,
            causal=True,
        )
        
        return attn_output
```

### 2. Import Resolution

**File:** [vllm/attention/utils/fa_utils.py](vllm/attention/utils/fa_utils.py)

```python
# Platform detection and kernel selection
if current_platform.is_cuda():
    # NVIDIA GPU Path
    from vllm import _custom_ops as ops
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    # ↑ Pre-compiled module with CUDA kernels
    
elif current_platform.is_xpu():
    # Intel GPU Path
    from vllm._ipex_ops import ipex_ops as ops
    flash_attn_varlen_func = ops.flash_attn_varlen_func
    # ↑ Intel IPEX optimized kernels
    
else:
    # Fallback or error
    flash_attn_varlen_func = None
```

**What is `vllm.vllm_flash_attn`?**

It's a pre-compiled Python extension module that contains:
- **Pre-compiled CUDA kernels** (binary machine code)
- **Python bindings** (Pybind11) to call those kernels
- **Configuration logic** for kernel launch parameters

---

## Deep Dive: Compilation Pipeline

### Stage 1: Building the vllm_flash_attn Module

#### A. How It Gets Compiled

**During vLLM Installation:**

```bash
# User runs:
pip install vllm

# Behind the scenes:
# 1. Setup.py detects CUDA installation
# 2. Invokes NVIDIA CUDA compiler (nvcc)
# 3. Compiles FlashAttention CUDA source code
# 4. Generates .so library (shared object)

# Directory structure:
vllm/
├── _custom_ops.so              ← PyTorch custom ops (CUDA)
├── vllm_flash_attn/
│   ├── __init__.py
│   ├── flash_attn_interface.py
│   └── libflashattn.so         ← FlashAttention kernel library
```

#### B. Source Code Structure

```
vllm/csrc/                          ← CUDA Source
├── attention/
│   ├── paged_attention_v1.cu      ← PagedAttention kernel
│   └── paged_attention_v2.cu      ← Optimized version
├── torch_bindings.cpp             ← Pybind11 bindings

setup.py                           ← Build configuration
├── Detects CUDA installation
├── Invokes nvcc compiler
└── Generates Python extension
```

### Stage 2: The Pybind11 Bridge

**What is Pybind11?**

A library that creates Python bindings for C++/CUDA code, allowing Python to call compiled kernels.

**Example Binding Code (pseudo-code):**

```cpp
// File: csrc/torch_bindings.cpp

#include <torch/extension.h>
#include "attention_kernels.cuh"

// Expose CUDA kernel to Python
torch::Tensor flash_attn_varlen_func(
    torch::Tensor q,                // Python tensor → C++ tensor
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    // ... other parameters ...
) {
    // 1. Check inputs (sizes, dtypes, device)
    TORCH_CHECK(q.is_cuda(), "q must be on CUDA device");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "Unsupported dtype");
    
    // 2. Extract raw pointers for GPU memory
    float* q_ptr = q.data_ptr<float>();
    float* key_cache_ptr = key_cache.data_ptr<float>();
    int* block_table_ptr = block_table.data_ptr<int>();
    
    // 3. Calculate grid/block dimensions
    dim3 grid(num_heads, batch_size);   // GPU grid
    dim3 block(THREADS_PER_BLOCK);      // Threads per block
    
    // 4. LAUNCH CUDA KERNEL
    paged_attention_v1_kernel<float, 16, 128>
        <<<grid, block, shared_mem_size, stream>>>(
            out_ptr, q_ptr, key_cache_ptr, value_cache_ptr,
            num_kv_heads, scale, block_table_ptr, seq_lens_ptr,
            max_num_blocks_per_seq, alibi_slopes_ptr,
            // ... all parameters ...
        );
    
    // 5. Check for CUDA errors
    TORCH_CHECK_CUDA_ERROR();
    
    // 6. Return output tensor
    return output;
}

// Register with PyTorch
m.def("flash_attn_varlen_func", &flash_attn_varlen_func);
```

---

## The Complete Execution Flow

### Step 1: Python Call Enters Pybind11

```python
# Python side (vllm/v1/attention/backends/flash_attn.py)
attn_output = flash_attn_varlen_func(
    q=query,              # torch.Tensor on GPU
    k=key_cache,          # torch.Tensor on GPU
    v=value_cache,        # torch.Tensor on GPU
    block_table=...,      # torch.Tensor on GPU
    # ... parameters ...
)

# ↓ Pybind11 intercepts this call ↓

# C++ side (csrc/torch_bindings.cpp)
torch::Tensor flash_attn_varlen_func(
    torch::Tensor q,          // ← Same tensor, now accessible in C++
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    // ...
)
```

### Step 2: Input Validation & Parameter Extraction

```cpp
// C++ side (Pybind11 handler)

// 1. Validate tensor properties
auto q_sizes = q.sizes();              // Get shape: [num_tokens, num_heads, head_size]
int num_tokens = q_sizes[0];
int num_heads = q_sizes[1];
int head_size = q_sizes[2];

// 2. Check device (must be CUDA)
TORCH_CHECK(q.device().type() == torch::kCUDA);
TORCH_CHECK(key_cache.device().type() == torch::kCUDA);

// 3. Check data types
TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16);
TORCH_CHECK(key_cache.dtype() == q.dtype());

// 4. Get raw GPU pointers
float* q_ptr = q.data_ptr<float>();
float* key_cache_ptr = key_cache.data_ptr<float>();
float* value_cache_ptr = value_cache.data_ptr<float>();
int* block_table_ptr = block_table.data_ptr<int>();
int* seq_lens_ptr = seq_lens.data_ptr<int>();

// 5. Get CUDA stream (GPU execution context)
cudaStream_t stream = at::cuda::getCurrentCUDAStream();
```

### Step 3: Kernel Configuration

```cpp
// Calculate GPU grid and block dimensions

// Grid: One block per head per sequence
// Block: THREADS_PER_BLOCK threads
dim3 grid(
    num_heads,              // X: num_heads blocks in X direction
    num_seqs,               // Y: num_seqs blocks in Y direction
    1                       // Z: 1 (not used)
);

dim3 block(
    THREADS_PER_BLOCK,      // Usually 128 or 256 threads per block
    1,
    1
);

// Shared memory size (SRAM) needed for this kernel
int shared_mem_size = max(
    logits_size,            // Space for attention logits
    outputs_size            // Space for partial outputs
);
// Typical: 32 KB for this kernel

// CUDA grid layout:
// Each block computes attention for one (head, sequence) pair
//
// Block (0, 0):  head_0, seq_0  ←→  grid_x=0, grid_y=0
// Block (1, 0):  head_1, seq_0  ←→  grid_x=1, grid_y=0
// Block (0, 1):  head_0, seq_1  ←→  grid_x=0, grid_y=1
// Block (1, 1):  head_1, seq_1  ←→  grid_x=1, grid_y=1
// ...
```

### Step 4: Kernel Launch

```cpp
// Launch the CUDA kernel!!!
paged_attention_v1_kernel<float, 128>
    <<<grid, block, shared_mem_size, stream>>>(
        // Input/Output
        out_ptr,                          // Output attention
        query_ptr,                        // Q tensor
        key_cache_ptr,                    // K cache (scattered blocks)
        value_cache_ptr,                  // V cache (scattered blocks)
        
        // Configuration
        num_kv_heads,
        scale,
        
        // ← PAGED ATTENTION INDIRECTION
        block_tables_ptr,                 // [num_seqs, max_blocks_per_seq]
        seq_lens_ptr,                     // [num_seqs]
        max_num_blocks_per_seq,
        
        // Optional features
        alibi_slopes_ptr,                 // Optional ALiBi
        q_stride, kv_block_stride, kv_head_stride,
        k_scale_ptr, v_scale_ptr,         // Optional quantization scales
        tp_rank,
        blocksparse_local_blocks,         // Optional block sparsity
        blocksparse_vert_stride,
        blocksparse_block_size,
        blocksparse_head_sliding_step
    );

// CUDA now asynchronously executes this on the GPU
// Pybind11 handler returns immediately (doesn't wait)
```

### Step 5: Inside the CUDA Kernel

**File:** [csrc/attention/paged_attention_v1.cu](csrc/attention/paged_attention_v1.cu)

```cuda-cpp
template <typename T, typename CACHE_T, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    T* out,
    T* query,
    CACHE_T* key_cache,
    CACHE_T* value_cache,
    int num_kv_heads,
    float scale,
    int* block_tables,           // ← PagedAttention block table
    int* seq_lens,
    int max_num_blocks_per_seq,
    // ... other params ...
) {
    // GPU Thread hierarchy:
    // blockIdx.x  = head_idx (0 to num_heads-1)
    // blockIdx.y  = seq_idx  (0 to num_seqs-1)
    // threadIdx.x = thread_idx within block (0 to NUM_THREADS-1)
    
    int head_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int token_idx = threadIdx.x;
    
    // Shared memory (fast SRAM on GPU)
    extern __shared__ char shared_mem[];
    float* logits = (float*)shared_mem;  // Logits buffer
    
    // ═══════════════════════════════════════════════════════════
    // CRITICAL: Get block table for this sequence
    // ═══════════════════════════════════════════════════════════
    int* seq_block_table = block_tables + seq_idx * max_num_blocks_per_seq;
    // ↑ Points to [block_0_physical_id, block_1_physical_id, ...]
    
    int seq_len = seq_lens[seq_idx];
    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // ═══════════════════════════════════════════════════════════
    // Compute QK^T (logits) using paged KV cache
    // ═══════════════════════════════════════════════════════════
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        
        // PAGE INDIRECTION: Get physical block ID from block table
        int physical_block_idx = seq_block_table[block_idx];
        // Example: seq_block_table[0] = 5 → physical block 5 in GPU memory
        //          seq_block_table[1] = 12 → physical block 12
        
        // Calculate actual physical address in GPU memory
        // K_cache is laid out as: [num_blocks][num_kv_heads][head_size/x][block_size][x]
        int k_offset = physical_block_idx * kv_block_stride +
                       head_idx * kv_head_stride;
        
        CACHE_T* k_block_ptr = key_cache + k_offset;
        // ↑ Now points to the actual K values for this block
        
        // Load key from scattered block location
        T k_vec[HEAD_SIZE];
        for (int i = token_idx; i < BLOCK_SIZE; i += blockDim.x) {
            // GPU memory access (might be scattered, but GPU cache helps)
            load_from_memory(k_block_ptr + i * HEAD_SIZE, k_vec);
            
            // Compute logit: Q·K^T
            float logit = dot_product(q_vec, k_vec) * scale;
            logits[i] = logit;  // Store in SRAM
        }
        __syncthreads();  // Sync threads within block
    }
    
    // ═══════════════════════════════════════════════════════════
    // Softmax on logits (in SRAM for speed)
    // ═══════════════════════════════════════════════════════════
    float max_logit = logits[0];
    for (int i = 1; i < seq_len; ++i)
        max_logit = max(max_logit, logits[i]);  // Find max
    
    float sum_exp = 0;
    for (int i = 0; i < seq_len; ++i) {
        logits[i] = exp(logits[i] - max_logit);  // Stable softmax
        sum_exp += logits[i];
    }
    __syncthreads();
    
    // ═══════════════════════════════════════════════════════════
    // Weighted sum of values (attention output)
    // ═══════════════════════════════════════════════════════════
    T accum[HEAD_SIZE] = {0};  // Accumulator in registers/SRAM
    
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        
        // PAGE INDIRECTION again!
        int physical_block_idx = seq_block_table[block_idx];
        
        int v_offset = physical_block_idx * kv_block_stride +
                       head_idx * kv_head_stride;
        
        CACHE_T* v_block_ptr = value_cache + v_offset;
        
        // Load values from scattered block
        for (int i = token_idx; i < BLOCK_SIZE; i += blockDim.x) {
            T v_vec[HEAD_SIZE];
            load_from_memory(v_block_ptr + i * HEAD_SIZE, v_vec);
            
            // Accumulate: out += attention_weight * V
            float weight = logits[i] / sum_exp;
            for (int d = 0; d < HEAD_SIZE; ++d)
                accum[d] += weight * v_vec[d];
        }
    }
    
    // ═══════════════════════════════════════════════════════════
    // Write output to GPU HBM
    // ═══════════════════════════════════════════════════════════
    int out_offset = seq_idx * stride + head_idx * HEAD_SIZE;
    T* out_ptr = out + out_offset;
    
    for (int d = threadIdx.x; d < HEAD_SIZE; d += blockDim.x)
        out_ptr[d] = accum[d];
}
```

### Step 6: Synchronization & Return

```cpp
// After kernel launch (back in Pybind11 handler)

// Wait for GPU to finish (synchronize)
// Actually: We don't wait! Kernel runs asynchronously
// Next Python operations will implicitly wait when needed

// Return output tensor to Python
return output;  // ← Python receives attention result

// ↓ Back to Python ↓

# Python side receives:
attn_output  # torch.Tensor on GPU with attention results
```

---

## Data Flow Diagram: Parameters to Kernel

```
Python Level (torch tensors):
┌─────────────────────────────────────┐
│ q:            [num_tokens, num_heads, head_size]
│ k_cache:      [num_blocks, num_kv_heads, head_size/x, block_size, x]
│ v_cache:      [num_blocks, num_kv_heads, head_size, block_size]
│ block_table:  [num_seqs, max_blocks_per_seq]
│ seq_lens:     [num_seqs]
└─────────────────────────────────────┘
         │  Pybind11 conversion
         ▼
C++ Level (raw pointers):
┌─────────────────────────────────────┐
│ float* q_ptr
│ float* key_cache_ptr
│ float* value_cache_ptr
│ int* block_table_ptr
│ int* seq_lens_ptr
└─────────────────────────────────────┘
         │  Kernel parameter passing
         ▼
CUDA Kernel Level (GPU execution):
┌─────────────────────────────────────┐
│ blockIdx.x = head_idx
│ blockIdx.y = seq_idx
│ threadIdx.x = token_idx within block
│
│ Access pattern:
│ physical_block = block_table_ptr[seq_idx][block_idx]
│                 ↑ Page indirection
│ k_ptr = key_cache_ptr[physical_block][head_idx]
│ v_ptr = value_cache_ptr[physical_block][head_idx]
│
│ → Compute attention
│ → Write output
└─────────────────────────────────────┘
```

---

## Multiple Kernel Variants (Template Specialization)

vLLM pre-compiles MANY kernel versions:

```cpp
// From csrc/attention/paged_attention_v1.cu

// Different head sizes need different code paths
switch (head_size) {
    case 32:
      LAUNCH_PAGED_ATTENTION_V1(32);    // Template instantiation
      break;
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    // ... more sizes ...
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
}

// Also: Different block sizes
// template<..., int BLOCK_SIZE, ...>
//   - BLOCK_SIZE=8, 16, 32, 64, 128, 256
```

**Why multiple versions?**
- CUDA kernels need compile-time constants for unrolling loops
- Different head sizes need different register layouts
- Different block sizes need different grid configurations
- Pre-compilation means all versions are ready to use

---

## Performance: Python Overhead

```
Total Time = Python Setup + Kernel Execution + Sync

┌──────────────────────────────────────────┐
│ Python Setup      (~0.1-0.5 ms)         │
│ ├─ Import module  (done once at startup) │
│ ├─ Convert params                        │
│ ├─ Check tensors                         │
│ └─ Calculate grid/block dims             │
└──────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│ GPU Kernel Execution (~1-10 ms)         │
│ ├─ Load Q                                │
│ ├─ For each block:                       │
│ │  ├─ Load K from scattered block (←PA)  │
│ │  ├─ Compute QK^T                       │
│ │  ├─ Load V from scattered block (←PA)  │
│ │  └─ Compute attention                  │
│ └─ Write output                          │
└──────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│ Synchronization (implicit in next op)   │
│ ├─ Next GPU operation waits for this    │
│ └─ Or explicit cudaDeviceSynchronize() │
└──────────────────────────────────────────┘

Result: Minimal Python overhead (kernel does real work)
        Python overhead < 1% typically
```

---

## Platform-Specific Implementations

### NVIDIA CUDA (Default)

```python
# vllm/vllm_flash_attn/
├── libflashattn.so           ← Compiled CUDA kernels
├── flash_attn_interface.py   ← Python bindings
└── __init__.py
    
from vllm.vllm_flash_attn import flash_attn_varlen_func
# Uses: paged_attention_v1.cu, paged_attention_v2.cu
```

### Intel XPU

```python
# vllm/_ipex_ops.py

class ipex_ops:
    @staticmethod
    def flash_attn_varlen_func(...):
        # Intel IPEX optimized kernel
        # Uses Intel GPU (Arc, Data Center GPU)
        return intel_kernel_result
        
from vllm._ipex_ops import ipex_ops
flash_attn_varlen_func = ipex_ops.flash_attn_varlen_func
```

### AMD ROCm / AITER

```python
# vllm/v1/attention/backends/rocm_aiter_fa.py

# Uses external 'aiter' library
from aiter import flash_attn_varlen_func

# Or custom kernel:
def flash_attn_varlen_func_impl(...):
    # Custom implementation using Triton/HIP
    ...
```

---

## Summary: Python → CUDA Journey

| Stage | What Happens |
|-------|-------------|
| **1. Python Call** | `flash_attn_varlen_func(q, k, v, block_table, ...)` |
| **2. Module Import** | Pybind11 resolves to pre-compiled .so library |
| **3. Pybind11** | Converts Python tensors to C++ pointers |
| **4. Validation** | Checks device, dtype, dimensions |
| **5. Kernel Config** | Calculates grid (num_heads × num_seqs) and block (128-256 threads) |
| **6. Kernel Launch** | CUDA kernel <<<grid, block>>> launches asynchronously |
| **7. GPU Execution** | Each thread block processes one (head, sequence) pair |
| **8. Block Indirection** | Kernel uses `block_table[seq_id][block_idx]` for scattered KV access |
| **9. Attention Compute** | Q·K^T, softmax, ∑(attn_weight × V) in SRAM |
| **10. Output Write** | Write results to GPU HBM |
| **11. Return** | Pybind11 returns output tensor to Python |

**Key Innovation: Paged Attention Block Table**
- Compiled CUDA kernel accepts `block_table` parameter
- Kernel uses table for indirection: `physical_block_id = block_table[seq_id][block_idx]`
- Enables flexible non-contiguous block layout
- No recompilation needed for different allocation patterns

---

## Build Configuration

**File:** [setup.py](setup.py)

```python
# Simplified view
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

# Define CUDA extensions
ext_modules = [
    CUDAExtension(
        name='vllm._custom_ops',
        sources=[
            'csrc/torch_bindings.cpp',
            'csrc/attention/paged_attention_v1.cu',
            'csrc/attention/paged_attention_v2.cu',
            # ... more source files ...
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '-gencode', 'arch=compute_80,code=sm_80']
                    # ↑ Compile for Ampere (A100, RTX30xx)
                    #   Also: compute_90,code=sm_90 for Hopper (H100)
        },
        include_dirs=[...],
    ),
    # ... more extensions ...
]

setup(
    name='vllm',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
```

**Build command:**
```bash
python setup.py build_ext --inplace
# ↓ Invokes NVIDIA nvcc compiler
# ↓ Generates libflashattn.so
```

---

## Conclusion

The translation from Python to CUDA kernels in vLLM:

1. **Pre-compilation** - CUDA kernels compiled to .so when vLLM installed
2. **Pybind11 bindings** - C++ wrapper code to accept Python tensors
3. **Runtime dispatch** - Python function call routed to compiled kernel
4. **Kernel execution** - GPU runs PagedAttention algorithm with block table indirection
5. **Result return** - Output tensor returned as PyTorch tensor

**Why this design?**
- **Performance**: No Python overhead during attention (compiled code runs)
- **Flexibility**: Block table allows PagedAttention indirection without recompilation
- **Compatibility**: Works with any PyTorch version (uses Pybind11)
- **Multiple variants**: Pre-compiled for all head sizes, block sizes, GPU architectures

The magic bridge: **Pybind11** converts between Python's tensor abstraction and C++'s raw GPU pointers!
