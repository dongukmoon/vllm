# GPT2.py Attention Kernel Mapping and TP Worker Interaction

## Part 1: Attention Kernel Mapping Flow

When `gpt2.py` is executed, the attention kernel is selected through the following flow:

### 1. Attention Layer Initialization in GPT2Attention

**File:** [vllm/model_executor/models/gpt2.py](vllm/model_executor/models/gpt2.py#L85-L91)

```python
# In GPT2Attention.__init__()
self.attn = Attention(self.num_heads,
                      self.head_dim,
                      scale=self.scale,
                      cache_config=cache_config,
                      quant_config=quant_config,
                      prefix=f"{prefix}.attn")
```

### 2. Backend Selection in Attention Layer

**File:** [vllm/attention/layer.py](vllm/attention/layer.py#L190-L210)

```python
# During Attention.__init__()
dtype = torch.get_default_dtype()
if attn_backend is None:
    self.attn_backend = get_attn_backend(
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla=use_mla,
        has_sink=self.has_sink,
        use_sparse=use_sparse
    )
else:
    self.attn_backend = attn_backend

impl_cls = self.attn_backend.get_impl_cls()
self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                     alibi_slopes, sliding_window, kv_cache_dtype,
                     logits_soft_cap, attn_type, ...)
```

### 3. Backend Selection Logic

**File:** [vllm/attention/selector.py](vllm/attention/selector.py#L146-215)

The `get_attn_backend()` function uses a priority-based selection:

```python
def _cached_get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    use_v1: bool = False,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
) -> type[AttentionBackend]:

    # Priority 1: Global forced backend (highest priority)
    selected_backend = None
    backend_by_global_setting: Optional[_Backend] = (
        get_global_forced_attn_backend())
    if backend_by_global_setting is not None:
        selected_backend = backend_by_global_setting
    else:
        # Priority 2: Environment variable (VLLM_ATTENTION_BACKEND)
        backend_by_env_var: Optional[str] = envs.VLLM_ATTENTION_BACKEND
        if backend_by_env_var is not None:
            if backend_by_env_var.endswith("_VLLM_V1"):
                backend_by_env_var = backend_by_env_var.removesuffix("_VLLM_V1")
            selected_backend = backend_name_to_enum(backend_by_env_var)

    # Priority 3: Platform-specific auto-detection
    # get device-specific attn_backend
    attention_cls = current_platform.get_attn_backend_cls(
        selected_backend, head_size, dtype, kv_cache_dtype, block_size, use_v1,
        use_mla, has_sink, use_sparse)
    
    return resolve_obj_by_qualname(attention_cls)
```

### 4. Kernel Binding at Runtime

**File:** [vllm/attention/utils/fa_utils.py](vllm/attention/utils/fa_utils.py#L13-18)

For CUDA platforms, the actual CUDA kernels are imported from compiled `.so` files:

```python
if current_platform.is_cuda():
    from vllm import _custom_ops as ops
    reshape_and_cache_flash = ops.reshape_and_cache_flash
    from vllm.vllm_flash_attn import (
        flash_attn_varlen_func,              # Main attention kernel
        get_scheduler_metadata
    )
```

### 5. Typical Attention Backend Selection for GPT2 on CUDA

For a standard NVIDIA GPU with CUDA support, GPT2 will typically use:

- **FlashAttention V3** (if CUDA 12.3+ and Hopper GPU SM 90)
- **FlashAttention V2** (if older CUDA or pre-Hopper GPU)
- **FlexAttention** (fallback for unsupported head sizes)

---

## Part 2: Tensor Parallelism (TP) Worker Interaction for Forward Pass

When `TP=2`, there are 2 workers processing different portions of tensors. Here's how they interact:

### Architecture Overview

```
TP=2 (2 workers on 2 GPUs)

Worker 0 (GPU 0)          Worker 1 (GPU 1)
    |                           |
    QKVParallelLinear(c_attn)  QKVParallelLinear(c_attn)
    |                           |
    ├─ Produces: q0, k0, v0     ├─ Produces: q1, k1, v1
    |                           |
    └─ [AllGather] ────────────────┘
            ↓
    Hidden state fully gathered across workers
            ↓
    Attention(q, k, v)
    |                           |
    └─ [AllReduce] ─────────────┘
            ↓
    Attention output combined
            ↓
    RowParallelLinear(c_proj)
    |                           |
    └─ [AllReduce] ─────────────┘
            ↓
    Final output
```

### 1. QKVParallelLinear (ColumnParallelLinear) - First Communication

**File:** [vllm/model_executor/models/gpt2.py](vllm/model_executor/models/gpt2.py#L70-76)

```python
class GPT2Attention(nn.Module):
    def __init__(self, ...):
        self.c_attn = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.c_attn",
        )
```

**Implementation:** [vllm/model_executor/layers/linear.py](vllm/model_executor/layers/linear.py#L405-560)

```python
class ColumnParallelLinear(LinearBase):
    """
    Parallelizes along output dimension.
    Weight matrix A = [A_1, ..., A_p] where each worker has A_i
    """
    
    def __init__(self, ...):
        self.tp_rank = get_tensor_model_parallel_rank()      # 0 or 1
        self.tp_size = get_tensor_model_parallel_world_size() # 2
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.gather_output = gather_output  # False for c_attn
    
    def forward(self, input_):
        # Each worker computes its portion
        output_parallel = self.quant_method.apply(self, input_, bias)
        
        if self.gather_output and self.tp_size > 1:
            # AllGather operation when gather_output=True
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        
        return output, output_bias
```

### Code Patch: ColumnParallelLinear Forward Pass with TP=2

```python
# BEFORE: Input arrives at both workers (same tensor on GPU 0 and GPU 1)
# hidden_states: shape (batch_size, seq_len, hidden_size)

# STEP 1: Each worker computes partial QKV
# Worker 0: output_parallel_0 = hidden_states @ A_0  # shape (..., output_size//2)
# Worker 1: output_parallel_1 = hidden_states @ A_1  # shape (..., output_size//2)

# STEP 2: For c_attn, gather_output=False, so NO AllGather
# Result: Each worker has only its portion
# Worker 0 qkv -> q0, k0, v0  (hidden_dim // 2)
# Worker 1 qkv -> q1, k1, v1  (hidden_dim // 2)
```

### 2. Attention Forward Pass - Second Communication

**File:** [vllm/v1/attention/backends/flash_attn.py](vllm/v1/attention/backends/flash_attn.py#L498-530)

Since each worker only has partial Q, K, V tensors, the attention operation needs full tensors:

```python
def forward(...):
    # The attention implementation typically handles AllGather internally
    # or the tensors are gathered before attention computation
    
    reshape_and_cache_flash(
        key, value,
        key_cache, value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale, v_scale
    )
    
    attn_output = flash_attn_varlen_func(
        q, k, v,  # These need to be gathered across workers
        seq_lens,
        max_seqlen,
        ...
    )
```

### 3. RowParallelLinear (c_proj) - Third Communication

**File:** [vllm/model_executor/models/gpt2.py](vllm/model_executor/models/gpt2.py#L87-91)

```python
self.c_proj = RowParallelLinear(
    self.hidden_size,
    self.hidden_size,
    bias=True,
    quant_config=quant_config,
    prefix=f"{prefix}.c_proj",
)
```

**Implementation:** [vllm/model_executor/layers/linear.py](vllm/model_executor/layers/linear.py#L1197-1365)

```python
class RowParallelLinear(LinearBase):
    """
    Parallelizes along input dimension.
    X = [X_1, ..., X_p], A = [A_1^T, ..., A_p^T]^T
    Each worker computes Y_i = X_i @ A_i, then AllReduce to get Y
    """
    
    def __init__(self, ...):
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.reduce_results = reduce_results  # True for c_proj
    
    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            # Split input across workers if not already parallel
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()
        
        # Each worker computes its portion
        # Bias is only added on rank 0
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)
        
        if self.reduce_results and self.tp_size > 1:
            # AllReduce: combine results from all workers
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel
        
        return output, output_bias
```

### Code Patch: RowParallelLinear Forward Pass with TP=2

```python
# BEFORE: Attention output from attention layer
# attn_output: shape (batch_size, seq_len, hidden_size)

# STEP 1: Split input along last dimension
# Worker 0: input_parallel_0 = attn_output[:, :, :hidden_size//2]
# Worker 1: input_parallel_1 = attn_output[:, :, hidden_size//2:]

# STEP 2: Each worker computes its portion
# Worker 0: output_parallel_0 = input_parallel_0 @ A_0  # shape (..., hidden_size)
# Worker 1: output_parallel_1 = input_parallel_1 @ A_1  # shape (..., hidden_size)

# STEP 3: AllReduce to sum results
# output = output_parallel_0 + output_parallel_1
# Result: Each worker has FULL output (same value on both GPUs)
```

---

## Summary: Forward Pass Communication Flow (TP=2)

### GPT2Attention Forward Pass Sequence

```
1. Input broadcast to all workers (hidden_states on GPU 0 and GPU 1)
   ↓
2. QKVParallelLinear (ColumnParallel)
   - Worker 0: q0, k0, v0 = hidden_states @ [A_0, B_0, C_0]
   - Worker 1: q1, k1, v1 = hidden_states @ [A_1, B_1, C_1]
   - NO communication (gather_output=False)
   ↓
3. Attention (internal all-gather or broadcast)
   - Needs full q, k, v for attention computation
   - May communicate to gather partial tensors
   ↓
4. RowParallelLinear (c_proj)
   - Worker 0: output_0 = attn_output_partial @ A_0 + bias
   - Worker 1: output_1 = attn_output_partial @ A_1
   - AllReduce: output = output_0 + output_1
   ↓
5. Output broadcast to all workers (ready for next layer)
```

### Communication Primitives Used

From [vllm/distributed/parallel_state.py](vllm/distributed/parallel_state.py):

```python
# These are NCCL collective operations wrapped by PyTorch

tensor_model_parallel_all_gather(tensor)
    # Concatenates tensor from all workers
    # TP=2: [data_0, data_1]
    
tensor_model_parallel_all_reduce(tensor)
    # Sums tensors from all workers (default: sum operation)
    # TP=2: data_0 + data_1

split_tensor_along_last_dim(tensor, num_partitions)
    # Splits tensor along last dimension
    # TP=2: [tensor[:, :, :d//2], tensor[:, :, d//2:]]
```

---

## Complete Forward Pass Trace with TP=2

```python
# vllm/model_executor/models/gpt2.py

class GPT2Attention(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Step 1: Project to QKV (ColumnParallel - scattered output)
        qkv, _ = self.c_attn(hidden_states)  
        # Worker 0 has qkv_0 (q0, k0, v0)
        # Worker 1 has qkv_1 (q1, k1, v1)
        
        # Step 2: Split QKV
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        
        # Step 3: Attention computation
        attn_output = self.attn(q, k, v)
        # attn_output: gathered/broadcast across workers
        
        # Step 4: Project back (RowParallel - reduce output)
        attn_output, _ = self.c_proj(attn_output)
        # Result: scattered across workers
        
        return attn_output
```

Each communication is a synchronized NCCL operation across all workers, ensuring data consistency during distributed computation.
