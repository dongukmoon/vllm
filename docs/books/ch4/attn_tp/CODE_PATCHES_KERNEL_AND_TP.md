# Code Patches: Attention Kernel Mapping and TP Worker Communication

## PATCH 1: Attention Kernel Selection Flow in gpt2.py

**File:** `vllm/model_executor/models/gpt2.py`

### How Attention Backend is Selected

```python
# Lines 83-91: GPT2Attention initialization
class GPT2Attention(nn.Module):

    def __init__(
        self,
        config: GPT2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim**-0.5

        # ... QKVParallelLinear and RowParallelLinear setup ...

        # CRITICAL: This is where the attention kernel gets bound
        self.attn = Attention(                              # ← Kernel Selection Point 1
            self.num_heads,
            self.head_dim,
            scale=self.scale,
            cache_config=cache_config,                     # ← Contains block_size, cache_dtype
            quant_config=quant_config,
            prefix=f"{prefix}.attn"
        )
```

---

## PATCH 2: Kernel Selection in Attention Layer

**File:** `vllm/attention/layer.py` (Lines 190-210)

```python
class Attention(nn.Module, AttentionLayerBase):
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        logits_soft_cap: Optional[float] = None,
        per_layer_sliding_window: Optional[int] = None,
        use_mla: bool = False,
        use_sparse: bool = False,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        attn_backend: Optional[type[AttentionBackend]] = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        # ... initialization code ...
        
        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        
        # ======= KERNEL SELECTION HAPPENS HERE =======
        if attn_backend is None:
            self.attn_backend = get_attn_backend(              # ← Kernel Selection Point 2
                head_size,                                      # From GPT2: 64 or 128
                dtype,                                          # float16 or bfloat16
                kv_cache_dtype,                                 # auto, fp8_e4m3, etc
                block_size,                                     # From cache_config: 16
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

---

## PATCH 3: Kernel Backend Selection Logic

**File:** `vllm/attention/selector.py` (Lines 163-215)

```python
@cache
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

    # ======= PRIORITY 1: Global forced backend =======
    selected_backend = None
    backend_by_global_setting: Optional[_Backend] = (
        get_global_forced_attn_backend())
    if backend_by_global_setting is not None:
        selected_backend = backend_by_global_setting
        logger.info(f"Using globally forced attention backend: {selected_backend}")
    else:
        # ======= PRIORITY 2: Environment variable =======
        backend_by_env_var: Optional[str] = envs.VLLM_ATTENTION_BACKEND
        if backend_by_env_var is not None:
            if backend_by_env_var.endswith("_VLLM_V1"):
                backend_by_env_var = backend_by_env_var.removesuffix("_VLLM_V1")
            selected_backend = backend_name_to_enum(backend_by_env_var)
            if selected_backend is None:
                raise ValueError(
                    f"Invalid attention backend: '{backend_by_env_var}'. "
                    f"Valid backends are: {list(_Backend.__members__.keys())}")
            logger.info(f"Using attention backend from env var: {selected_backend}")

    # ======= PRIORITY 3: Platform-specific auto-detection =======
    attention_cls = current_platform.get_attn_backend_cls(
        selected_backend,           # Can be None for auto-detection
        head_size,                  # e.g., 64
        dtype,                      # e.g., torch.float16
        kv_cache_dtype,            # e.g., "auto"
        block_size,                # e.g., 16
        use_v1,                    # True for modern vLLM
        use_mla,                   # False for GPT2
        has_sink,                  # False for standard models
        use_sparse                 # False for GPT2
    )
    
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}")
    
    # ======= KERNEL RESOLUTION =======
    backend_class = resolve_obj_by_qualname(attention_cls)
    logger.info(f"Selected attention backend: {backend_class.__name__}")
    return backend_class
```

---

## PATCH 4: CUDA Kernel Runtime Binding

**File:** `vllm/attention/utils/fa_utils.py` (Lines 13-18)

```python
from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# ======= RUNTIME KERNEL BINDING =======
if current_platform.is_cuda():
    # Import generic CUDA ops
    from vllm import _custom_ops as ops
    reshape_and_cache_flash = ops.reshape_and_cache_flash
    
    # Import compiled FlashAttention kernels from .so files
    # These are built during vLLM installation from vllm-flash-attn
    from vllm.vllm_flash_attn import (
        flash_attn_varlen_func,        # Main attention kernel (compiled CUDA)
        get_scheduler_metadata
    )
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops
    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = ops.flash_attn_varlen_func
    get_scheduler_metadata = ops.get_scheduler_metadata
```

---

## PATCH 5: TP=2 Worker Communication - ColumnParallelLinear

**File:** `vllm/model_executor/layers/linear.py` (Lines 405-560)

### Forward Pass with TP=2

```python
class ColumnParallelLinear(LinearBase):
    """
    Linear layer with column parallelism (splits output dimension).
    Y = XA where A = [A_1, ..., A_p] across p workers
    
    With TP=2:
    - Worker 0: computes Y_0 = X @ A_0 (hidden_size -> output_size/2)
    - Worker 1: computes Y_1 = X @ A_1 (hidden_size -> output_size/2)
    """

    def __init__(self, ...):
        # Key TP setup for column-parallel
        self.tp_rank = (get_tensor_model_parallel_rank()      # 0 or 1
                        if not disable_tp else 0)
        self.tp_size = (get_tensor_model_parallel_world_size() # 2 for TP=2
                        if not disable_tp else 1)
        self.input_size_per_partition = input_size             # unchanged
        self.output_size_per_partition = divide(output_size, self.tp_size)  # /2
        
        self.gather_output = gather_output                     # False for c_attn

    def forward(self, input_: torch.Tensor):
        # input_: shape (batch_size, seq_len, hidden_size)
        # Same input on Worker 0 and Worker 1 (broadcast)
        
        bias = self.bias if not self.skip_bias_add else None

        # ======= COMPUTE: Each worker computes its partition =======
        # Worker 0: output_parallel_0 = input @ weight_0 (shape: ..., output_size//2)
        # Worker 1: output_parallel_1 = input @ weight_1 (shape: ..., output_size//2)
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)

        # ======= GATHER: Only if gather_output=True =======
        if self.gather_output and self.tp_size > 1:
            # AllGather across partitions
            # [output_parallel_0] on GPU 0, [output_parallel_1] on GPU 1
            #          ↓ AllGather(NCCL)
            # [output_parallel_0 + output_parallel_1] on both GPUs
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            # No communication for c_attn (gather_output=False)
            # Worker 0 keeps output_parallel_0
            # Worker 1 keeps output_parallel_1
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias
```

### Imports for TP Communication

```python
# Lines 14-16: Communication primitives
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,      # ← AllGather operation
    tensor_model_parallel_all_reduce,      # ← AllReduce operation
    split_tensor_along_last_dim,           # ← Split operation
)
```

---

## PATCH 6: TP=2 Worker Communication - RowParallelLinear

**File:** `vllm/model_executor/layers/linear.py` (Lines 1197-1365)

### Forward Pass with TP=2

```python
class RowParallelLinear(LinearBase):
    """
    Linear layer with row parallelism (splits input dimension).
    Y = XA where X = [X_1, ..., X_p] across p workers
    
    With TP=2:
    - Worker 0: computes Y_0 = X_0 @ A_0 (input_size/2 -> output_size)
    - Worker 1: computes Y_1 = X_1 @ A_1 (input_size/2 -> output_size)
    - AllReduce: Y = Y_0 + Y_1
    """

    def __init__(self, ...):
        # Key TP setup for row-parallel
        self.tp_rank = (get_tensor_model_parallel_rank()           # 0 or 1
                        if not disable_tp else 0)
        self.tp_size = (get_tensor_model_parallel_world_size()     # 2 for TP=2
                        if not disable_tp else 1)
        self.input_size_per_partition = divide(input_size, self.tp_size)  # /2
        self.output_size_per_partition = output_size               # unchanged
        
        self.input_is_parallel = input_is_parallel                 # True
        self.reduce_results = reduce_results                       # True for c_proj

    def forward(self, input_: torch.Tensor):
        # input_: shape (batch_size, seq_len, hidden_size)
        
        # ======= SPLIT: Each worker gets its partition =======
        if self.input_is_parallel:
            # Input already scattered (from ColumnParallel output)
            input_parallel = input_
        else:
            # Split full input across workers
            # Worker 0: input_0 = input[:, :, :hidden_size//2]
            # Worker 1: input_1 = input[:, :, hidden_size//2:]
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # ======= COMPUTE: Each worker computes its partition =======
        assert self.quant_method is not None
        # Only rank 0 adds bias (prevent duplicate addition)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        
        # Worker 0: output_parallel_0 = input_parallel_0 @ A_0 + bias
        # Worker 1: output_parallel_1 = input_parallel_1 @ A_1
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        # ======= REDUCE: AllReduce to combine results =======
        if self.reduce_results and self.tp_size > 1:
            # AllReduce (sum operation)
            # output_parallel_0 on GPU 0, output_parallel_1 on GPU 1
            #          ↓ AllReduce(NCCL, op=SUM)
            # (output_parallel_0 + output_parallel_1) on both GPUs
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            # No reduction
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias
```

---

## Complete GPT2 Forward Pass with TP=2

**File:** `vllm/model_executor/models/gpt2.py` (Lines 96-102)

```python
class GPT2Attention(nn.Module):

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Input: hidden_states shape (batch, seq_len, hidden_size)
        # Distributed: Same tensor replicated on Worker 0 and Worker 1
        
        # ======= STEP 1: Project to QKV (ColumnParallel) =======
        # scatter_output=False → each worker keeps its partition
        qkv, _ = self.c_attn(hidden_states)
        # Worker 0: qkv_0 shape (..., 3*hidden_size//2)  [no communication]
        # Worker 1: qkv_1 shape (..., 3*hidden_size//2)
        
        # ======= STEP 2: Split into Q, K, V =======
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        # Worker 0: q_0 shape (..., hidden_size//2), k_0, v_0
        # Worker 1: q_1 shape (..., hidden_size//2), k_1, v_1
        
        # ======= STEP 3: Attention Computation =======
        # The attention implementation handles tensor communication
        # (may AllGather q, k, v for full computation)
        attn_output = self.attn(q, k, v)
        # Output: gathered or broadcast across workers
        
        # ======= STEP 4: Project back (RowParallel) =======
        # reduce_results=True → AllReduce at the end
        attn_output, _ = self.c_proj(attn_output)
        # output = output_0 + output_1 (after AllReduce)
        # Both workers have same final output
        
        return attn_output
```

---

## NCCL Communication Summary

### AllGather Operation (ColumnParallel)
```
Worker 0 tensor: [A, B]     Worker 1 tensor: [C, D]
        ↓ AllGather(NCCL) ↓
Worker 0 result: [A, B, C, D]    Worker 1 result: [A, B, C, D]
(Concatenation: all workers get all data)
```

### AllReduce Operation (RowParallel)
```
Worker 0 tensor: [1, 2, 3]     Worker 1 tensor: [4, 5, 6]
        ↓ AllReduce(SUM, NCCL) ↓
Worker 0 result: [5, 7, 9]    Worker 1 result: [5, 7, 9]
(Reduction: all workers get summed result)
```

### Split Operation (for input_is_parallel=False)
```
Input on Worker 0: [A, B, C, D]
        ↓ SplitTensor ↓
Worker 0: [A, B]    Worker 1: [C, D]
(No communication: local operation)
```

---

## Environment Variables Affecting Kernel Selection

```bash
# Force specific attention backend
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ATTENTION_BACKEND=TRITON

# FlashAttention version
export VLLM_FLASH_ATTN_VERSION=2  # or 3

# Tensor parallelism workers
export VLLM_TENSOR_PARALLEL_SIZE=2

# Development override for local FlashAttention source
export VLLM_FLASH_ATTN_SRC_DIR=/path/to/local/flash-attn
```

---

## Key Implementation Files

1. **Kernel Selection:**
   - `vllm/attention/selector.py` - Selects backend based on hardware/config
   - `vllm/attention/layer.py` - Instantiates attention backend

2. **Attention Backends:**
   - `vllm/v1/attention/backends/flash_attn.py` - FlashAttention implementation
   - `vllm/v1/attention/backends/flex_attention.py` - FlexAttention fallback

3. **Distributed Communication:**
   - `vllm/model_executor/layers/linear.py` - ColumnParallel & RowParallel
   - `vllm/distributed/parallel_state.py` - NCCL collective operations

4. **Kernel Binding:**
   - `vllm/attention/utils/fa_utils.py` - FlashAttention kernel imports
   - `vllm/vllm_flash_attn/` - Compiled `.so` files (built at install time)
