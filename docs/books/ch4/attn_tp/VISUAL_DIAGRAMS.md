# Visual Flow Diagrams: Attention Kernel Mapping and TP Communication

## Diagram 1: Attention Kernel Selection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  GPT2 Model Initialization                                     │
│  gpt2.py: GPT2Attention.__init__()                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Attention Layer Creation                                       │
│  attention/layer.py: Attention.__init__()                      │
│                                                                  │
│  Parameters:                                                     │
│  - num_heads = 12                                              │
│  - head_size = 64                                              │
│  - dtype = torch.float16                                       │
│  - cache_config.block_size = 16                                │
│  - cache_config.cache_dtype = "auto"                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                   get_attn_backend()
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Priority 1: Check Global Force Override                        │
│  forced_attn_backend = None? → Continue                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Priority 2: Check Environment Variable                         │
│  VLLM_ATTENTION_BACKEND = None? → Continue                      │
│  (e.g., could be FLASH_ATTN, XFORMERS, FLEX_ATTENTION)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Priority 3: Platform-Specific Auto-Detection                   │
│  current_platform.get_attn_backend_cls()                        │
│                                                                  │
│  Checks:                                                         │
│  - GPU Capability (CUDA, ROCm, etc.)                           │
│  - Head size support (32, 64, 96, 128, 160, 192, 224, 256)     │
│  - Dtype support (float16, bfloat16)                           │
│  - CUDA version (for FA3 requires 12.3+)                       │
│  - SM capability (FA3 requires SM 90)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Decision Tree for CUDA:
                              ↓
        ┌───────────────────────────────────────────────┐
        │ Is CUDA available and device SM 90 (Hopper)?  │
        └───────────────────────────────────────────────┘
        ↙ Yes                                        ↖ No
        │                                             │
┌──────────────────────────────┐        ┌────────────────────────────┐
│ FlashAttention V3            │        │ Check other backends       │
│ ._vllm_fa3_C.abi3.so        │        │                            │
│ (CUDA 12.3+, SM 90)         │        │ - FlashAttention V2        │
│                              │        │   ._vllm_fa2_C.abi3.so    │
│ Special Features:            │        │ - FlexAttention            │
│ - FP8 support              │        │ - Xformers                 │
│ - MLA support              │        │ - Triton                   │
│ - Max perf on Hopper       │        │ - or fallback              │
└──────────────────────────────┘        └────────────────────────────┘
        ↓                                 ↓
        └─────────────────┬───────────────┘
                          ↓
            ┌─────────────────────────────────┐
            │ resolve_obj_by_qualname()       │
            │ Load actual backend class       │
            └─────────────────────────────────┘
                          ↓
         ┌────────────────────────────────────────┐
         │ Runtime Kernel Binding                 │
         │ from vllm.vllm_flash_attn import       │
         │   flash_attn_varlen_func               │
         │   get_scheduler_metadata               │
         │                                         │
         │ (Compiled CUDA kernels from .so files) │
         └────────────────────────────────────────┘
                          ↓
         ┌────────────────────────────────────────┐
         │ Attention.impl initialized with        │
         │ selected backend implementation        │
         │                                         │
         │ Ready for forward() calls               │
         └────────────────────────────────────────┘
```

---

## Diagram 2: GPT2 Forward Pass with TP=2

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU 0 (Worker 0)                             │
│                    GPU 1 (Worker 1)                             │
└─────────────────────────────────────────────────────────────────┘

                    INPUT PREPARATION
                            ↓
      ┌─────────────────────────────────────┐
      │ hidden_states: (batch, seq_len, D)  │
      │ where D = hidden_size = 768         │
      │                                      │
      │ Broadcast to both workers:           │
      │ GPU 0: hidden_states [replicated]   │
      │ GPU 1: hidden_states [replicated]   │
      └─────────────────────────────────────┘

                    STEP 1: QKVParallelLinear
            (ColumnParallel - scatter output)
                            ↓
      ┌─────────────────────────────────────┐
      │ self.c_attn(hidden_states)          │
      │                                      │
      │ GPU 0 Weight: A_0  (D × 3D/2)       │
      │ GPU 1 Weight: A_1  (D × 3D/2)       │
      │                                      │
      │ Computation:                         │
      │ GPU 0: qkv_0 = hidden @ A_0         │
      │        Shape: (..., 3D/2)           │
      │                                      │
      │ GPU 1: qkv_1 = hidden @ A_1         │
      │        Shape: (..., 3D/2)           │
      └─────────────────────────────────────┘
                            ↓
      ┌─────────────────────────────────────┐
      │ NO AllGather (gather_output=False)   │
      │                                      │
      │ GPU 0: qkv_0 STAYS on GPU 0        │
      │ GPU 1: qkv_1 STAYS on GPU 1        │
      │                                      │
      │ Each worker has PARTIAL tensor      │
      └─────────────────────────────────────┘

                    STEP 2: Split QKV
                            ↓
      ┌─────────────────────────────────────┐
      │ q, k, v = qkv.chunk(3, dim=-1)      │
      │                                      │
      │ GPU 0:                               │
      │   q_0: (..., D/4)  [first half]     │
      │   k_0: (..., D/4)                    │
      │   v_0: (..., D/4)                    │
      │                                      │
      │ GPU 1:                               │
      │   q_1: (..., D/4)  [second half]    │
      │   k_1: (..., D/4)                    │
      │   v_1: (..., D/4)                    │
      │                                      │
      │ Total across workers: full Q, K, V   │
      └─────────────────────────────────────┘

                    STEP 3: Attention Computation
            (flash_attn_varlen_func - may gather internally)
                            ↓
      ┌─────────────────────────────────────┐
      │ attn_output = self.attn(q, k, v)    │
      │                                      │
      │ Internal attention flow:             │
      │ - May AllGather q, k, v tensors    │
      │ - Compute attention on full inputs   │
      │ - Output typically broadcast/shared  │
      │                                      │
      │ GPU 0: attn_output_0 (gathered)    │
      │ GPU 1: attn_output_1 (gathered)    │
      │                                      │
      │ Both workers have FULL output        │
      │ (or full intermediate for next layer)│
      └─────────────────────────────────────┘

                    STEP 4: RowParallelLinear
            (Split input, reduce output)
                            ↓
      ┌─────────────────────────────────────┐
      │ self.c_proj(attn_output)            │
      │                                      │
      │ Input split (input_is_parallel=F):   │
      │ GPU 0: attn_output_0 = full[:,:,:D/2]│
      │ GPU 1: attn_output_1 = full[:,  :,D/2:]│
      │                                      │
      │ GPU 0 Weight: A_0  (D/2 × D)       │
      │ GPU 1 Weight: A_1  (D/2 × D)       │
      │                                      │
      │ Computation:                         │
      │ GPU 0: out_0 = attn_output_0 @ A_0  │
      │        + bias (only on rank 0)       │
      │        Shape: (..., D)               │
      │                                      │
      │ GPU 1: out_1 = attn_output_1 @ A_1  │
      │        Shape: (..., D)               │
      └─────────────────────────────────────┘
                            ↓
      ┌─────────────────────────────────────┐
      │ AllReduce(SUM) - collect results    │
      │                                      │
      │ GPU 0: out_0  ──┐                   │
      │                  ├─→ AllReduce ────┐│
      │ GPU 1: out_1  ──┘                   ││
      │                                     ││
      │ Result: out_0 + out_1              ││
      │                                     ││
      │ GPU 0: output = out_0 + out_1 ←────┤│
      │ GPU 1: output = out_0 + out_1 ←────┘│
      │                                      │
      │ Both workers have SAME final output  │
      │ Ready for next layer (norm, MLP)    │
      └─────────────────────────────────────┘
```

---

## Diagram 3: Communication Operations Detail

### AllGather Operation (ColumnParallel with gather_output=True)

```
Timeline:
Before AllGather:
┌──────────────────┬──────────────────┐
│  GPU 0           │  GPU 1           │
├──────────────────┼──────────────────┤
│ Tensor A         │ Tensor C         │
│ ┌──────────┐     │ ┌──────────┐     │
│ │ [1,2]    │     │ │ [3,4]    │     │
│ └──────────┘     │ └──────────┘     │
└──────────────────┴──────────────────┘

             ↓ AllGather() [NCCL]

After AllGather:
┌──────────────────────────────────────┐
│  GPU 0                               │
├──────────────────────────────────────┤
│ Tensor A + C                         │
│ ┌──────────────────────────────────┐ │
│ │ [1, 2, 3, 4]                    │ │
│ └──────────────────────────────────┘ │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  GPU 1                               │
├──────────────────────────────────────┤
│ Tensor A + C                         │
│ ┌──────────────────────────────────┐ │
│ │ [1, 2, 3, 4]                    │ │
│ └──────────────────────────────────┘ │
└──────────────────────────────────────┘

Result: Both workers have concatenated data
```

### AllReduce Operation (RowParallel with reduce_results=True)

```
Timeline:
Before AllReduce:
┌──────────────────┬──────────────────┐
│  GPU 0           │  GPU 1           │
├──────────────────┼──────────────────┤
│ Tensor A         │ Tensor B         │
│ ┌──────────┐     │ ┌──────────┐     │
│ │ [1,2,3]  │     │ │ [4,5,6]  │     │
│ └──────────┘     │ └──────────┘     │
└──────────────────┴──────────────────┘

             ↓ AllReduce(SUM) [NCCL]

After AllReduce:
┌──────────────────┬──────────────────┐
│  GPU 0           │  GPU 1           │
├──────────────────┼──────────────────┤
│ Result = A + B   │ Result = A + B   │
│ ┌──────────┐     │ ┌──────────┐     │
│ │ [5,7,9]  │     │ │ [5,7,9]  │     │
│ └──────────┘     │ └──────────┘     │
└──────────────────┴──────────────────┘

Result: Both workers have summed data
Operation: Element-wise addition across all workers
```

### Split Operation (RowParallel input preparation)

```
Before Split:
┌──────────────────────────────────────────┐
│  GPU 0 (Full Tensor on GPU 0)            │
├──────────────────────────────────────────┤
│ Input                                    │
│ ┌──────────────────────────────────────┐ │
│ │ [A₁, A₂ | B₁, B₂]                  │ │
│ └──────────────────────────────────────┘ │
└──────────────────────────────────────────┘

             ↓ split_tensor_along_last_dim()

After Split (No NCCL - Local Operation):
┌──────────────────┬──────────────────┐
│  GPU 0           │  GPU 1           │
├──────────────────┼──────────────────┤
│ Left Half        │ Right Half       │
│ ┌──────────┐     │ ┌──────────┐     │
│ │ [A₁, A₂] │     │ │ [B₁, B₂] │     │
│ └──────────┘     │ └──────────┘     │
└──────────────────┴──────────────────┘

Result: Each worker has independent partition (no communication)
```

---

## Diagram 4: Complete Forward Pass Sequence with Timelines

```
TIME  OPERATION                    GPU 0                GPU 1
════════════════════════════════════════════════════════════════
T0    Input Broadcast              ✓ hidden_states      ✓ hidden_states
      (no comm needed,              (replicated)         (replicated)
       pre-broadcast)

T1    ColumnParallel (c_attn)
      - Compute                     → qkv_0              → qkv_1
      - No gather                   (local, no sync)     (local, no sync)

T2    Attention (may need internal communication)
      - If needs AllGather:        ↔ Sync Point: AllGather ↔
      - Compute attention
      - Result:                     ✓ attn_out           ✓ attn_out
                                    (both workers        (same)
                                     synchronized)

T3    RowParallel (c_proj)
      - Split input                 → left partition     → right partition
      - Compute                     → output_0           → output_1
                                    (local compute)      (local compute)

T4    AllReduce                    ↔ Sync Point: AllReduce ↔
                                   (combine: sum all)

T5    Result                        ✓ final_output       ✓ final_output
                                    (same value on all)  (same value on all)

════════════════════════════════════════════════════════════════
Sync Points (NCCL collective operations):
- T2: AllGather (may be implicit in attention impl)
- T4: AllReduce (explicit in RowParallel)

Communication Overhead:
- AllGather: bandwidth-bound (concatenation)
- AllReduce: bandwidth + latency (reduction across network)
- Split: no communication (local slicing)
```

---

## Diagram 5: Attention Backend Decision Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                 ATTENTION BACKEND SELECTION MATRIX                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ HEAD_SIZE          DTYPE              GPU           BACKEND         │
│ ════════════════════════════════════════════════════════════════    │
│                                                                      │
│ 32, 64, 96,        float16,           NVIDIA        FlashAttn V3    │
│ 128, 160, 192,     bfloat16           Hopper        (CUDA 12.3+)    │
│ 224, 256                              (SM 90)       .abi3.so        │
│                                                      ✓ BEST          │
│ ─────────────────────────────────────────────────────────────────   │
│                                                                      │
│ 32, 64, 96,        float16,           NVIDIA        FlashAttn V2    │
│ 128, 160, 192,     bfloat16           Pre-Hopper    .abi3.so        │
│ 224, 256                              (SM 80, 89)   ✓ Good          │
│                                                                      │
│ ─────────────────────────────────────────────────────────────────   │
│                                                                      │
│ Any                Any                NVIDIA        FlexAttention   │
│                                       (fallback)    (Triton)        │
│                                                      ✓ Fallback      │
│                                                                      │
│ ─────────────────────────────────────────────────────────────────   │
│                                                                      │
│ 32, 64,            float16,           AMD           Triton/         │
│ 128, 160, 192,     bfloat16           (ROCm)        xFormers        │
│ 224, 256                                            ✓ Limited       │
│                                                                      │
│ ─────────────────────────────────────────────────────────────────   │
│                                                                      │
│ Any                Any                CPU           CPU Fallback    │
│                                       (inference)   (naive attn)     │
│                                                      ✓ Slow          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Environment Variable Overrides:
export VLLM_ATTENTION_BACKEND=FLASH_ATTN      → Force FlashAttention
export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION  → Force FlexAttention
export VLLM_ATTENTION_BACKEND=XFORMERS        → Force xFormers
export VLLM_ATTENTION_BACKEND=TRITON          → Force Triton

Selection Priority:
1. Programmatic force override (highest priority)
2. Environment variable
3. Auto-detection based on hardware (lowest priority)
```

---

## Diagram 6: NCCL Operations on Network

```
Multiple Machines with NVLink/Ethernet:

Machine 1                    Machine 2
┌──────────────┐            ┌──────────────┐
│  GPU 0       │            │  GPU 1       │
│              │            │              │
│ Tensor_0     │  NVLink/   │ Tensor_1     │
│              │  Ethernet  │              │
└──────────────┘            └──────────────┘
       ↑                           ↑
       └───────────────────────────┘

AllGather Result:
┌──────────────┐            ┌──────────────┐
│  GPU 0       │            │  GPU 1       │
│              │            │              │
│ [Tensor_0 |  │            │ [Tensor_0 |  │
│  Tensor_1]   │            │  Tensor_1]   │
└──────────────┘            └──────────────┘

AllReduce Result:
┌──────────────┐            ┌──────────────┐
│  GPU 0       │            │  GPU 1       │
│              │            │              │
│ Tensor_0 +   │            │ Tensor_0 +   │
│ Tensor_1     │            │ Tensor_1     │
└──────────────┘            └──────────────┘

Bandwidth Utilization:
- NVLink: 600 GB/s (full duplex)
- PCIe Gen4: 64 GB/s (full duplex)
- 100Gbps Ethernet: 12.5 GB/s
```

---

## Summary Table: TP=2 Communication Cost

```
╔════════════════════════════════════════════════════════════════╗
║              COMMUNICATION COST ANALYSIS (TP=2)                ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║ Operation          Type          Data Size    Latency Impact   ║
║ ────────────────────────────────────────────────────────────   ║
║                                                                ║
║ ColumnParallel     Scatter       ~D*B         None             ║
║ (c_attn)           (no comm)      (local)      (no sync)        ║
║                                                                ║
║ Attention          AllGather     ~3*D*B*S    ~1-5 µs           ║
║ (implicit)         (gather q,k,v)            per element       ║
║                                                                ║
║ RowParallel        AllReduce     ~D*B        ~1-5 µs           ║
║ (c_proj)           (sum)         (full out)   per element       ║
║                                                                ║
║ ────────────────────────────────────────────────────────────   ║
║                                                                ║
║ Where:                                                         ║
║ - D = hidden_size (e.g., 768)                                 ║
║ - B = batch_size                                              ║
║ - S = seq_length                                              ║
║                                                                ║
║ Total Collective Operations per Forward Pass: 2-3             ║
║ (depending on attention implementation)                       ║
║                                                                ║
║ Communication Overhead: 5-15% of total inference time         ║
║ (vs. 100% if purely sequential)                               ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```
