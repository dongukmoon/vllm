# PagedAttention vs FlashAttention: Comprehensive Comparison

## Quick Summary

| Aspect | PagedAttention | FlashAttention |
|--------|----------------|----------------|
| **Purpose** | Memory-efficient KV cache management | Fast attention computation algorithm |
| **Key Focus** | Block-based storage with indirection tables | Reduced memory I/O and GPU bandwidth |
| **Kernel Type** | Procedural (CUDA/CPU) | Algorithmic (fused kernel) |
| **Use Case** | Multi-sequence batching | Single/batch attention computation |
| **Memory Model** | Paged (non-contiguous blocks) | Streaming (I/O aware) |

---

## Conceptual Differences

### PagedAttention

**Goal:** Enable efficient memory management for KV caches in multi-request scenarios

**Key Idea:** Organize KV cache into fixed-size blocks with indirection tables (block tables)

```
Physical GPU Memory:
[Block 0] [Block 5] [Block 2] [Block 8] ...
  ↑                    ↑
Scattered in memory

Request's Block Table:
[0, 5, 2, 8, ...]  → Maps logical blocks to physical block IDs
```

**Advantages:**
- Variable sequence lengths in same batch (no padding waste)
- Efficient block sharing via prefix caching
- Reduced memory fragmentation
- Non-contiguous allocation flexibility

**Trade-off:**
- Requires block table indirection (extra memory access)
- Not optimized for single-sequence latency

---

### FlashAttention

**Goal:** Reduce attention computation time through I/O-aware algorithm design

**Key Idea:** Reorder computation to minimize GPU memory traffic by exploiting fast SRAM

```
Traditional Attention Memory Flow:
HBM (GPU Memory) → GPU Core → HBM
[Slow I/O]         [Fast]      [Slow I/O]

FlashAttention Flow:
HBM → SRAM (Fast!) → HBM
      [Minimize data movement]
```

**Algorithm Strategy:**
1. Load small blocks of Q, K, V into fast SRAM
2. Compute attention within SRAM
3. Write output back to HBM
4. Repeat for all blocks (tiling)

**Advantages:**
- 2-4x faster than standard attention
- Reduces memory bandwidth requirements
- Lower power consumption
- Better GPU utilization

**Trade-off:**
- Requires specific GPU architecture support
- Limited head size support (must be compile-time configured)
- Works best with contiguous memory

---

## Implementation Differences

### PagedAttention Kernel Structure

**File:** [csrc/attention/paged_attention_v1.cu](csrc/attention/paged_attention_v1.cu)

```cuda-cpp
// Kernel parameters focused on MEMORY INDIRECTION
void paged_attention_v1_kernel(
    T* out,                    // Output
    T* query,                  // Current query
    CACHE_T* key_cache,        // ALL K blocks in GPU memory
    CACHE_T* value_cache,      // ALL V blocks in GPU memory
    int* block_tables,         // [num_seqs, max_blocks] - KEY DIFFERENCE!
    int* seq_lens,             // Sequence lengths
    ...
);

// Algorithm:
// 1. For each sequence:
//    - Read from block_tables[seq_id] to find physical block addresses
//    - Use block indirection to access non-contiguous KV blocks
//    - Perform attention computation on scattered blocks
```

**Memory Access Pattern:**
```
Thread 0: Read block_table[seq_0, block_0] → physical_block_5
          Load key_cache[block_5, ...]  (non-contiguous jump)
          
Thread 1: Read block_table[seq_1, block_0] → physical_block_2
          Load key_cache[block_2, ...]  (different sequence)
          
Result: Random memory access pattern, good batch efficiency
```

---

### FlashAttention Kernel Structure

**File:** [vllm/v1/attention/backends/flash_attn.py](vllm/v1/attention/backends/flash_attn.py)

```python
class FlashAttentionImpl(AttentionImpl):
    def forward(
        self,
        layer,
        query: torch.Tensor,       # [num_tokens, num_heads, head_size]
        key: torch.Tensor,         # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,       # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,    # Contiguous cache storage
        attn_metadata: FlashAttentionMetadata,
        ...
    ) -> torch.Tensor:
        # Call external FlashAttention library (e.g., flash_attn_varlen_func)
        # Kernel is COMPILED SEPARATELY with specific block sizes
```

**Algorithm (from triton_flash_attention.py):**
```python
# Triton implementation of Flash Attention v2
# Key concepts:
# 1. Tile outer product computation
# 2. Store intermediate results in SRAM
# 3. Recompute attention weights in backward pass
# 4. Minimize HBM traffic

def flash_attention_kernel(Q, K, V, scale, ...):
    # Load BLOCK_M rows of Q into SRAM
    # For each block of K:
    #   - Compute QK^T (stays in SRAM)
    #   - Apply softmax (in SRAM)
    #   - Compute attention values (stays in SRAM)
    #   - Write results to HBM
```

**Memory Access Pattern:**
```
Sequential Loading:
Load Q[0:BLOCK_M]   → SRAM (sequential)
Load K[0:BLOCK_N]   → SRAM (sequential)
Compute in SRAM
Write Output         → HBM (sequential)

Result: Highly predictable, cache-friendly access
```

---

## Architectural Comparison

### PagedAttention Architecture

```
┌─────────────────────────────────────────┐
│         KV Cache Manager                │
│  (Manages block allocation & freeing)   │
└────────────────┬────────────────────────┘
                 │
┌────────────────v────────────────────────┐
│      Block Table (Indirection)          │
│  [Request 1] [Request 2] [Request 3]   │
│  [0,5,12,18] [2,3,8,15] [5,9,...]      │
└────────────────┬────────────────────────┘
                 │
┌────────────────v────────────────────────┐
│    Physical GPU Memory (Blocks)         │
│  Block₀ Block₁ Block₂ ... Block₁₀₀     │
│  (K,V)  (K,V)  (K,V)  ... (K,V)        │
│  └─ Non-contiguous allocation           │
└─────────────────────────────────────────┘
```

**Key Components:**
- `KVCacheManager`: High-level allocation
- `Block Table`: Indirection table per request
- `FreeKVCacheBlockQueue`: LRU eviction management
- `KVCacheBlock`: Metadata for each block

---

### FlashAttention Architecture

```
┌──────────────────────────────────────────┐
│      Input (Q, K, V tensors)             │
│     [num_tokens, num_heads, head_size]   │
└──────────────┬───────────────────────────┘
               │
┌──────────────v───────────────────────────┐
│   FlashAttention Kernel Launch           │
│  (Fused, compiled with specific config)  │
└──────────────┬───────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
       v               v
   ┌─────┐         ┌────────┐
   │SRAM │         │HBM     │
   │Fast │←───────→│Slow    │
   │     │   I/O   │        │
   └─────┘ minimal └────────┘
   (Fast!)  
```

**Key Components:**
- `Flash Attention Algorithm`: I/O-aware computation
- `SRAM Tiling`: Load small blocks into fast memory
- `Softmax Recomputation`: Trade compute for memory saving
- `Fused Kernel`: Q, K, V computation combined

---

## When to Use Each

### Use PagedAttention When:

✅ **Multi-request batch processing** (LLM serving)
```python
batch_size = 32
variable_seq_lens = [100, 256, 512, 1024, ...]  # Mixed lengths
# Requests share GPU memory efficiently
```

✅ **Prefix caching benefits** (repeated prompts)
```python
# Request 1: "Once upon a time... [100 new tokens]"
# Request 2: "Once upon a time... [200 new tokens]"
# Request 3: "Once upon a time... [150 new tokens]"
# ↓ Reuse first block across all requests
```

✅ **Memory efficiency priority** (maximize throughput)
```python
# Block-based allocation prevents fragmentation
# Can fit more sequences in same GPU memory
```

✅ **Complex attention patterns** (sliding window, sparse)
```python
# Block table indirection supports any pattern
# No recompilation needed for different window sizes
```

---

### Use FlashAttention When:

✅ **Single sequence latency** (chat inference)
```python
batch_size = 1
seq_len = 4096
# Need ultra-low latency per token
```

✅ **High throughput with fixed patterns** (batch prefill)
```python
# Batch of similar-length sequences
# Contiguous memory layout optimal
```

✅ **Compatibility with model architecture** (supported head sizes)
```python
head_size in [32, 64, 96, 128, 160, 192, 224, 256]
# Works efficiently with these dimensions
```

✅ **Standard attention only** (no special patterns)
```python
# Full attention, causal masking, sliding window
# ALiBi, logits soft-cap supported
# But not block-sparse or other exotic patterns
```

---

## Performance Characteristics

### Memory Bandwidth Comparison

**Standard Attention:**
```
HBM → Compute → HBM
O(N²) intermediate matrix storage
Bandwidth: ~4x K cache + ~4x V cache loads
```

**PagedAttention:**
```
HBM → Compute → HBM
O(N²) intermediate matrix storage (unchanged)
+ Block table indirection overhead (~0.1% extra)
Bandwidth: Same + indirection lookups
```

**FlashAttention:**
```
HBM → SRAM → HBM
O(1) intermediate storage (stays in SRAM!)
Bandwidth: ~2x K cache + ~2x V cache loads
Reduction: 50% of bandwidth vs standard
```

### Computational Complexity

| Operation | Time | Space | I/O |
|-----------|------|-------|-----|
| Standard Attn | O(N²d) | O(N²) | O(Nd) |
| PagedAttention | O(N²d) | O(N²) | O(Nd) + indirection |
| FlashAttention | O(N²d) | O(Nd) | O(Nd) ✓ optimal |

---

## Code Example: Using Both Together

vLLM often combines both approaches:

```python
# In vllm/v1/attention/backends/flash_attn.py

class FlashAttentionImpl(AttentionImpl):
    def forward(self, query, key, value, kv_cache, attn_metadata):
        
        # Step 1: Handle KV cache (PagedAttention style)
        key_cache, value_cache = kv_cache.unbind(0)
        
        # Update cache with new keys/values (scattered blocks)
        reshape_and_cache_flash(
            key, value,
            key_cache, value_cache,
            attn_metadata.slot_mapping,  # Paging!
            ...
        )
        
        # Step 2: Compute attention (FlashAttention style)
        # Use block_table indirection + FlashAttention kernel
        flash_attn_varlen_func(
            query,
            key_cache,      # Non-contiguous blocks
            value_cache,    # Scattered in memory
            cu_seqlens_q,
            seqused_k,
            max_seqlen_q,
            self.scale,
            block_table=attn_metadata.block_table,  # Paging!
            ...
        )
```

**Key Insight:** Modern systems like vLLM use PagedAttention for **memory management** and FlashAttention for **computation efficiency** together!

---

## GPU Support Matrix

### PagedAttention
- **NVIDIA CUDA**: All architectures (compute capability ≥ 6.0)
- **AMD ROCm**: All RDNA/CDNA architectures
- **CPU**: OpenMP CPU implementations for testing
- **Block-sparse variant**: Optional, compile-time feature

### FlashAttention
- **NVIDIA CUDA**: Ampere+ (A100, RTX30xx, H100)
- **AMD ROCm**: RDNA2+ (RX 6000, MI300)
- **Head sizes**: Only [32, 64, 96, 128, 160, 192, 224, 256] (compiled)
- **Data types**: FP16, BF16 (FP8 requires FA3)

---

## Summary Table

```
┌──────────────────────┬──────────────────┬──────────────────┐
│ Aspect               │ PagedAttention   │ FlashAttention   │
├──────────────────────┼──────────────────┼──────────────────┤
│ Abstraction Level    │ Memory/Storage   │ Computation      │
│ Primary Benefit      │ Flexibility      │ Speed            │
│ Memory I/O           │ O(Nd)            │ O(Nd) [optimized]│
│ Intermediate Storage │ O(N²) full       │ O(Nd) streaming  │
│ Batch Diversity      │ Excellent        │ Good             │
│ Latency (1 seq)      │ OK               │ Excellent        │
│ Throughput (batch)   │ Excellent        │ Excellent        │
│ Configurability      │ High             │ Limited          │
│ Prefix Caching       │ Native           │ Compatible       │
│ Special Patterns     │ Flexible         │ Predefined       │
│ GPU Support          │ Broad            │ Modern only      │
│ Compile Time         │ Once             │ Per-config       │
└──────────────────────┴──────────────────┴──────────────────┘
```

---

## Conclusion

**PagedAttention** and **FlashAttention** solve different problems:

- **PagedAttention**: "How do we manage memory efficiently for multiple requests?"
  - Answer: Block-based allocation with indirection tables
  
- **FlashAttention**: "How do we compute attention faster?"
  - Answer: Minimize GPU memory bandwidth through I/O-aware algorithm

**Best Practice:** Use both together
- PagedAttention for KV cache management (flexible, high-throughput batching)
- FlashAttention for attention computation (low-latency, efficient)
- Result: Fast, memory-efficient LLM serving for both single and batch queries
