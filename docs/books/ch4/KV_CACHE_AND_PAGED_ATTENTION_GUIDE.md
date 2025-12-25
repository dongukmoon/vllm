# KV Cache Management and PagedAttention in vLLM

## Overview
vLLM implements **PagedAttention**, a memory-efficient technique that organizes KV (Key-Value) caches into fixed-size blocks, similar to virtual memory paging. This allows for efficient memory management and enables **prefix caching** - reusing cached KV values across similar requests.

---

## Architecture Components

### 1. KV Cache Interface ([vllm/v1/kv_cache_interface.py](vllm/v1/kv_cache_interface.py))

#### Core Data Structures

**`KVCacheSpec`** (Base class)
- Defines the format of KV cache for one transformer layer
- Key property: `block_size` - number of tokens per block
- Provides `page_size_bytes` - memory size of a single block

**`AttentionSpec`** (Extends KVCacheSpec)
```python
@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int      # Number of KV attention heads
    head_size: int         # Head dimension size
    dtype: torch.dtype     # Data type of cached values
```
- **Page size calculation**: `2 * block_size * num_kv_heads * head_size * dtype_size`
  - Factor of 2 accounts for both Key and Value caches
  - Memory = (block_size tokens) × (num_kv_heads) × (head_size dims) × (bytes per value) × 2

**`FullAttentionSpec`** (Extends AttentionSpec)
- Standard full attention with optional sliding window support
- Supports both full-length attention and sliding window attention
- Memory calculated as: `(max_model_len / block_size) * page_size_bytes`

**`MLAAttentionSpec`** (Extends FullAttentionSpec)
- Multi-head Latent Attention variant
- Special handling for fp8_ds_mla: fixed page size of 656 bytes per block

**`SlidingWindowSpec`** and **`ChunkedLocalAttentionSpec`**
- Specialized attention patterns with different memory requirements

---

### 2. Block-Level Management ([vllm/v1/core/kv_cache_utils.py](vllm/v1/core/kv_cache_utils.py))

#### KVCacheBlock
```python
@dataclass
class KVCacheBlock:
    block_id: int                           # ID: 0 to num_gpu_blocks-1
    ref_cnt: int = 0                        # Reference count
    _block_hash: Optional[BlockHashWithGroupId] = None  # Prefix cache hash
    prev_free_block: Optional[KVCacheBlock] = None      # Linked list pointers
    next_free_block: Optional[KVCacheBlock] = None
    is_null: bool = False                   # Never cache null blocks
```

**Key Features:**
- **Metadata tracking**: Reference counting for safe deallocation
- **Prefix caching**: `block_hash` stores computed tokens for reuse
- **Free list**: Doubly-linked list for O(1) block management
- **Null blocks**: Placeholder blocks that should never be cached

#### FreeKVCacheBlockQueue
Implements an efficient free block management system using a doubly-linked list:

```python
class FreeKVCacheBlockQueue:
    def popleft(self) -> KVCacheBlock:      # Pop LRU block
    def popleft_n(self, n: int) -> list[KVCacheBlock]:  # Pop n blocks
    def remove(self, block: KVCacheBlock) -> None:     # Remove specific block
    def append_block(self, block: KVCacheBlock) -> None: # Return block
```

**Eviction Order:**
1. Least Recently Used (LRU) blocks at the front
2. Among blocks with same access time, blocks with more hash tokens (chain tail) are prioritized
3. Blocks are reversed when freed from a request to maintain order

**Performance Optimization:**
- Uses PyObject attributes for linked list (no additional allocations)
- Fake head/tail blocks reduce branching logic
- O(1) removal from queue middle

---

### 3. Manager-Level Control ([vllm/v1/core/kv_cache_manager.py](vllm/v1/core/kv_cache_manager.py))

#### KVCacheManager
Main interface between scheduler and KV cache system:

```python
class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        ...
    )
```

**Key Methods:**

**`get_computed_blocks(request) -> tuple[KVCacheBlocks, int]`**
- Retrieves cached (already computed) KV blocks for a request
- Used for **prefix caching**: reuses blocks across similar requests
- Returns cached blocks + number of computed tokens
- Skips cache if:
  - Prefix caching disabled
  - Request needs prompt logprobs (incompatible with caching)
- Safety: Recomputes last token even if entire cache hits (ensures valid logits)

**`allocate_slots(request, num_new_tokens, ...) -> Optional[KVCacheBlocks]`**
Allocates GPU memory blocks for new tokens:

```python
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: Optional[KVCacheBlocks] = None,
    num_lookahead_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
) -> Optional[KVCacheBlocks]:
```

**Block Layout (during allocation):**
```
[computed] [new_computed] [new] [pre-allocated]
[_________required_________][_____full_____]
                            [new_full]
```

**Steps:**
1. Free blocks skipped during attention (e.g., outside sliding window)
2. Calculate total blocks needed (computed + new + lookahead tokens)
3. Check if sufficient free blocks available (returns None if not)
4. Touch computed blocks to prevent eviction
5. Allocate new blocks from block pool
6. Cache blocks via prefix caching if enabled

---

### 4. Coordinator Level ([vllm/v1/core/kv_cache_coordinator.py](vllm/v1/core/kv_cache_coordinator.py))

Orchestrates block allocation across multiple KV cache groups:
- **Block pool management**: Tracks free/used blocks
- **Request tracking**: Maps request IDs to allocated blocks
- **Prefix cache coordination**: Manages block hashing and reuse
- **Multi-group support**: Handles models with different attention types

---

## PagedAttention Implementation

### CUDA Kernels

#### PagedAttention V1 ([csrc/attention/paged_attention_v1.cu](csrc/attention/paged_attention_v1.cu))

```cuda-cpp
template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128>
void paged_attention_v1_kernel(
    T* out,                    // [num_seqs, num_heads, head_size]
    T* query,                  // [num_seqs, num_heads, head_size]
    CACHE_T* key_cache,        // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    CACHE_T* value_cache,      // [num_blocks, num_kv_heads, head_size, block_size]
    int num_kv_heads,
    float scale,
    int* block_tables,         // [num_seqs, max_num_blocks_per_seq]
    int* seq_lens,             // [num_seqs]
    int max_num_blocks_per_seq,
    float* alibi_slopes,       // [num_heads]
    int q_stride, int kv_block_stride, int kv_head_stride,
    float* k_scale, float* v_scale,
    int tp_rank,
    int blocksparse_local_blocks, int blocksparse_vert_stride,
    int blocksparse_block_size, int blocksparse_head_sliding_step
);
```

**Algorithm:**
1. **Per-sequence block table**: `block_tables[seq_id]` maps logical token positions to physical block IDs
2. **QK computation**: For each block in sequence:
   - Load query from current position
   - Load K cache from block (physical_block_idx = block_tables[seq_id][block_idx])
   - Compute attention logits: `logits = Q @ K^T * scale`
3. **Softmax**: Normalize logits across all blocks
4. **Value aggregation**: Weighted sum of V values using softmax weights
5. **Output**: Attention output for this query position

**Key Design Features:**
- **Block-wise indirection**: `block_tables` provides flexible, non-contiguous block arrangement
- **Memory efficiency**: Only load blocks that contain tokens in the sequence
- **Supports quantization**: Can work with INT8 KV cache (fp8_ds_mla)
- **Sparse attention**: Optional block-sparse attention patterns
- **ALiBi slopes**: Supports Attention with Linear Biases for longer context

#### PagedAttention V2 ([csrc/attention/paged_attention_v2.cu](csrc/attention/paged_attention_v2.cu))

Enhanced version with multi-partition approach:

```cuda-cpp
void paged_attention_v2_kernel(
    float* exp_sums,           // [num_seqs, num_heads, max_num_partitions]
    float* max_logits,         // [num_seqs, num_heads, max_num_partitions]
    float* tmp_out,            // [num_seqs, num_heads, max_num_partitions, head_size]
    T* query,                  // [num_seqs, num_heads, head_size]
    CACHE_T* key_cache,        // [num_blocks, num_kv_heads, ...]
    CACHE_T* value_cache,      // [num_blocks, num_kv_heads, ...]
    ...
);
```

**Improvements:**
- **Partition-based reduction**: Reduces attention computation into partitions for memory efficiency
- **Intermediate results**: Stores exp_sums and max_logits for two-pass attention
- **Better memory utilization**: Avoids storing full attention matrix

### CPU Kernels

For CPU execution ([csrc/cpu/attention.cpp](csrc/cpu/attention.cpp)):
- Single-threaded and multi-threaded implementations
- Similar logic to CUDA but using OpenMP for parallelization
- Designed for testing and CPU inference scenarios

---

## Block Memory Layout

### Physical Storage Structure

```
GPU Memory (KV Cache):
┌─────────────────────────────────────────────────────┐
│ Block 0  │ Block 1  │ Block 2  │ ... │ Block N-1   │
├─────────────────────────────────────────────────────┤
│ K | V    │ K | V    │ K | V    │     │ K | V       │
└─────────────────────────────────────────────────────┘

Each Block = [num_heads, head_size/x, block_size, x]
```

**Data Layout Per Block:**
- **Key Cache**: `[num_kv_heads, head_size/x, block_size, x]`
  - Padding factor `x` for efficient memory access
  - Shape allows vectorized loads in CUDA kernels
  
- **Value Cache**: `[num_kv_heads, head_size, block_size]`
  - Direct layout for sequential access

**Indirection Table (Block Table):**
```
Request 1: [0, 5, 12, ...]    // Logical blocks 0,1,2 -> Physical blocks 0,5,12
Request 2: [2, 3, 8, ...]     // Logical blocks 0,1,2 -> Physical blocks 2,3,8
Request 3: [5, 9, ...]        // Logical blocks 0,1 -> Physical blocks 5,9
```

This allows:
- **Non-contiguous allocation**: Blocks scattered in GPU memory
- **Efficient reuse**: Requests can share computed blocks via prefix caching
- **Block sharing**: Multiple requests reference same physical blocks for prefixes

---

## Prefix Caching

### Block Hashing System

**Hash Computation:**
```python
# Hash represents computed tokens in a block
BlockHash = NewType("BlockHash", bytes)
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

def make_block_hash_with_group_id(block_hash: BlockHash, 
                                  group_id: int) -> BlockHashWithGroupId:
    # Combines 32-byte hash + 4-byte group ID
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big"))
```

**Usage:**
1. **Initial state**: Block created without hash (`_block_hash = None`)
2. **After computation**: Block receives hash based on tokens it contains
3. **Cache lookup**: New request checks if prefix tokens match existing block hashes
4. **Block reuse**: If hash matches, skip recomputation and reuse physical block
5. **Eviction**: When block freed, hash reset (`_block_hash = None`)

### Cache Hit Scenario

```
Request A: "Once upon a"          -> Tokens: [t1, t2, t3] -> Block 0 (hash: ABC)
Request B: "Once upon a time"     -> Tokens: [t1, t2, t3, t4]
                                     ↓ Cache hit on Block 0 with hash ABC
                                     → Reuse Block 0 + allocate Block 1 for t4

Result: 75% computation saved (3 out of 4 tokens from cache)
```

---

## Memory Management Lifecycle

### Request Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│ 1. REQUEST ARRIVES                                          │
│    - Check for prefix cache hits (get_computed_blocks)      │
│    - Determine num_new_tokens to allocate                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────v──────────────────────────────────────┐
│ 2. BLOCK ALLOCATION (allocate_slots)                        │
│    - Free blocks outside sliding window                     │
│    - Calculate total blocks needed                          │
│    - Check if free blocks available                         │
│    - Allocate from FreeKVCacheBlockQueue                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────v──────────────────────────────────────┐
│ 3. ATTENTION COMPUTATION                                    │
│    - Use block_tables[request_id] for indirection           │
│    - Load K/V from physical blocks via PagedAttention       │
│    - Compute attention output                               │
│    - Populate block_hash upon completion                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────v──────────────────────────────────────┐
│ 4. CACHE FINALIZATION                                       │
│    - Mark blocks as "computed" with hash                    │
│    - Save state for prefix caching in future requests       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────v──────────────────────────────────────┐
│ 5. REQUEST COMPLETION / EVICTION                            │
│    - Decrement ref_cnt on all blocks                        │
│    - Move blocks back to free queue                         │
│    - Reset block_hash when evicted                          │
│    - Blocks available for new requests                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration Example

### KV Cache Configuration

```python
# From vllm/v1/kv_cache_interface.py

kv_cache_spec = FullAttentionSpec(
    block_size=16,           # 16 tokens per block
    num_kv_heads=8,          # 8 KV heads
    head_size=64,            # 64-dim head
    dtype=torch.float16,     # FP16 storage
    sliding_window=None,     # Full attention (no window)
)

# Page size = 2 * 16 * 8 * 64 * 2 = 32,768 bytes = 32 KB
# For max_model_len=4096: num_blocks = ceil(4096/16) = 256 blocks
# Total memory ≈ 256 * 32 KB = 8 MB per layer per sequence
```

---

## Key Optimization Techniques

### 1. **Memory Efficiency Through Paging**
- Variable block utilization avoids wasting memory on partial blocks
- Enables batching sequences with different lengths

### 2. **Prefix Caching**
- Reuses computed blocks across similar requests
- Can dramatically reduce computation for common prefixes (e.g., system prompts)
- Configurable via `enable_caching` parameter

### 3. **Efficient Block Management**
- O(1) allocation/deallocation with linked list
- LRU eviction strategy via FreeKVCacheBlockQueue
- Smart eviction: prioritizes blocks with more hash tokens

### 4. **Quantized KV Cache**
- Support for INT8 (FP8) KV storage
- Reduces memory by 50% with minimal accuracy loss
- Both V1 and V2 kernels support quantization

### 5. **Batch Diversity**
- Block tables allow non-contiguous block allocation
- Enables mixing sequences of different lengths in same batch
- Improves GPU utilization

---

## Performance Characteristics

### Computational Complexity
- **Attention computation**: O(seq_len × num_heads × head_size)
- **Block lookup**: O(1) per block via block_tables indirection
- **Memory access**: Sequential within blocks (good cache locality)

### Memory Complexity
- **Per-sequence KV memory**: O((seq_len / block_size) × page_size_bytes)
- **Overhead**: O(max_seqs × max_blocks_per_seq) for block_tables metadata
- **Typical ratio**: ~40-60% of dense attention for typical block_size=16

### Cache Characteristics (with prefix caching)
- Best case: 100% cache hits (entire sequence in prefix)
- Typical case: 20-40% cache hits (system prompt + partial query reuse)
- Worst case: 0% hits (completely novel sequences)

---

## Summary

vLLM's KV cache and PagedAttention implementation provides:

1. **Flexible memory management**: Block-based allocation with fine-grained control
2. **Prefix caching**: Significant speedup through computed token reuse
3. **Efficient kernels**: Optimized CUDA/CPU implementations for attention computation
4. **Quantization support**: Reduced memory footprint with INT8 KV cache
5. **Production-ready**: Extensive metadata tracking, error handling, and logging

The system balances memory efficiency, computational speed, and ease of use through careful abstraction layers from block-level operations to high-level manager interfaces.
