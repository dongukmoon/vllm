# vLLM LLM Forward Pass Execution: PagedAttention + FlashAttention Flow

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    OpenAI-Compatible API Server                │
│                 (e.g., /v1/completions endpoint)               │
└─────────────────────────────┬──────────────────────────────────┘
                              │
┌─────────────────────────────v──────────────────────────────────┐
│                      AsyncLLMEngine                            │
│    (vllm/v1/engine/async_llm.py)                              │
│  - Manages requests & output streams                           │
│  - Runs background output handler                              │
└─────────────────────────────┬──────────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
    ┌────v─────┐                             ┌────v─────┐
    │ Processor │                             │  Engine  │
    │           │                             │   Core   │
    │ (Tokenize)│                             │          │
    └────┬─────┘                             └────┬─────┘
         │                                         │
         │  EngineCoreRequest                      │
         └─────────────────────┬────────────────────┘
                               │
          ┌────────────────────v──────────────────────┐
          │   Engine Core Event Loop (Continuous)    │
          │  vllm/v1/engine/core_client.py           │
          │                                          │
          │  while True:                             │
          │    1. scheduler.schedule()  ← Decides   │
          │    2. executor.execute()    ← Runs      │
          │    3. scheduler.update()    ← Processes │
          └────────────────────┬──────────────────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────────┐
        │         SCHEDULER LAYER (vllm/v1/core/)     │
        │  - Block allocation/deallocation (KV cache) │
        │  - Request scheduling (batching)            │
        │  - Returns: SchedulerOutput                 │
        │    ├── sequences to run                     │
        │    ├── block table (PagedAttention!)        │
        │    └── metadata                             │
        └─────────────────────┬───────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────────────┐
        │        EXECUTOR LAYER (vllm/v1/executor/)        │
        │  - Distributes work to GPU (single machine)      │
        │  - Calls: ModelRunner.execute_model()           │
        │  - Returns: List of token outputs               │
        └──────────────────────┬───────────────────────────┘
                               │
                               ▼
        ┌───────────────────────────────────────────────────┐
        │    MODEL RUNNER (vllm/v1/engine/model_runner.py) │
        │  - Prepares input tensors from scheduler output  │
        │  - Manages KV cache allocation                   │
        │  - Calls model forward pass                      │
        │  - Applies sampling                              │
        └─────────────────────┬─────────────────────────────┘
                              │
                              ▼
     ┌────────────────────────────────────────────────┐
     │    MODEL FORWARD PASS (LLaMA/Mistral/etc)    │
     │  vllm/model_executor/models/llama.py          │
     │                                                │
     │  for each layer:                              │
     │    1. LayerNorm(hidden_states)                │
     │    2. self_attn(query, key, value)            │
     │    3. mlp(hidden_states)                      │
     │    4. Residual connections                    │
     └────────────────┬───────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────v────────┐          ┌────v───────┐
    │Attention    │          │MLP Forward │
    │Operations   │          │(unchanged) │
    └────┬────────┘          └────────────┘
         │
         ▼
     ╔════════════════════════════════════════════════════════╗
     ║         ATTENTION FORWARD PASS (KEY PART!)             ║
     ║      vllm/attention/layer.py (wrapper)                ║
     ║      vllm/v1/attention/backends/flash_attn.py         ║
     ║                                                        ║
     ║  Q, K, V projection from hidden_states                ║
     ║         ↓                                              ║
     ║  Call: attention.forward(Q, K, V, attn_metadata)      ║
     ║         ↓                                              ║
     ║  PagedAttention + FlashAttention [SEE BELOW]          ║
     ║         ↓                                              ║
     ║  Output: Attention weights × Values                   ║
     ║         ↓                                              ║
     ║  Linear output projection                             ║
     ╚════════════════════════════════════════════════════════╝
         │
         └────────────────┬──────────────────┘
                          │
                    ┌─────v────────┐
                    │ Logits output│
                    └──────┬───────┘
                           │
                    ┌──────v──────────┐
                    │ Sampler/Token   │
                    │ Selection       │
                    └──────┬──────────┘
                           │
                           ▼
                    ┌──────────────────┐
                    │  Return outputs  │
                    │  (token_ids,     │
                    │   logprobs, etc) │
                    └──────────────────┘
```

---

## Step-by-Step Execution Flow

### Phase 1: Scheduling (KV Cache + Batch Preparation)

**File:** [vllm/v1/core/scheduler.py](vllm/v1/core/scheduler.py)

```python
# SCHEDULER DETERMINES:
# 1. Which sequences to run in this batch
# 2. How many NEW tokens each sequence needs
# 3. KV cache allocation (PagedAttention)

scheduler_output = scheduler.schedule()

# Returns:
# - scheduler_output.requests: List of RequestId to execute
# - scheduler_output.seq_data: {RequestId → (tokens, seq_len)}
# - Block allocation info passed to model runner
```

**KV Cache Manager Integration:**

```python
# Inside scheduler.schedule():

for request in pending_requests:
    # Step 1: Check for prefix cache hits
    computed_blocks, num_new_computed_tokens = kv_cache_manager.get_computed_blocks(
        request=request
    )
    # ↑ PagedAttention: Finds if this request shares prefix with cached blocks
    
    # Step 2: Allocate new blocks for new tokens
    new_blocks = kv_cache_manager.allocate_slots(
        request=request,
        num_new_tokens=num_new_tokens,  # Tokens generated in this step
        num_new_computed_tokens=num_new_computed_tokens,
        new_computed_blocks=computed_blocks,
    )
    # ↑ PagedAttention: Allocates blocks for new tokens
    
    # Step 3: Build block table for this request
    request.block_table = build_block_table(
        computed_blocks=computed_blocks,
        new_blocks=new_blocks
    )
    # ↑ PagedAttention: Block table maps logical→physical block IDs
    
    scheduled_requests.append(request)

# Result: Each request has:
# - request.block_table: [block_0_physical_id, block_1_physical_id, ...]
# - request.seq_lens: Current sequence length
# - request.num_computed_tokens: Number of cached tokens
```

**Block Table Example:**

```
Request A:
  Prompt: "Once upon a time"  (3 tokens)
  New Generation: "there was" (2 tokens)
  
  ├─ Block 0: tokens[0:16]  (cached from prefix) → Physical block 5
  ├─ Block 1: tokens[16:32] (new)               → Physical block 12
  └─ Block table: [5, 12, ...]

Request B:
  Prompt: "Once upon a time"  (3 tokens, SAME PREFIX!)
  New Generation: "a beautiful" (2 tokens)
  
  ├─ Block 0: tokens[0:16]   (REUSED from Request A!) → Physical block 5
  ├─ Block 1: tokens[16:32]  (new)                    → Physical block 8
  └─ Block table: [5, 8, ...]
  
Result: Two requests sharing the same physical block 5 (prefix caching!)
```

---

### Phase 2: Input Preparation

**File:** [vllm/v1/engine/model_runner.py](vllm/v1/engine/model_runner.py)

```python
def _prepare_inputs(scheduler_output: SchedulerOutput) -> PreparedInputs:
    """Convert scheduler output → model input tensors"""
    
    # 1. TOKENIZATION
    # Flatten all sequences into single batch
    all_token_ids = []
    for request in scheduler_output.requests:
        all_token_ids.extend(request.tokens[-batch_size:])  # Last N tokens
    
    input_ids_tensor = torch.tensor(all_token_ids)  # [num_tokens]
    
    # 2. POSITION EMBEDDING
    positions = []
    token_idx = 0
    for request in scheduler_output.requests:
        num_tokens = len(request.tokens[-batch_size:])
        # Position = index in sequence
        positions.extend(range(request.num_computed_tokens, 
                             request.num_computed_tokens + num_tokens))
    
    positions_tensor = torch.tensor(positions)  # [num_tokens]
    
    # 3. ATTENTION METADATA
    # Prepare for FlashAttention + PagedAttention
    attn_metadata = _build_attention_metadata(
        requests=scheduler_output.requests,
        block_tables=scheduler_output.block_tables,  # ← PagedAttention!
        seq_lens=scheduler_output.seq_lens,
    )
    
    return PreparedInputs(
        input_ids=input_ids_tensor,
        positions=positions_tensor,
        attn_metadata=attn_metadata,
        kv_cache=self.kv_cache,  # Shared GPU memory (all blocks)
    )
```

**Attention Metadata Building:**

```python
def _build_attention_metadata(requests, block_tables, seq_lens):
    """Build metadata for FlashAttention + PagedAttention"""
    
    metadata = FlashAttentionMetadata(
        num_actual_tokens=sum(len(r.tokens) for r in requests),
        query_start_loc=compute_query_start_locations(requests),
        seq_lens=torch.tensor(seq_lens),           # Per-request sequence lengths
        
        # ← PAGED ATTENTION PARAMETERS
        block_table=torch.tensor(block_tables),    # [num_reqs, max_blocks]
        # Physical block addresses for each logical block of each request
        
        # ← FLASH ATTENTION PARAMETERS
        max_seq_len=max(seq_lens),
        max_query_len=max(r.new_token_count for r in requests),
        slot_mapping=compute_slot_mapping(block_tables),
        # Maps token positions to KV cache slots (for cache updates)
        
        # Multi-head parameters
        use_cascade=False,  # Or True for large-batch cascade attention
    )
    
    return metadata
```

---

### Phase 3: Model Forward Pass

**File:** [vllm/model_executor/models/llama.py](vllm/model_executor/models/llama.py)

```python
class LlamaForCausalLM(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,           # [num_tokens]
        positions: torch.Tensor,           # [num_tokens] - position IDs
        kv_caches: list[torch.Tensor],     # List of KV cache for each layer
        attn_metadata: FlashAttentionMetadata,  # Metadata from Phase 2
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [num_tokens] - token IDs to process
            positions: [num_tokens] - position index for RoPE
            kv_caches: [num_layers, 2, num_blocks, block_size, num_kv_heads, head_size]
                      - All KV blocks in GPU memory (scattered)
            attn_metadata: Metadata with block_table, seq_lens, etc.
        """
        
        # Embedding lookup
        hidden_states = self.embed_tokens(input_ids)  # [num_tokens, hidden_size]
        
        # Forward through each transformer layer
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states, residual = decoder_layer.forward(
                positions=positions,
                hidden_states=hidden_states,
                residual=None,
                kv_cache=kv_caches[layer_idx],  # ← KV cache for this layer
                attn_metadata=attn_metadata,     # ← Block table, seq_lens
            )
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states  # [num_tokens, hidden_size]
```

**Decoder Layer Forward:**

```python
class LlamaDecoderLayer(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Attention Sub-layer
        # ─────────────────
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        
        # Self-Attention: THE KEY PART!
        attn_output = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        # ↑ See Phase 4 below for what happens here
        
        # MLP Sub-layer
        # ─────────────
        hidden_states, residual = self.post_attention_layernorm(
            attn_output, residual)
        mlp_output = self.mlp(hidden_states)
        
        return mlp_output, residual
```

---

### Phase 4: Attention Forward (PagedAttention + FlashAttention)

**File:** [vllm/model_executor/models/llama.py](vllm/model_executor/models/llama.py) + [vllm/v1/attention/backends/flash_attn.py](vllm/v1/attention/backends/flash_attn.py)

#### Step 4a: QKV Projection & Rotary Embedding

```python
class LlamaAttention(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,          # [num_tokens]
        hidden_states: torch.Tensor,      # [num_tokens, hidden_size]
    ) -> torch.Tensor:
        
        # Q, K, V projection
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        # q: [num_tokens, num_heads, head_size]
        # k: [num_tokens, num_kv_heads, head_size]
        # v: [num_tokens, num_kv_heads, head_size]
        
        # Rotary Position Embedding (RoPE)
        q, k = self.rotary_emb(positions, q, k)
        # ↑ Applies rotation based on token positions
        
        # Attention computation
        attn_output = self.attn(
            q=q,
            k=k,
            v=v,
            # And implicit KV cache, block_table, etc. from attn_metadata
        )
        
        # Output projection
        output, _ = self.o_proj(attn_output)
        return output
```

#### Step 4b: FlashAttention Backend Forward

```python
class FlashAttentionImpl(AttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,        # Q: [num_tokens, num_heads, head_size]
        key: torch.Tensor,          # K: [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,        # V: [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,     # [2, num_blocks, block_size, ...]
        attn_metadata: FlashAttentionMetadata,  # Block table, seq_lens, etc.
        output: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass combining PagedAttention + FlashAttention
        """
        
        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)
        # key_cache: [num_blocks, block_size, num_kv_heads, head_size]
        # value_cache: [num_blocks, block_size, num_kv_heads, head_size]
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Update KV Cache (PagedAttention indirection)
        # ═══════════════════════════════════════════════════════════
        if key is not None and value is not None:
            # Insert new keys/values into scattered blocks using slot_mapping
            reshape_and_cache_flash(
                key,
                value,
                key_cache,              # Scattered blocks in GPU memory
                value_cache,
                attn_metadata.slot_mapping,  # [num_tokens] → slot indices
                # ↑ PagedAttention: Maps token positions to KV cache slots
                # Handles scattered block placement automatically!
                self.kv_cache_dtype,
                k_scale=getattr(layer, '_k_scale', None),
                v_scale=getattr(layer, '_v_scale', None),
            )
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: Compute Attention with FlashAttention
        # ═══════════════════════════════════════════════════════════
        
        # Key decision: Use cascade attention for very large batches?
        if attn_metadata.use_cascade:
            # Cascade attention: Split into prefix + suffix
            attn_output = self._forward_with_cascade(
                query, key_cache, value_cache, attn_metadata
            )
        else:
            # Standard batch attention: All sequences together
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            
            # Call external FlashAttention library
            # This is where PagedAttention indirection is used!
            attn_output = flash_attn_varlen_func(
                q=query,                                # [total_q_tokens, num_heads, head_size]
                k=key_cache,                            # [num_blocks, block_size, num_kv_heads, head_size]
                v=value_cache,                          # [num_blocks, block_size, num_kv_heads, head_size]
                cu_seqlens_q=cu_seqlens_q,             # Cumulative query lengths
                seqused_k=seqused_k,                   # Actual KV sequence lengths (per request)
                max_seqlen_q=max_seqlen_q,             # Max query sequence length
                max_seqlen_k=attn_metadata.max_seq_len, # Max KV sequence length
                
                # ← PAGED ATTENTION PARAMETER
                block_table=attn_metadata.block_table,  # [num_seqs, max_blocks_per_seq]
                # Maps logical block indices → physical block IDs
                # FlashAttention kernel uses this for indirection!
                
                # Other parameters
                softmax_scale=self.scale,
                sliding_window=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                logits_soft_cap=self.logits_soft_cap,
                causal=True,  # or False for non-causal
            )
            # ↑ This is the KERNEL CALL!
            # - FlashAttention kernel receives block_table
            # - For each query token, kernel:
            #   1. Loads corresponding K, V blocks using block_table indirection
            #   2. Computes attention efficiently (I/O aware)
            #   3. Returns attention output
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: Optional quantization/scaling
        # ═══════════════════════════════════════════════════════════
        if output is None:
            output = attn_output
        else:
            output.copy_(attn_output)
        
        return output
```

---

## Detailed Kernel Execution: Inside `flash_attn_varlen_func`

### What FlashAttention Kernel Does (with PagedAttention):

```cuda-cpp
// Pseudo-code for what happens in the fused kernel

void flash_attention_kernel(
    Q: [num_query_tokens, num_heads, head_size],      // Query tensors (contiguous)
    K_cache: [num_blocks, block_size, num_kv_heads, head_size],  // All blocks (scattered)
    V_cache: [num_blocks, block_size, num_kv_heads, head_size],  // All blocks (scattered)
    block_table: [num_seqs, max_blocks_per_seq],       // ← Indirection table
    seq_lens: [num_seqs],                              // Sequence lengths
    ... other params ...
) {
    
    // For each query token
    for (int q_idx = 0; q_idx < num_query_tokens; q_idx++) {
        
        // Determine which sequence this token belongs to
        int seq_id = which_sequence(q_idx);
        int token_pos = position_in_sequence(q_idx, seq_id);
        
        // Determine which block this token can attend to
        int num_blocks_to_process = cdiv(seq_lens[seq_id], block_size);
        
        // Initialize attention accumulators (in fast SRAM)
        float accum[head_size] = {0};
        float log_sum_exp = 0;
        
        // ════════════════════════════════════════════════════════════
        // KEY PART: Loop over blocks using PAGED ATTENTION indirection
        // ════════════════════════════════════════════════════════════
        for (int block_idx = 0; block_idx < num_blocks_to_process; block_idx++) {
            
            // PAGED ATTENTION: Get physical block ID from block table
            int physical_block_id = block_table[seq_id][block_idx];
            // ↑ This is the indirection! Blocks are scattered in GPU memory
            
            // Load K, V from this physical block into SRAM
            // These loads might not be contiguous (scattered blocks)
            // but SRAM is still fast enough to handle this
            
            for (int token_in_block = 0; token_in_block < block_size; token_in_block++) {
                
                // Global token index in KV cache
                int k_idx = physical_block_id * block_size + token_in_block;
                
                // Load K vector from scattered block
                float k_vec[head_size] = load_from_cache(K_cache[physical_block_id][token_in_block]);
                
                // Compute attention score: Q^T @ K
                float score = dot_product(Q[q_idx], k_vec) * softmax_scale;
                
                // Track max score (for numerical stability)
                max_score = max(max_score, score);
            }
        }
        
        // Softmax computation (in SRAM)
        for (int block_idx = 0; block_idx < num_blocks_to_process; block_idx++) {
            int physical_block_id = block_table[seq_id][block_idx];
            
            for (int token_in_block = 0; token_in_block < block_size; token_in_block++) {
                // Normalized attention weights
                float attention_weight = exp(score - max_score) / partition_sum;
                
                // Load V from scattered block
                float v_vec[head_size] = load_from_cache(V_cache[physical_block_id][token_in_block]);
                
                // Accumulate: accum += attention_weight * V
                for (int d = 0; d < head_size; d++) {
                    accum[d] += attention_weight * v_vec[d];
                }
            }
        }
        
        // Write output to HBM
        O[q_idx] = accum;
    }
}
```

### Memory Access Pattern with PagedAttention:

```
Request A: block_table = [5, 12, ...]
Request B: block_table = [5, 8, ...]

GPU Memory Layout:
┌──────┬──────┬──────┬──────┬──────┬──────┐
│Blk 0 │Blk 1 │Blk 2 │ ... │Blk 5 │ ... │Blk 8 │ ... │Blk 12│
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
        ↑                    ↑             ↑                ↑
     Unused               Req A,B       Req B            Req A
                          (shared)

Kernel Execution:
- Process Req A token 0: block_table[A][0]=5 → load K,V from Block 5
- Process Req B token 0: block_table[B][0]=5 → load K,V from Block 5 (CACHE HIT!)
- Process Req A token 16: block_table[A][1]=12 → load K,V from Block 12
- Process Req B token 16: block_table[B][1]=8 → load K,V from Block 8

Result: Mixed random memory accesses, but GPU handles this well because:
1. Within same warp/block, some accesses are sequential
2. GPU cache can hide scattered access latency
3. SRAM tiling keeps hot data in fast memory
```

---

## Complete Data Flow Example

### Example Request Sequence

```
Request 1: "Once upon a time there was"
Request 2: "Once upon a"

Step 1: TOKENIZATION
┌────────────────────────────────────────┐
│Request 1: [t1, t2, t3, t4, t5, t6] (6 tokens)
│Request 2: [t1, t2, t3] (3 tokens, shared prefix!)
└────────────────────────────────────────┘

Step 2: SCHEDULER - KV Cache Allocation
┌─────────────────────────────────────────────────────┐
│Request 1:                                           │
│ - prefix_cached: tokens 0-2 (Block 5)              │
│ - new: tokens 3-6 need allocation                  │
│ - allocate: Block 12, 18 (2 new blocks)            │
│ - block_table: [5, 12, 18, ...]                    │
│                                                     │
│Request 2:                                           │
│ - prefix_cached: tokens 0-2 (Block 5, REUSED!)    │
│ - new: token 3 needs allocation                    │
│ - allocate: Block 8 (1 new block)                  │
│ - block_table: [5, 8, ...]                         │
└─────────────────────────────────────────────────────┘

GPU Memory State:
Block 5:  [t1_K, t1_V, t2_K, t2_V, t3_K, t3_V, ...] (SHARED!)
Block 8:  [t4_K, t4_V, ...] (Request 2, new)
Block 12: [t4_K, t4_V, ...] (Request 1, new)
Block 18: [t5_K, t5_V, t6_K, t6_V, ...] (Request 1, new)

Step 3: INPUT PREPARATION
┌─────────────────────────────────────┐
│Flattened input_ids:                │
│  [t1, t2, t3, t4, t5, t6, t4, t5]   │
│   └─Request 1────────┘  └Request 2─┘
│                                     │
│Attention Metadata:                  │
│ - seq_lens: [6, 3]                  │
│ - block_table:                      │
│    [5, 12, 18, ...]  ← Request 1   │
│    [5, 8, ...]       ← Request 2   │
│ - query_start_loc: [0, 6]           │
└─────────────────────────────────────┘

Step 4: MODEL FORWARD PASS
┌──────────────────────────────────────────┐
│For each transformer layer:              │
│                                          │
│1. Embed tokens → hidden_states           │
│   Shape: [8, hidden_size]               │
│                                          │
│2. Q, K, V projection                    │
│   Q: [8, num_heads, head_size]           │
│   K, V: [8, num_kv_heads, head_size]    │
│                                          │
│3. Attention (using FlashAttention!)     │
│   Kernel receives:                      │
│   - Q: [8, num_heads, head_size]        │
│   - K_cache, V_cache: [num_blocks, ...]│
│   - block_table: [2, max_blocks]        │
│                                          │
│4. Kernel processes:                     │
│   Token 0-5 (Request 1):               │
│     Uses block_table[0]: [5, 12, 18, ...]
│   Token 6-7 (Request 2):               │
│     Uses block_table[1]: [5, 8, ...]   │
│     ↑ Shares Block 5 with Request 1!   │
│                                          │
│5. Output: [8, hidden_size]              │
└──────────────────────────────────────────┘

Step 5: SAMPLING & OUTPUT
┌─────────────────────────────────────┐
│Request 1: sample next token from    │
│  logits[5] (last token of request)  │
│                                      │
│Request 2: sample next token from    │
│  logits[7] (last token of request)  │
└─────────────────────────────────────┘

Step 6: CACHE FINALIZATION
┌──────────────────────────────────┐
│Save KV cache blocks for next step│
│                                   │
│Request 1:                         │
│  Mark Block 12, 18 as computed   │
│  Store block hashes for          │
│  future prefix caching           │
│                                   │
│Request 2:                         │
│  Mark Block 8 as computed        │
│  Block 5 already computed        │
└──────────────────────────────────┘
```

---

## Key Insights

### How PagedAttention Enables High Efficiency:

1. **Flexible Batching**: Mix sequences of different lengths
   - Request 1: 6 tokens
   - Request 2: 3 tokens
   - No padding waste!

2. **Prefix Caching**: Requests with shared prefixes reuse blocks
   - Request 1 & 2 share Block 5
   - Computation saved: ~33% for Request 2

3. **Non-Contiguous Memory**: Block table indirection
   ```
   Logical view:  [Block 0][Block 1][Block 2]...
   Physical view: [scattered locations]
   Mapping:       block_table[] → physical addresses
   ```
   Result: Efficient memory utilization, no fragmentation

### How FlashAttention Speeds Up Computation:

1. **I/O Aware Algorithm**: Minimizes GPU memory bandwidth
   - Stores intermediate attention matrix in fast SRAM
   - Only reads/writes K, V, O from HBM

2. **Fused Kernel**: Q @ K @ V computation in single kernel
   - Avoids multiple kernel launches
   - Reduces kernel launch overhead

3. **Works with PagedAttention**: Block table indirection built-in
   - Kernel loads blocks via block_table[seq_id][block_idx]
   - Scattered blocks accessed efficiently

### Together: PagedAttention + FlashAttention

```
PagedAttention: "Where to store & find KV cache"
               (Memory Management)
                
FlashAttention: "How to compute attention fast"
               (Computation Optimization)
               
Combined:      Efficient multi-request batching + fast computation
              = High throughput LLM serving
```

---

## Performance Characteristics

### Example: Batch of 32 Requests

```
Without PagedAttention + FlashAttention:
├─ Sequence length variance: 100-4096 tokens
├─ Max padding waste: 90% (small seqs padded to largest)
├─ Memory usage: ~2x optimal
├─ Attention computation: Standard O(N²d)
└─ Throughput: ~500 tok/sec

With PagedAttention + FlashAttention:
├─ Zero padding waste (block-based allocation)
├─ Memory usage: Optimal
├─ Attention computation: O(N²d) but 2-4x faster
├─ Prefix caching bonus: 20-40% less recomputation
└─ Throughput: ~2000 tok/sec (4x improvement!)
```

---

## Summary

**vLLM's LLM Forward Pass Execution:**

1. **Request arrives** → Async engine queues it
2. **Scheduler decides** → Which requests to run, allocates KV blocks (PagedAttention)
3. **Prepare inputs** → Flatten batch, build block tables, create attention metadata
4. **Model forward** → Embedding + transformer layers
5. **Attention layer** → Q, K, V projections + rotary embeddings
6. **FlashAttention kernel** → 
   - Receives block_table for PagedAttention indirection
   - Loads K, V blocks using block_table (scattered blocks)
   - Computes attention I/O-efficiently
   - Returns attention output
7. **MLP layer** → Continue transformer processing
8. **Sampling** → Select next tokens
9. **Cache update** → Update KV blocks for next iteration

The magic: **PagedAttention** manages memory, **FlashAttention** computes fast, together they enable efficient LLM serving!
