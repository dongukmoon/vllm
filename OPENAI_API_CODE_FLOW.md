# vLLM OpenAI-Compatible API Code Flow Analysis (v0.1.0)

This document traces the complete code flow from the OpenAI-compatible API entry point down to the actual model inference execution in vLLM v0.1.0.

## Overview

The flow follows this path:
```
HTTP Request → FastAPI Endpoint → AsyncLLMEngine → LLMEngine → Scheduler → Worker → Model → Sampler → Response
```

---

## 1. Entry Point: OpenAI API Server

**File:** `vllm/entrypoints/openai/api_server.py`

### Main Entry Point
```python
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(...)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    
    # Create the AsyncLLMEngine from arguments
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Start uvicorn server
    uvicorn.run(app, host=args.host, port=args.port, ...)
```

**Key Components:**
- FastAPI app with CORS middleware
- Tokenizer for mapping token IDs to strings
- Global `engine` (AsyncLLMEngine) and `served_model` variables

### API Endpoint Handler
```python
@app.post("/v1/completions")
async def create_completion(raw_request: Request):
    # 1. Parse and validate request
    request = CompletionRequest(**await raw_request.json())
    
    # 2. Create SamplingParams from request
    sampling_params = SamplingParams(
        n=request.n,
        best_of=request.best_of,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        ignore_eos=request.ignore_eos,
        max_tokens=request.max_tokens,
        logprobs=request.logprobs,
        use_beam_search=request.use_beam_search,
    )
    
    # 3. Call engine.generate() - returns async generator
    result_generator = engine.generate(prompt, sampling_params, request_id)
    
    # 4. Stream or batch response
    if stream:
        return StreamingResponse(completion_stream_generator(), ...)
    else:
        # Iterate through generator until completion
        async for res in result_generator:
            final_res = res
        return CompletionResponse(...)
```

**Flow to Next Layer:** `engine.generate()` → AsyncLLMEngine

---

## 2. Async Engine Layer

**File:** `vllm/engine/async_llm_engine.py`

### AsyncLLMEngine Class
```python
class AsyncLLMEngine:
    def __init__(self, worker_use_ray, engine_use_ray, ...):
        # Create underlying LLMEngine (sync or Ray remote)
        if not self.engine_use_ray:
            engine_class = LLMEngine
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(LLMEngine).remote
        else:
            engine_class = ray.remote(num_gpus=1)(LLMEngine).remote
        self.engine = engine_class(*args, **kwargs)
        
        # Request tracking
        self.request_outputs: Dict[str, RequestOutput] = {}
        self.request_events: Dict[str, asyncio.Event] = {}
```

### Generate Method (Async Generator)
```python
async def generate(
    self, prompt, sampling_params, request_id, prompt_token_ids=None
) -> RequestOutput:
    # 1. Create event for this request
    request_event = asyncio.Event()
    self.request_events[request_id] = request_event
    
    # 2. Add request to engine's waiting queue
    if self.engine_use_ray:
        await self.engine.add_request.remote(
            request_id, prompt, sampling_params, ...)
    else:
        self.engine.add_request(
            request_id, prompt, sampling_params, ...)
    
    # 3. Continuously kick engine and yield outputs
    while True:
        # Kick engine if not running
        if not self.is_engine_running:
            await self.engine_step(request_id)
        
        # Wait for new output with timeout
        await asyncio.wait_for(request_event.wait(), timeout=1)
        request_event.clear()
        
        # Yield the output
        request_output = self.request_outputs[request_id]
        yield request_output
        
        # Break if finished
        if request_output.finished():
            del self.request_outputs[request_id]
            del self.request_events[request_id]
            break
```

### Engine Step
```python
async def engine_step(self, kicking_request_id=None):
    self.is_engine_running = True
    
    # Call LLMEngine.step() (sync or remote)
    if self.engine_use_ray:
        request_outputs = await self.engine.step.remote()
    else:
        await asyncio.sleep(0)  # Yield to event loop
        request_outputs = self.engine.step()
    
    self.is_engine_running = False
    
    # Notify waiting coroutines of new outputs
    for request_output in request_outputs:
        request_id = request_output.request_id
        self.request_outputs[request_id] = request_output
        self.request_events[request_id].set()
```

**Flow to Next Layer:** `engine.step()` → LLMEngine

---

## 3. LLM Engine Layer

**File:** `vllm/engine/llm_engine.py`

### LLMEngine Initialization
```python
class LLMEngine:
    def __init__(self, model_config, cache_config, parallel_config, 
                 scheduler_config, distributed_init_method, stage_devices, ...):
        # 1. Initialize tokenizer
        self.tokenizer = get_tokenizer(model_config.model)
        
        # 2. Create GPU workers (one per GPU)
        self.workers: List[Worker] = []
        for rank, node_resource, _ in stage_devices[0]:
            worker_cls = Worker
            if self.parallel_config.worker_use_ray:
                worker_cls = ray.remote(...)(worker_cls).remote
            
            worker = worker_cls(
                model_config, parallel_config, scheduler_config,
                rank, distributed_init_method)
            self.workers.append(worker)
        
        # 3. Profile memory and initialize KV cache
        self._init_cache()
        
        # 4. Create scheduler
        self.scheduler = Scheduler(scheduler_config, cache_config, log_stats)
```

### Add Request
```python
def add_request(self, request_id, prompt, sampling_params, 
                prompt_token_ids=None, arrival_time=None):
    # 1. Tokenize prompt if needed
    if prompt_token_ids is None:
        prompt_token_ids = self.tokenizer.encode(prompt)
    
    # 2. Create sequences (one per sampling_params.best_of)
    seqs: List[Sequence] = []
    for _ in range(sampling_params.best_of):
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)
        seqs.append(seq)
    
    # 3. Create sequence group
    seq_group = SequenceGroup(request_id, seqs, sampling_params, arrival_time)
    
    # 4. Add to scheduler's waiting queue
    self.scheduler.add_seq_group(seq_group)
```

### Step Method (Core Iteration Loop)
```python
def step(self) -> List[RequestOutput]:
    # 1. SCHEDULING: Decide what to execute
    seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
    
    if not seq_group_metadata_list and scheduler_outputs.is_empty():
        return []  # Nothing to do
    
    # 2. EXECUTION: Run model on workers
    output = self._run_workers(
        "execute_model",
        seq_group_metadata_list=seq_group_metadata_list,
        blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
        blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
        blocks_to_copy=scheduler_outputs.blocks_to_copy,
    )
    
    # 3. UPDATE: Update scheduler with model outputs
    seq_groups = self.scheduler.update(output)
    
    # 4. DECODE: Convert token IDs to text
    self._decode_sequences(seq_groups)
    
    # 5. STOP CHECK: Check stopping criteria
    self._stop_sequences(seq_groups)
    
    # 6. CLEANUP: Free finished sequences
    self.scheduler.free_finished_seq_groups()
    
    # 7. CREATE OUTPUT: Build RequestOutput objects
    request_outputs: List[RequestOutput] = []
    for seq_group in seq_groups:
        request_output = RequestOutput.from_seq_group(seq_group)
        request_outputs.append(request_output)
    return request_outputs
```

**Flow to Next Layer:** `scheduler.schedule()` → Scheduler, `_run_workers("execute_model")` → Worker

---

## 4. Scheduler Layer

**File:** `vllm/core/scheduler.py`

### Scheduler Class
```python
class Scheduler:
    def __init__(self, scheduler_config, cache_config, log_stats):
        # Policy for prioritizing sequences
        self.policy = PolicyFactory.get_policy(policy_name='fcfs')
        
        # Block manager for KV cache
        self.block_manager = BlockSpaceManager(
            block_size=cache_config.block_size,
            num_gpu_blocks=cache_config.num_gpu_blocks,
            num_cpu_blocks=cache_config.num_cpu_blocks,
        )
        
        # Three queues for sequence states
        self.waiting: List[SequenceGroup] = []   # New requests
        self.running: List[SequenceGroup] = []   # Currently running
        self.swapped: List[SequenceGroup] = []   # Swapped to CPU
```

### Schedule Method
```python
def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
    # 1. Call internal _schedule
    scheduler_outputs, prompt_group_ids = self._schedule()
    
    # 2. Create metadata for running sequences
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    for seq_group in self.running:
        is_prompt = seq_group.request_id in prompt_group_ids
        
        # Collect sequence data and block tables
        seq_data: Dict[int, SequenceData] = {}
        block_tables: Dict[int, List[int]] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq_id = seq.seq_id
            seq_data[seq_id] = seq.data
            block_tables[seq_id] = self.block_manager.get_block_table(seq)
        
        seq_group_metadata = SequenceGroupMetadata(
            request_id=seq_group.request_id,
            is_prompt=is_prompt,
            seq_data=seq_data,
            sampling_params=seq_group.sampling_params,
            block_tables=block_tables,
        )
        seq_group_metadata_list.append(seq_group_metadata)
    
    return seq_group_metadata_list, scheduler_outputs
```

### Internal Scheduling Logic
```python
def _schedule(self) -> Tuple[SchedulerOutputs, List[str]]:
    blocks_to_swap_in: Dict[int, int] = {}
    blocks_to_swap_out: Dict[int, int] = {}
    blocks_to_copy: Dict[int, List[int]] = {}
    
    # 1. RUNNING SEQUENCES: Reserve slots for running seqs
    self.running = self.policy.sort_by_priority(now, self.running)
    running: List[SequenceGroup] = []
    preempted: List[SequenceGroup] = []
    
    while self.running:
        seq_group = self.running.pop(0)
        # Try to append new slot
        while not self.block_manager.can_append_slot(seq_group):
            # Preempt lowest priority sequence if needed
            if self.running:
                victim = self.running.pop(-1)
                self._preempt(victim, blocks_to_swap_out)
                preempted.append(victim)
            else:
                self._preempt(seq_group, blocks_to_swap_out)
                preempted.append(seq_group)
                break
        else:
            self._append_slot(seq_group, blocks_to_copy)
            running.append(seq_group)
    self.running = running
    
    # 2. SWAPPED SEQUENCES: Swap back in if possible
    while self.swapped and not blocks_to_swap_out:
        seq_group = self.swapped[0]
        if not self.block_manager.can_swap_in(seq_group):
            break
        seq_group = self.swapped.pop(0)
        self._swap_in(seq_group, blocks_to_swap_in)
        self._append_slot(seq_group, blocks_to_copy)
        self.running.append(seq_group)
    
    # 3. WAITING SEQUENCES: Start new requests if possible
    prompt_group_ids: List[str] = []
    if not self.swapped:
        while self.waiting:
            seq_group = self.waiting[0]
            if not self.block_manager.can_allocate(seq_group):
                break
            # Check token budget
            num_prompt_tokens = seq_group.get_seqs()[0].get_len()
            if (num_batched_tokens + num_prompt_tokens 
                > self.scheduler_config.max_num_batched_tokens):
                break
            
            seq_group = self.waiting.pop(0)
            self._allocate(seq_group)
            self.running.append(seq_group)
            prompt_group_ids.append(seq_group.request_id)
    
    return SchedulerOutputs(...), prompt_group_ids
```

### Update Method
```python
def update(self, seq_outputs: Dict[int, SequenceOutputs]) -> List[SequenceGroup]:
    for seq_group in self.running:
        # Handle beam search (forking sequences)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            output = seq_outputs[seq.seq_id]
            if seq.seq_id != output.parent_seq_id:
                # Fork from parent
                self.block_manager.free(seq)
                parent_seq = seq_group.find(output.parent_seq_id)
                parent_seq.fork(seq)
                self.block_manager.fork(parent_seq, seq)
        
        # Append new tokens to sequences
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            output = seq_outputs[seq.seq_id]
            seq.append_token_id(output.output_token, output.logprobs)
    
    return self.running.copy()
```

**Flow to Next Layer:** Scheduler provides metadata to Worker for execution

---

## 5. Worker Layer

**File:** `vllm/worker/worker.py`

### Worker Initialization
```python
class Worker:
    def __init__(self, model_config, parallel_config, scheduler_config, 
                 rank, distributed_init_method):
        # 1. Initialize distributed environment
        _init_distributed_environment(parallel_config, rank, distributed_init_method)
        
        # 2. Initialize model
        set_random_seed(self.model_config.seed)
        self.model = get_model(model_config)  # Load LLM model
        
        # 3. Initialize all-reduce for tensor parallelism
        initialize_all_reduce_launcher(...)
        
        # Cache engine initialized later by init_cache_engine()
        self.cache_engine = None
        self.gpu_cache = None
```

### Execute Model Method
```python
@torch.inference_mode()
def execute_model(
    self,
    seq_group_metadata_list: List[SequenceGroupMetadata],
    blocks_to_swap_in: Dict[int, int],
    blocks_to_swap_out: Dict[int, int],
    blocks_to_copy: Dict[int, List[int]],
) -> Dict[int, SequenceOutputs]:
    # 1. CACHE OPERATIONS: Perform swap in/out/copy
    issued_cache_op = False
    if blocks_to_swap_in:
        self.cache_engine.swap_in(blocks_to_swap_in)
        issued_cache_op = True
    if blocks_to_swap_out:
        self.cache_engine.swap_out(blocks_to_swap_out)
        issued_cache_op = True
    if blocks_to_copy:
        self.cache_engine.copy(blocks_to_copy)
        issued_cache_op = True
    
    cache_events = self.cache_events if issued_cache_op else None
    
    # 2. PREPARE INPUTS: Convert metadata to tensors
    input_tokens, input_positions, input_metadata = self._prepare_inputs(
        seq_group_metadata_list)
    
    # 3. MODEL INFERENCE: Execute the model forward pass
    output = self.model(
        input_ids=input_tokens,
        positions=input_positions,
        kv_caches=self.gpu_cache,
        input_metadata=input_metadata,
        cache_events=cache_events,
    )
    
    return output
```

### Prepare Inputs Method
```python
def _prepare_inputs(
    self, seq_group_metadata_list
) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
    seq_groups: List[Tuple[List[int], SamplingParams]] = []
    input_tokens: List[int] = []
    input_positions: List[int] = []
    slot_mapping: List[int] = []
    
    # 1. PROMPTS: Add prompt tokens
    prompt_lens: List[int] = []
    for seq_group_metadata in seq_group_metadata_list:
        if not seq_group_metadata.is_prompt:
            continue
        
        seq_ids = list(seq_group_metadata.seq_data.keys())
        seq_groups.append((seq_ids, seq_group_metadata.sampling_params))
        
        seq_data = seq_group_metadata.seq_data[seq_ids[0]]
        prompt_tokens = seq_data.get_token_ids()
        prompt_len = len(prompt_tokens)
        prompt_lens.append(prompt_len)
        
        input_tokens.extend(prompt_tokens)
        input_positions.extend(range(prompt_len))
        
        # Compute slot mapping for KV cache
        block_table = seq_group_metadata.block_tables[seq_ids[0]]
        for i in range(prompt_len):
            block_number = block_table[i // self.block_size]
            block_offset = i % self.block_size
            slot = block_number * self.block_size + block_offset
            slot_mapping.append(slot)
    
    # 2. GENERATION TOKENS: Add generation tokens
    for seq_group_metadata in seq_group_metadata_list:
        if seq_group_metadata.is_prompt:
            continue
        
        seq_ids = list(seq_group_metadata.seq_data.keys())
        seq_groups.append((seq_ids, seq_group_metadata.sampling_params))
        
        for seq_id in seq_ids:
            seq_data = seq_group_metadata.seq_data[seq_id]
            generation_token = seq_data.get_last_token_id()
            input_tokens.append(generation_token)
            
            context_len = seq_data.get_len()
            position = context_len - 1
            input_positions.append(position)
            
            # Compute slot mapping
            block_table = seq_group_metadata.block_tables[seq_id]
            block_number = block_table[position // self.block_size]
            block_offset = position % self.block_size
            slot = block_number * self.block_size + block_offset
            slot_mapping.append(slot)
    
    # 3. Convert to tensors (pad to multiple of 8 for Tensor Cores)
    input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
    input_positions = _pad_to_alignment(input_positions, multiple_of=8)
    
    tokens_tensor = torch.cuda.LongTensor(input_tokens)
    positions_tensor = torch.cuda.LongTensor(input_positions)
    slot_mapping_tensor = torch.cuda.IntTensor(slot_mapping)
    # ... create other tensors
    
    input_metadata = InputMetadata(
        seq_groups=seq_groups,
        seq_data=seq_data,
        prompt_lens=prompt_lens,
        slot_mapping=slot_mapping_tensor,
        context_lens=context_lens_tensor,
        max_context_len=max_context_len,
        block_tables=block_tables_tensor,
    )
    
    return tokens_tensor, positions_tensor, input_metadata
```

**Flow to Next Layer:** `self.model()` → Model (e.g., LlamaForCausalLM)

---

## 6. Model Layer

**File:** `vllm/model_executor/models/llama.py` (example)

### Model Loader
**File:** `vllm/model_executor/model_loader.py`
```python
def get_model(model_config: ModelConfig) -> nn.Module:
    # 1. Get model class from architecture
    model_class = _get_model_architecture(model_config.hf_config)
    # e.g., LlamaForCausalLM, GPT2LMHeadModel, etc.
    
    # 2. Set default dtype
    torch.set_default_dtype(model_config.dtype)
    
    # 3. Create model instance
    model = model_class(model_config.hf_config)
    
    # 4. Load weights
    if model_config.use_dummy_weights:
        model = model.cuda()
        initialize_dummy_weights(model)
    else:
        model.load_weights(model_config.model, model_config.download_dir, ...)
        model = model.cuda()
    
    return model.eval()
```

### LlamaForCausalLM Forward Pass
```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)  # Transformer layers
        self.lm_head = ColumnParallelLinear(  # Output projection
            config.hidden_size, config.vocab_size, bias=False, ...)
        self.sampler = Sampler(config.vocab_size)  # Token sampler
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        # 1. TRANSFORMER: Run through model layers
        hidden_states = self.model(
            input_ids, positions, kv_caches, input_metadata, cache_events)
        
        # 2. SAMPLING: Sample next tokens from hidden states
        next_tokens = self.sampler(
            self.lm_head.weight, hidden_states, input_metadata)
        
        return next_tokens  # Dict[seq_id, SequenceOutputs]
```

### LlamaModel (Transformer Layers)
```python
class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(...)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self, input_ids, positions, kv_caches, input_metadata, cache_events
    ) -> torch.Tensor:
        # 1. EMBEDDING: Convert token IDs to embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # 2. LAYERS: Pass through all transformer layers
        for i in range(len(self.layers)):
            layer = self.layers[i]
            cache_event = None if cache_events is None else cache_events[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],  # KV cache for this layer
                input_metadata,
                cache_event,
            )
        
        # 3. NORM: Final layer normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
```

### LlamaDecoderLayer
```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(  # Multi-head attention
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
        )
        self.mlp = LlamaMLP(  # Feed-forward network
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
    
    def forward(self, positions, hidden_states, kv_cache, 
                input_metadata, cache_event):
        # 1. ATTENTION: Self-attention with KV cache
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states
        
        # 2. MLP: Feed-forward network
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
```

**Flow to Next Layer:** `self.sampler()` → Sampler

---

## 7. Sampler Layer

**File:** `vllm/model_executor/layers/sampler.py`

### Sampler Forward Pass
```python
class Sampler(nn.Module):
    def forward(
        self,
        embedding: torch.Tensor,  # lm_head weights
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> Dict[int, SequenceOutputs]:
        # 1. PRUNE: Get hidden states for sampling (last tokens only)
        hidden_states = _prune_hidden_states(hidden_states, input_metadata)
        
        # 2. LOGITS: Compute logits (hidden @ embedding^T)
        logits = torch.matmul(hidden_states, embedding.t())
        logits = gather_from_tensor_model_parallel_region(logits)  # All-gather
        logits = logits[:, :self.vocab_size]  # Remove padding
        
        # 3. PENALTIES: Apply presence and frequency penalties
        output_tokens = _get_output_tokens(input_metadata)
        presence_penalties, frequency_penalties = _get_penalties(input_metadata)
        logits = _apply_penalties(
            logits, output_tokens, presence_penalties, 
            frequency_penalties, self.vocab_size)
        
        # 4. TEMPERATURE: Apply temperature scaling
        temperatures = _get_temperatures(input_metadata)
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)
            logits.div_(t.unsqueeze(dim=1))
        
        # 5. PROBABILITIES: Convert to probabilities
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        logprobs = torch.log(probs)
        
        # 6. TOP-P/TOP-K: Apply truncation
        top_ps, top_ks = _get_top_p_top_k(input_metadata, self.vocab_size)
        if any(p < 1.0 for p in top_ps) or any(k != self.vocab_size for k in top_ks):
            probs = _apply_top_p_top_k(probs, top_ps, top_ks)
        
        # 7. SAMPLE: Sample next tokens
        return _sample(probs, logprobs, input_metadata)
```

### Sampling Function
```python
def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    input_metadata: InputMetadata,
) -> Dict[int, SequenceOutputs]:
    seq_outputs: Dict[int, SequenceOutputs] = {}
    
    idx = 0
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        
        if i < input_metadata.num_prompts:
            # PROMPT INPUT: Generate first tokens
            prob = probs[idx]
            logprob = logprobs[idx]
            idx += 1
            
            # Sample based on method
            if sampling_params.use_beam_search:
                # Beam search: Take top-k tokens
                beam_width = sampling_params.best_of
                _, next_token_ids = torch.topk(prob, beam_width)
                next_token_ids = next_token_ids.tolist()
            elif sampling_params.temperature == 0.0:
                # Greedy: Take argmax
                next_token_id = torch.argmax(prob)
                next_token_ids = [next_token_id.item()]
            else:
                # Random: Multinomial sampling
                next_token_ids = torch.multinomial(
                    prob, num_samples=sampling_params.best_of, replacement=True)
                next_token_ids = next_token_ids.tolist()
            
            # Get top-k logprobs
            next_logprobs = _get_topk_logprobs(logprob, sampling_params.logprobs)
            
            # Build outputs
            for seq_id, next_token_id in zip(seq_ids, next_token_ids):
                output_logprobs = next_logprobs.copy()
                output_logprobs[next_token_id] = logprob[next_token_id].item()
                seq_outputs[seq_id] = SequenceOutputs(
                    seq_id, seq_id, next_token_id, output_logprobs)
        else:
            # GENERATION TOKENS: Continue generation
            prob = probs[idx:idx + len(seq_ids)]
            logprob = logprobs[idx:idx + len(seq_ids)]
            idx += len(seq_ids)
            
            seq_logprobs = [
                input_metadata.seq_data[seq_id].cumulative_logprob
                for seq_id in seq_ids]
            
            # Sample for beam search or greedy/random
            if sampling_params.use_beam_search:
                # Add cumulative logprobs and select top beams
                seq_logprobs_tensor = torch.tensor(seq_logprobs, dtype=torch.float, 
                                                   device=logprob.device)
                logprob_with_cumulative = logprob + seq_logprobs_tensor.unsqueeze(dim=1)
                
                vocab_size = logprob.size(-1)
                beam_width = len(seq_ids)
                _, topk_ids = torch.topk(logprob_with_cumulative.flatten(), beam_width)
                
                seq_idx = [i // vocab_size for i in topk_ids.tolist()]
                parent_seq_ids = [seq_ids[i] for i in seq_idx]
                next_token_ids = [i % vocab_size for i in topk_ids.tolist()]
            elif sampling_params.temperature == 0.0:
                # Greedy
                next_token_id = torch.argmax(prob, dim=-1)
                next_token_ids = [int(next_token_id.item())]
                parent_seq_ids = seq_ids
            else:
                # Random sampling
                next_token_ids = torch.multinomial(prob, num_samples=1, replacement=True)
                next_token_ids = next_token_ids.squeeze(dim=-1).tolist()
                parent_seq_ids = seq_ids
            
            # Get top-k logprobs
            next_logprobs: Dict[int, Dict[int, float]] = {}
            for i, seq_id in enumerate(seq_ids):
                next_logprobs[seq_id] = _get_topk_logprobs(
                    logprob[i], sampling_params.logprobs)
            
            # Build outputs
            for seq_id, parent_seq_id, next_token_id in zip(
                seq_ids, parent_seq_ids, next_token_ids):
                i = seq_ids.index(parent_seq_id)
                output_logprobs = next_logprobs[parent_seq_id].copy()
                output_logprobs[next_token_id] = logprob[i, next_token_id].item()
                seq_outputs[seq_id] = SequenceOutputs(
                    seq_id, parent_seq_id, next_token_id, output_logprobs)
    
    return seq_outputs
```

**Return Flow:** SequenceOutputs → Worker → Scheduler → LLMEngine → AsyncLLMEngine → FastAPI

---

## 8. Complete Flow Summary

### Request Processing Flow

1. **HTTP Request arrives** at FastAPI endpoint `/v1/completions`
2. **API Server** parses request, creates `SamplingParams`, calls `engine.generate()`
3. **AsyncLLMEngine** adds request to queue, enters async loop:
   - Calls `engine_step()` to process requests
   - Waits for outputs and yields them
4. **LLMEngine.step()** orchestrates one iteration:
   - **Scheduler** decides what to execute (schedule, swap, allocate)
   - **Workers** execute model on selected sequences
   - **Scheduler** updates state with outputs
   - Decodes tokens to text, checks stopping criteria
   - Returns `RequestOutput` objects
5. **Worker.execute_model()** performs actual inference:
   - Manages KV cache operations (swap, copy)
   - Prepares input tensors from metadata
   - Calls model forward pass
6. **Model forward pass** (e.g., LlamaForCausalLM):
   - Embeds input tokens
   - Passes through transformer layers (attention + MLP)
   - Applies final normalization
7. **Sampler** generates next tokens:
   - Computes logits from hidden states
   - Applies penalties, temperature, top-p/top-k
   - Samples tokens (greedy, random, or beam search)
   - Returns `SequenceOutputs` with token IDs and logprobs
8. **Response flows back**:
   - SequenceOutputs → Worker → Scheduler → LLMEngine
   - LLMEngine decodes tokens and creates RequestOutput
   - AsyncLLMEngine yields RequestOutput to API handler
   - API handler streams or batches response to client

### Key Data Structures

- **CompletionRequest**: API request from client
- **SamplingParams**: Sampling configuration (temperature, top-p, etc.)
- **Sequence**: Single generation sequence with tokens and state
- **SequenceGroup**: Group of sequences from one request
- **SequenceGroupMetadata**: Metadata for execution (seq_data, block_tables)
- **InputMetadata**: Batched metadata for model forward pass
- **SchedulerOutputs**: Cache operations to perform
- **SequenceOutputs**: Sampled token ID and logprobs per sequence
- **RequestOutput**: Output for one request (multiple sequences)

### Performance Optimizations

1. **Continuous Batching**: Scheduler dynamically batches requests
2. **PagedAttention**: KV cache stored in paged blocks for efficient memory
3. **Tensor Parallelism**: Model sharded across GPUs
4. **Iteration-level Scheduling**: Fine-grained scheduling per iteration
5. **Async Processing**: Non-blocking request handling
6. **Memory Swapping**: Swap KV cache between GPU and CPU
7. **Preemption**: Pause low-priority requests when memory is tight

---

## Key Files Reference

| Layer | File Path | Description |
|-------|-----------|-------------|
| API | `vllm/entrypoints/openai/api_server.py` | FastAPI server and endpoints |
| API | `vllm/entrypoints/openai/protocol.py` | Request/response data models |
| Async Engine | `vllm/engine/async_llm_engine.py` | Async wrapper for LLMEngine |
| Engine | `vllm/engine/llm_engine.py` | Core engine orchestrating inference |
| Scheduler | `vllm/core/scheduler.py` | Request scheduling and memory management |
| Worker | `vllm/worker/worker.py` | GPU worker executing model |
| Model Loader | `vllm/model_executor/model_loader.py` | Model instantiation and loading |
| Models | `vllm/model_executor/models/*.py` | Model architectures (Llama, GPT2, etc.) |
| Sampler | `vllm/model_executor/layers/sampler.py` | Token sampling logic |
| Attention | `vllm/model_executor/layers/attention.py` | PagedAttention implementation |

---

## Conclusion

The vLLM architecture is designed for high-throughput LLM serving with:
- **Modular design**: Clear separation of API, engine, scheduler, and execution
- **Efficient memory management**: PagedAttention and dynamic block allocation
- **Flexible scheduling**: Support for streaming, batching, preemption, and swapping
- **Distributed execution**: Ray-based multi-GPU support with tensor parallelism
- **OpenAI compatibility**: Drop-in replacement for OpenAI API

This flow analysis provides a complete understanding of how a request flows through vLLM from API entry to model inference and back.
