# vLLM v0.6.0: OpenAI-Compatible API to Model Execution Code Flow

This document traces the complete code flow from OpenAI-compatible API calls down to actual model inference execution in vLLM v0.6.0.

---

## Architecture Overview

vLLM v0.6.0 uses a **two-tier architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI HTTP Server                        │
│           (vllm/entrypoints/openai/api_server.py)          │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
    In-Process            Multiprocess RPC
  AsyncLLMEngine         (Engine in separate process)
         │                        │
    ┌────▼─────────────────────┐ │
    │  AsyncLLMEngine          │ │
    │  (vllm/engine/           │ │
    │   async_llm_engine.py)   │ │
    └────┬────────────────────┘ │
         │                      │
    ┌────▼──────────────────────┴─┐
    │      LLMEngine              │
    │  (vllm/engine/llm_engine.py)│
    └────┬───────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │  Scheduler                         │
    │  (vllm/core/scheduler.py)          │
    └────┬──────────────────────────────┘
         │
    ┌────▼────────────────────────────────┐
    │  Executor                            │
    │  (vllm/executor/*)                   │
    └────┬─────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────┐
    │  Worker (GPU)                        │
    │  (vllm/worker/worker.py)             │
    └────┬──────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  Model Runner / Sampler                 │
    │  (vllm/worker/model_runner.py)          │
    │  (vllm/model_executor/layers/sampler.py)│
    └──────────────────────────────────────────┘
```

---

## 1. Entry Point: OpenAI-Compatible API Server

**File:** `vllm/entrypoints/openai/api_server.py`

### Server Setup
```python
# Line ~48-57: CLI argument parsing
parser = make_arg_parser()
args = parser.parse_args()

# Line ~66-114: Build engine client
async with build_async_engine_client(args) as async_engine_client:
    # Line ~118-124: Create FastAPI app with endpoints
    app = FastAPI(lifespan=lifespan)
    
    # Line ~131-141: Add CORS middleware
    app.add_middleware(CORSMiddleware, ...)
    
    # Line ~181-190: Mount OpenAI API router
    app.include_router(router)
    
    # Line ~191-196: Start uvicorn server
    await serve_http(
        app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ...
    )
```

### Engine Client Creation
```python
# Line ~115-166: build_async_engine_client_from_engine_args()
# Creates either in-process or RPC-based async engine

if disable_frontend_multiprocessing or is_embedding_model:
    # In-process: AsyncLLMEngine directly
    engine_client = AsyncLLMEngine.from_engine_args(
        engine_args, 
        usage_context=UsageContext.OPENAI_API_SERVER
    )
else:
    # Multiprocess RPC: engine runs in separate process
    rpc_client = AsyncEngineRPCClient(rpc_path)
    rpc_server_process = multiprocessing.Process(
        target=run_rpc_server,
        args=(engine_args, UsageContext.OPENAI_API_SERVER, rpc_path)
    )
    rpc_server_process.start()
```

### HTTP Endpoints
```python
# Line ~211-302: POST /v1/chat/completions
@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = openai_serving_chat  # OpenAIServingChat instance
    try:
        # Call chat handler (see Section 2)
        generator = await handler.create_chat_completion(request, raw_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.dict(), status_code=400)
    
    # Streaming response
    if request.stream:
        return StreamingResponse(generator, media_type="text/event-stream")
    # Non-streaming response
    else:
        return ChatCompletionResponse(...)

# Line ~305-380: POST /v1/completions
@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    handler = openai_serving_completion  # OpenAIServingCompletion instance
    try:
        generator = await handler.create_completion(request, raw_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.dict(), status_code=400)
    
    if request.stream:
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        return CompletionResponse(...)
```

---

## 2. Serving Layer: Chat and Completion Handlers

### OpenAIServingChat
**File:** `vllm/entrypoints/openai/serving_chat.py`

```python
# Line ~84-185: create_chat_completion()
async def create_chat_completion(
    self,
    request: ChatCompletionRequest,
    raw_request: Optional[Request] = None,
) -> Union[AsyncGenerator[str, None], ChatCompletionResponse, ErrorResponse]:
    
    # 1. Validate model
    error_check_ret = await self._check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    
    # 2. Get LoRA and prompt adapter if needed
    (lora_request, prompt_adapter_request) = self._maybe_get_adapters(request)
    
    # 3. Get tokenizer
    tokenizer = await self.async_engine_client.get_tokenizer(lora_request)
    
    # 4. Parse chat messages
    conversation, mm_data_future = parse_chat_messages_futures(
        request.messages, 
        model_config, 
        tokenizer
    )
    
    # 5. Apply chat template
    prompt = apply_chat_template(
        tokenizer,
        conversation=conversation,
        chat_template=request.chat_template or self.chat_template,
        add_generation_prompt=request.add_generation_prompt,
        tools=tool_dicts,
        documents=request.documents,
        **(request.chat_template_kwargs or {}),
    )
    
    # 6. Wait for multimodal data
    mm_data = await mm_data_future
    
    # 7. Create sampling parameters
    sampling_params = SamplingParams(
        n=request.n,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        max_tokens=request.max_tokens,
        logprobs=request.logprobs,
        ...
    )
    
    # 8. Generate unique request ID
    request_id = f"chatcmpl-{random_uuid()}"
    
    # 9. Call engine.generate() - Main inference call!
    result_generator = self.async_engine_client.generate(
        prompt,
        sampling_params,
        request_id,
        lora_request=lora_request,
        prompt_adapter_request=prompt_adapter_request,
    )
    
    # 10. Format and return responses
    if request.stream:
        # Streaming: yield chunks as they arrive
        return self._chat_completion_stream_generator(
            request, result_generator, ...
        )
    else:
        # Non-streaming: collect all outputs
        return await self._chat_completion_full_generator(
            request, result_generator, ...
        )
```

### OpenAIServingCompletion
**File:** `vllm/entrypoints/openai/serving_completion.py`

Similar structure:
```python
# Tokenize prompt
prompt_input = self._tokenize_prompt_input(
    request,
    request.prompt,
    truncate_prompt_tokens=request.truncate_prompt_tokens,
)

# Create sampling parameters
sampling_params = self._create_sampling_params(request)

# Generate request ID
request_id = f"cmpl-{random_uuid()}"

# Call engine
result_generator = self.async_engine_client.generate(
    prompt_input,
    sampling_params,
    request_id,
    lora_request=lora_request,
)

# Format responses
if request.stream:
    return self._completion_stream_generator(...)
else:
    return await self._completion_full_generator(...)
```

---

## 3. Async Engine Client

**File:** `vllm/engine/async_llm_engine.py` (lines 1-1284)

### AsyncLLMEngine Class
```python
# Line ~166-200: from_engine_args() - Factory method
@classmethod
async def from_engine_args(
    cls,
    engine_args: AsyncEngineArgs,
    log_requests: bool = True,
    log_stats: bool = True,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
) -> "AsyncLLMEngine":
    # Create engine configs
    engine_configs = engine_args.create_engine_config()
    
    # Initialize parallel environment if needed
    distributed_init_method, devices = initialize_ray_cluster(
        parallel_config,
        engine_args.engine_use_ray,
    )
    
    # Create AsyncLLMEngine instance
    return cls(
        *engine_configs,
        distributed_init_method,
        devices,
        log_requests=log_requests,
        log_stats=log_stats,
        usage_context=usage_context,
    )
```

### AsyncLLMEngine __init__
```python
# Line ~201-300: __init__()
def __init__(
    self,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    parallel_config: ParallelConfig,
    scheduler_config: SchedulerConfig,
    device_config: DeviceConfig,
    load_config: LoadConfig,
    lora_config: Optional[LoRAConfig],
    speculative_config: Optional[SpeculativeConfig],
    prompt_adapter_config: Optional[PromptAdapterConfig],
    distributed_init_method: str,
    devices: List[Device],
    log_requests: bool = True,
    log_stats: bool = True,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
):
    # Create synchronous LLMEngine (see Section 4)
    self.engine = LLMEngine(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        lora_config=lora_config,
        speculative_config=speculative_config,
        prompt_adapter_config=prompt_adapter_config,
        distributed_init_method=distributed_init_method,
        devices=devices,
        log_requests=log_requests,
        log_stats=log_stats,
        usage_context=usage_context,
    )
    
    # Create background execution task
    self._background_loop: Optional[asyncio.Task] = None
```

### AsyncLLMEngine.generate() - Async Generator
```python
# Line ~350-425: generate()
async def generate(
    self,
    prompt: Optional[PromptType],
    sampling_params: SamplingParams,
    request_id: str,
    prompt_token_ids: Optional[List[int]] = None,
    lora_request: Optional[LoRARequest] = None,
    ...
) -> AsyncGenerator[RequestOutput, None]:
    # 1. Add request to engine's waiting queue
    self.engine.add_request(
        request_id,
        prompt,
        sampling_params,
        prompt_token_ids=prompt_token_ids,
        lora_request=lora_request,
        ...
    )
    
    # 2. Get async stream for this request
    stream = self.request_streams[request_id]
    
    # 3. Yield outputs as they become available
    async for output in stream:
        if isinstance(output, Exception):
            raise output
        yield output
```

### Background Engine Loop
```python
# Line ~472-530: _run_engine_loop()
async def _run_engine_loop(self) -> None:
    while True:
        # 1. Get one iteration of results from engine
        outputs = self.engine.step()
        
        # 2. Process and distribute outputs to streams
        for output in outputs:
            request_id = output.request_id
            if request_id not in self.request_streams:
                continue
            
            stream = self.request_streams[request_id]
            stream.put(output)
            
            # If finished, close stream
            if output.finished:
                stream.finish()
                del self.request_streams[request_id]
        
        # 3. Small sleep to avoid busy-waiting
        await asyncio.sleep(0)
```

---

## 4. Synchronous LLMEngine

**File:** `vllm/engine/llm_engine.py`

### LLMEngine.__init__
```python
# Line ~170-350: __init__()
def __init__(
    self,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    parallel_config: ParallelConfig,
    scheduler_config: SchedulerConfig,
    device_config: DeviceConfig,
    load_config: LoadConfig,
    lora_config: Optional[LoRAConfig],
    speculative_config: Optional[SpeculativeConfig],
    prompt_adapter_config: Optional[PromptAdapterConfig],
    distributed_init_method: str,
    devices: List[Device],
    ...
):
    # 1. Store configs
    self.model_config = model_config
    self.cache_config = cache_config
    self.parallel_config = parallel_config
    self.scheduler_config = scheduler_config
    self.device_config = device_config
    self.load_config = load_config
    self.lora_config = lora_config
    self.speculative_config = speculative_config
    self.prompt_adapter_config = prompt_adapter_config
    
    # 2. Initialize tokenizer
    self.tokenizer = init_tokenizer_from_configs(
        model_config, load_config
    )
    
    # 3. Create scheduler (see Section 5)
    self.scheduler = Scheduler(
        scheduler_config,
        cache_config,
        lora_config,
    )
    
    # 4. Create executor (GPU workers)
    self.executor = get_executor_cls(parallel_config)(
        model_config,
        cache_config,
        parallel_config,
        scheduler_config,
        device_config,
        load_config,
        lora_config,
        speculative_config,
        prompt_adapter_config,
        distributed_init_method,
        devices,
    )
    
    # 5. Initialize KV cache
    self._init_cache()
```

### add_request()
```python
# Line ~590-620: add_request()
def add_request(
    self,
    request_id: str,
    prompt: Optional[PromptType],
    sampling_params: SamplingParams,
    prompt_token_ids: Optional[List[int]] = None,
    lora_request: Optional[LoRARequest] = None,
    ...
) -> None:
    # 1. Compute prompt tokens if needed
    if prompt_token_ids is None:
        prompt_token_ids = self.tokenizer.encode(prompt)
    
    # 2. Create sequence objects
    seqs: List[Sequence] = []
    for _ in range(sampling_params.best_of):
        seq = Sequence(
            request_id,
            prompt,
            prompt_token_ids,
            self.model_config.max_model_len,
        )
        seqs.append(seq)
    
    # 3. Create sequence group
    seq_group = SequenceGroup(
        request_id,
        seqs,
        sampling_params,
        lora_request,
        ...
    )
    
    # 4. Add to scheduler's waiting queue
    self.scheduler.add_seq_group(seq_group)
```

### step() - Main Iteration Loop
```python
# Line ~686-750: step()
def step(self) -> List[RequestOutput]:
    # 1. SCHEDULING: Decide what to execute
    seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
    
    if scheduler_outputs.is_empty():
        return []
    
    # 2. EXECUTE: Run model on scheduled requests
    # This is the key inference call! (see Section 6)
    output = self.executor.execute_model(
        seq_group_metadata_list=seq_group_metadata_list,
        blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
        blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
        blocks_to_copy=scheduler_outputs.blocks_to_copy,
    )
    
    # 3. UPDATE: Update scheduler with model outputs
    seq_groups = self.scheduler.update(output)
    
    # 4. DECODE: Convert tokens to text (post-processing)
    self._decode_sequences(seq_groups)
    
    # 5. STOP CHECK: Check stopping conditions
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

---

## 5. Scheduler

**File:** `vllm/core/scheduler.py`

### Scheduler.__init__
```python
# Line ~343-410: __init__()
def __init__(
    self,
    scheduler_config: SchedulerConfig,
    cache_config: CacheConfig,
    lora_config: Optional[LoRAConfig],
    ...
):
    self.scheduler_config = scheduler_config
    self.cache_config = cache_config
    self.lora_config = lora_config
    
    # Request queues
    self.waiting: Deque[SequenceGroup] = deque()  # New requests
    self.running: List[SequenceGroup] = []         # Currently running
    self.swapped: List[SequenceGroup] = deque()    # Swapped to CPU
    
    # KV cache manager
    self.block_manager: BlockSpaceManager = get_block_space_manager(
        cache_config, is_cache_manager_v2=False
    )
```

### schedule()
```python
# Line ~560-650: schedule()
def schedule(
    self,
    ...
) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
    
    # 1. Call internal scheduling logic
    scheduler_outputs = self._schedule()
    
    # 2. Create metadata for running sequences
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    for seq_group in self.running:
        # Get block table for each sequence
        block_table = self.block_manager.get_block_table(seq_group)
        
        # Create metadata
        seq_group_metadata = SequenceGroupMetadata(
            request_id=seq_group.request_id,
            is_prompt=seq_group.is_prompt,
            seq_data={seq.seq_id: seq.data for seq in seq_group.get_seqs()},
            sampling_params=seq_group.sampling_params,
            block_tables={seq.seq_id: block_table for seq in seq_group.get_seqs()},
            lora_request=seq_group.lora_request,
            ...
        )
        seq_group_metadata_list.append(seq_group_metadata)
    
    return seq_group_metadata_list, scheduler_outputs
```

### _schedule() - Scheduling Logic
```python
# Line ~750-900: _schedule()
def _schedule(self) -> SchedulerOutputs:
    
    # 1. MOVE WAITING → RUNNING if space available
    while self.waiting:
        seq_group = self.waiting[0]
        
        # Check if we can allocate KV cache blocks
        if not self.block_manager.can_allocate(seq_group):
            break
        
        # Allocate blocks
        self.block_manager.allocate(seq_group)
        
        # Move to running
        self.waiting.popleft()
        self.running.append(seq_group)
    
    # 2. PREEMPTION if needed (when running out of resources)
    while not enough_space_for_running:
        # Preempt lowest priority sequence group
        victim = self._preempt_by_priority()
        
        if victim.is_single_seq:
            # Recomputation: restart from beginning
            victim.status = SequenceStatus.WAITING
            self.waiting.appendleft(victim)
        else:
            # Swapping: save KV cache to CPU
            self._swap_out(victim)
            self.swapped.append(victim)
    
    # 3. SWAP IN if possible
    while self.swapped and space_available:
        seq_group = self.swapped.popleft()
        self._swap_in(seq_group)
        self.running.append(seq_group)
    
    # 4. Prepare cache operations (swap in/out/copy)
    blocks_to_swap_in = [...]
    blocks_to_swap_out = [...]
    blocks_to_copy = [...]
    
    return SchedulerOutputs(
        scheduled_seq_groups=self.running,
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_swap_out=blocks_to_swap_out,
        blocks_to_copy=blocks_to_copy,
        num_prefill_groups=...,
        num_batched_tokens=...,
        ...
    )
```

---

## 6. Executor (GPU Workers)

**File:** `vllm/executor/` (e.g., `gpu_executor.py`)

### GPUExecutor.execute_model()
```python
def execute_model(
    self,
    seq_group_metadata_list: List[SequenceGroupMetadata],
    blocks_to_swap_in: List[Tuple[int, int]],
    blocks_to_swap_out: List[Tuple[int, int]],
    blocks_to_copy: List[Tuple[int, int]],
) -> SamplerOutput:
    
    # 1. Send task to all GPU workers
    output = self._run_workers(
        "execute_model",
        seq_group_metadata_list=seq_group_metadata_list,
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_swap_out=blocks_to_swap_out,
        blocks_to_copy=blocks_to_copy,
    )
    
    return output
```

---

## 7. Worker (GPU)

**File:** `vllm/worker/worker.py`

### Worker.execute_model()
```python
# Line ~250-350: execute_model()
def execute_model(
    self,
    seq_group_metadata_list: List[SequenceGroupMetadata],
    blocks_to_swap_in: List[Tuple[int, int]],
    blocks_to_swap_out: List[Tuple[int, int]],
    blocks_to_copy: List[Tuple[int, int]],
) -> SamplerOutput:
    
    # 1. Perform cache operations (CPU ↔ GPU transfers)
    if blocks_to_swap_in:
        self.cache_engine[0].swap_in(blocks_to_swap_in)
    if blocks_to_swap_out:
        self.cache_engine[0].swap_out(blocks_to_swap_out)
    if blocks_to_copy:
        self.cache_engine[0].copy(blocks_to_copy)
    
    # 2. Delegate to model runner (see Section 8)
    return self.model_runner.execute_model(seq_group_metadata_list)
```

---

## 8. Model Runner

**File:** `vllm/worker/model_runner.py`

### ModelRunner.execute_model()
```python
@torch.inference_mode()
def execute_model(
    self,
    seq_group_metadata_list: List[SequenceGroupMetadata],
) -> SamplerOutput:
    
    # 1. PREPARE INPUTS: Convert metadata to tensors
    input_tokens, input_positions, attn_metadata = \
        self._prepare_inputs(seq_group_metadata_list)
    
    # 2. FORWARD PASS: Execute model
    hidden_states = self.model(
        input_ids=input_tokens,
        positions=input_positions,
        kv_caches=self.kv_cache,
        attention_metadata=attn_metadata,
    )
    
    # 3. GET LOGITS & SAMPLE: Compute next tokens
    sampler_output = self.sampler(
        logits=hidden_states @ self.lm_head.weight,
        sampling_metadata=sampling_metadata,
    )
    
    return sampler_output
```

---

## 9. Model Layer

**File:** `vllm/model_executor/models/` (e.g., `llama.py`)

### LlamaForCausalLM.forward()
```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    attention_metadata: AttentionMetadata,
) -> torch.Tensor:
    
    # 1. EMBEDDING: Token IDs → embeddings
    hidden_states = self.embed_tokens(input_ids)
    
    # 2. TRANSFORMER LAYERS: Apply attention + MLP
    for i, layer in enumerate(self.layers):
        hidden_states = layer(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_caches[i],
            attention_metadata=attention_metadata,
        )
    
    # 3. NORMALIZATION: Final layer norm
    hidden_states = self.norm(hidden_states)
    
    return hidden_states
```

### LlamaDecoderLayer.forward()
```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor],
    attention_metadata: AttentionMetadata,
) -> torch.Tensor:
    
    # 1. SELF-ATTENTION with residual
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    hidden_states = residual + hidden_states
    
    # 2. FEED-FORWARD with residual
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    
    return hidden_states
```

---

## 10. Sampler (Token Sampling)

**File:** `vllm/model_executor/layers/sampler.py`

### Sampler.forward()
```python
def forward(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> SamplerOutput:
    
    # 1. Apply sampling parameters to logits
    probs = self._apply_sampling_params(
        logits,
        sampling_metadata,
    )
    
    # 2. Sample next token
    next_token_ids = torch.multinomial(
        probs,
        num_samples=1,
    )
    
    # 3. Get logprobs if requested
    logprobs = torch.log(probs) if sampling_metadata.get_logprobs else None
    
    return SamplerOutput(
        next_token_ids=next_token_ids,
        logprobs=logprobs,
    )
```

### _apply_sampling_params()
```python
def _apply_sampling_params(self, logits, sampling_metadata):
    # 1. TEMPERATURE: Scale logits
    if temperature != 1.0:
        logits = logits / temperature
    
    # 2. TOP-P: Nucleus sampling
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1))
        mask = cumsum_probs > top_p
        logits[sorted_indices[mask]] = -float('inf')
    
    # 3. TOP-K: Keep only top-k tokens
    if top_k > 0:
        topk_logits, topk_indices = torch.topk(logits, top_k)
        logits[~topk_indices] = -float('inf')
    
    # 4. SOFTMAX: Convert to probabilities
    probs = torch.softmax(logits, dim=-1)
    
    return probs
```

---

## Complete Request Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│ 1. HTTP POST /v1/chat/completions                   │
│    OpenAIServingChat.create_chat_completion()       │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────▼──────────┐
         │ 2. Parse & Validate   │
         │    - Chat template    │
         │    - Sampling params  │
         └───────────┬──────────┘
                     │
     ┌───────────────▼──────────────────┐
     │ 3. AsyncLLMEngine.generate()     │
     │    - Add request to queue        │
     │    - Yield outputs from stream   │
     └───────────┬──────────────────────┘
                 │
     ┌───────────▼──────────────────────┐
     │ 4. Background Loop                │
     │    engine_step() continuously     │
     └───────────┬──────────────────────┘
                 │
     ┌───────────▼────────────────────────────┐
     │ 5. LLMEngine.step()                     │
     │    ├─ Scheduler.schedule()              │
     │    ├─ Executor.execute_model()          │
     │    ├─ Scheduler.update()                │
     │    └─ Return RequestOutput              │
     └───────────┬────────────────────────────┘
                 │
     ┌───────────▼────────────────────────────┐
     │ 6. Scheduler.schedule()                 │
     │    ├─ Move WAITING → RUNNING            │
     │    ├─ Allocate KV cache                 │
     │    ├─ Return SequenceGroupMetadata      │
     │    └─ Return SchedulerOutputs           │
     └───────────┬────────────────────────────┘
                 │
     ┌───────────▼────────────────────────────┐
     │ 7. Executor.execute_model()             │
     │    └─ Delegate to GPU workers           │
     └───────────┬────────────────────────────┘
                 │
     ┌───────────▼────────────────────────────┐
     │ 8. Worker.execute_model()               │
     │    ├─ Swap KV cache blocks              │
     │    └─ Call ModelRunner.execute_model()  │
     └───────────┬────────────────────────────┘
                 │
     ┌───────────▼────────────────────────────┐
     │ 9. ModelRunner.execute_model()          │
     │    ├─ Prepare inputs (tokens, pos)      │
     │    ├─ Forward pass through model        │
     │    └─ Get logits                        │
     └───────────┬────────────────────────────┘
                 │
     ┌───────────▼────────────────────────────┐
     │ 10. Model.forward() - LlamaForCausalLM  │
     │     ├─ Embedding: tokens → embeddings   │
     │     ├─ Transformer layers (attn + FFN)  │
     │     ├─ Each layer processes KV cache    │
     │     └─ Final norm, return hidden states │
     └───────────┬────────────────────────────┘
                 │
     ┌───────────▼────────────────────────────┐
     │ 11. Sampler.forward()                   │
     │     ├─ Apply temperature/top-p/top-k    │
     │     ├─ Softmax → probabilities          │
     │     ├─ Multinomial sampling             │
     │     └─ Return sampled token             │
     └───────────┬────────────────────────────┘
                 │
     ┌───────────▼────────────────────────────┐
     │ 12. Output Processing                   │
     │     ├─ Update sequence with token       │
     │     ├─ Check stopping conditions        │
     │     ├─ Detokenize to text               │
     │     └─ Return to stream                 │
     └───────────┬────────────────────────────┘
                 │
     ┌───────────▼────────────────────────────┐
     │ 13. Return to Client                    │
     │     - Single response (non-streaming)   │
     │     - Streaming chunks (streaming)      │
     └───────────────────────────────────────┘
```

---

## Key Data Structures

| Structure | Location | Purpose |
|-----------|----------|---------|
| `ChatCompletionRequest` | `protocol.py` | OpenAI API request |
| `SamplingParams` | `sampling_params.py` | Sampling configuration (temp, top-p, etc) |
| `Sequence` / `SequenceGroup` | `sequence.py` | Internal representation of requests |
| `SequenceGroupMetadata` | `sequence.py` | Metadata for scheduler/executor |
| `SchedulerOutputs` | `scheduler.py` | Scheduling decisions (what to run) |
| `RequestOutput` | `outputs.py` | Per-request generation output |
| `SamplerOutput` | `sampler.py` | Sampled tokens + logprobs |
| `ExecuteModelRequest` | `sequence.py` | Model execution input |

---

## Performance Optimizations in v0.6.0

1. **Continuous Batching**: Multiple requests processed together
2. **PagedAttention**: KV cache stored in fixed-size blocks
3. **Tensor Parallelism**: Model weights distributed across GPUs
4. **Iteration-level Scheduling**: Per-iteration scheduling decisions
5. **KV Cache Reuse**: Previous tokens' KV cached for prefill/decode
6. **Memory Swapping**: CPU ↔ GPU swapping for overcommitted resources
7. **Preemption**: Low-priority requests paused when resources tight
8. **Async Processing**: Non-blocking request handling via asyncio

---

## Summary

The vLLM v0.6.0 architecture provides a clean, efficient pipeline from OpenAI-compatible API calls to GPU model execution:

- **Two-tier design**: Optional multiprocessing for scalability
- **Async I/O**: Non-blocking request handling
- **Dynamic scheduling**: Efficient resource management
- **Clean separation**: Each layer handles one concern
- **OpenAI compatibility**: Drop-in replacement for OpenAI API

This design enables vLLM to achieve industry-leading throughput and latency for LLM serving.
