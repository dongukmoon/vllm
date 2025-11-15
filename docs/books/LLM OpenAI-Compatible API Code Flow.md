# vLLM OpenAI-Compatible API Code Flow (Latest Version)

This document traces the complete code flow from the OpenAI-compatible API entry point down to the actual model inference execution in vLLM's latest architecture.

## Architecture Overview

vLLM has evolved to support two engine versions:
- **V0 Engine**: Legacy engine (similar to v0.1.0)
- **V1 Engine**: New high-performance engine (used when `VLLM_USE_V1=1`)

The flow follows this path:
```
CLI Command → FastAPI Server → Serving Layer → Engine Client → Engine Core → Executor → Model → Sampler
```

---

## 1. Entry Point: CLI Command

**File:** `pyproject.toml`
```toml
[project.scripts]
vllm = "vllm.entrypoints.cli.main:main"
```

### Command Usage:
```bash
vllm serve meta-llama/Llama-3-8B --host 0.0.0.0 --port 8000
```

### Main CLI Handler
**File:** `vllm/entrypoints/cli/main.py`

```python
def main():
    # Import subcommand modules lazily
    import vllm.entrypoints.cli.serve
    import vllm.entrypoints.cli.openai
    # ... other modules
    
    CMD_MODULES = [
        vllm.entrypoints.cli.serve,
        vllm.entrypoints.cli.openai,
        # ...
    ]
    
    # Create argument parser
    parser = FlexibleArgumentParser(description="vLLM CLI")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    
    # Register all subcommands
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers)
            cmd.set_defaults(dispatch_function=cmd.cmd)
    
    # Parse and dispatch
    args = parser.parse_args()
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)  # Calls ServeSubcommand.cmd()
```

---

## 2. Serve Subcommand

**File:** `vllm/entrypoints/cli/serve.py`

```python
class ServeSubcommand(CLISubcommand):
    name = "serve"
    
    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # Handle model positional argument
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag
        
        if args.headless or args.api_server_count < 1:
            run_headless(args)  # Engine-only mode
        else:
            if args.api_server_count > 1:
                run_multi_api_server(args)  # Multiple API servers
            else:
                # Single API server (most common)
                uvloop.run(run_server(args))
```

### Run Server Function
**File:** `vllm/entrypoints/openai/api_server.py`

```python
async def run_server(args: Namespace):
    # 1. Setup server socket
    listen_address, sock = setup_server(args)
    
    # 2. Create engine configuration
    engine_args = AsyncEngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )
    
    # 3. Create FastAPI app with lifespan
    app = create_app(args, vllm_config)
    
    # 4. Start uvicorn server
    config = uvicorn.Config(
        app,
        host=listen_address[0],
        port=listen_address[1],
        log_config=None,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )
    await uvicorn.Server(config).serve(sockets=[sock])
```

---

## 3. FastAPI Application Setup

**File:** `vllm/entrypoints/openai/api_server.py`

### Application Lifespan
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Start background stats logging if enabled
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client
            
            async def _force_log():
                while True:
                    await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
                    await engine_client.do_log_stats()
            
            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
        
        yield  # Server is running
    finally:
        # Cleanup on shutdown
        await engine_client.shutdown()
        # Cancel background tasks
        for task in _running_tasks:
            task.cancel()
```

### App Creation
```python
def create_app(args: Namespace, vllm_config: VllmConfig) -> FastAPI:
    # Create FastAPI app
    app = FastAPI(lifespan=lifespan)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    
    # Create engine client
    engine_client = _make_engine_client(args, vllm_config)
    
    # Store state
    app.state.engine_client = engine_client
    app.state.vllm_config = vllm_config
    app.state.log_stats = not args.disable_log_stats
    
    # Mount router with all endpoints
    app.include_router(router)
    
    # Mount Prometheus metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    return app
```

---

## 4. API Endpoint Handlers

**File:** `vllm/entrypoints/openai/api_server.py`

### Chat Completion Endpoint
```python
@router.post("/v1/chat/completions")
@with_cancellation
@load_aware_call
async def create_chat_completion(
    request: ChatCompletionRequest, 
    raw_request: Request
):
    # 1. Get the chat handler
    handler = chat(raw_request)  # Returns OpenAIServingChat instance
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Chat Completions API"
        )
    
    # 2. Create chat completion via handler
    try:
        generator = await handler.create_chat_completion(request, raw_request)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, 
            detail=str(e)
        ) from e
    
    # 3. Return response based on type
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), 
            status_code=generator.error.code
        )
    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())
    
    # Streaming response
    return StreamingResponse(content=generator, media_type="text/event-stream")
```

### Completion Endpoint
```python
@router.post("/v1/completions")
@with_cancellation
@load_aware_call
async def create_completion(
    request: CompletionRequest, 
    raw_request: Request
):
    # 1. Get completion handler
    handler = completion(raw_request)  # Returns OpenAIServingCompletion
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Completions API"
        )
    
    # 2. Create completion
    try:
        generator = await handler.create_completion(request, raw_request)
    except OverflowError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value, 
            detail=str(e)
        ) from e
    
    # 3. Return appropriate response
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), 
            status_code=generator.error.code
        )
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())
    
    return StreamingResponse(content=generator, media_type="text/event-stream")
```

---

## 5. Serving Layer

### OpenAIServingChat
**File:** `vllm/entrypoints/openai/serving_chat.py`

```python
class OpenAIServingChat(OpenAIServing):
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        
        # 1. Validate model
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret
        
        # 2. Check if engine is alive
        if self.engine_client.errored:
            raise self.engine_client.dead_error
        
        # 3. Parse and validate request
        conversation = parse_chat_input(request, self.models_manager, ...)
        
        # 4. Apply chat template
        prompt_input = self._apply_chat_template(
            conversation,
            chat_template=request.chat_template or self.chat_template,
            ...
        )
        
        # 5. Create sampling parameters
        sampling_params = self._create_sampling_params(request)
        
        # 6. Generate request ID
        request_id = f"chat-{random_uuid()}"
        
        # 7. Call engine to generate
        try:
            result_generator = self.engine_client.generate(
                inputs=prompt_input,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )
        except ValueError as e:
            return self.create_error_response(str(e))
        
        # 8. Process outputs
        if request.stream:
            return self._chat_completion_stream_generator(
                request, result_generator, ...
            )
        else:
            return await self._chat_completion_full_generator(
                request, result_generator, ...
            )
```

### OpenAIServingCompletion
**File:** `vllm/entrypoints/openai/serving_completion.py`

```python
class OpenAIServingCompletion(OpenAIServing):
    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | CompletionResponse | ErrorResponse:
        
        # 1. Validate and check errors
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret
        
        # 2. Tokenize prompt
        prompt_input = self._tokenize_prompt_input(
            request,
            request.prompt,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
        )
        
        # 3. Create sampling parameters
        sampling_params = self._create_sampling_params(request)
        
        # 4. Generate request ID
        request_id = f"cmpl-{random_uuid()}"
        
        # 5. Call engine
        try:
            result_generator = self.engine_client.generate(
                inputs=prompt_input,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )
        except ValueError as e:
            return self.create_error_response(str(e))
        
        # 6. Process results
        if request.stream:
            return self._completion_stream_generator(
                request, result_generator, ...
            )
        else:
            return await self._completion_full_generator(
                request, result_generator, ...
            )
```

---

## 6. Engine Client Layer

**File:** `vllm/engine/protocol.py`

```python
class EngineClient(ABC):
    """Protocol class for Clients to Engine"""
    
    @abstractmethod
    def generate(
        self,
        inputs: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: dict[str, str] | None = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...
```

### V1 Engine Implementation
**File:** `vllm/v1/engine/llm_engine.py`

```python
class LLMEngine:
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        ...
    ):
        # 1. Initialize processor (tokenization, input processing)
        self.processor = Processor(self.vllm_config, tokenizer)
        
        # 2. Initialize I/O processor
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.model_config.io_processor_plugin,
        )
        
        # 3. Initialize output processor
        self.output_processor = OutputProcessor(
            self.tokenizer, 
            log_stats=self.log_stats
        )
        
        # 4. Create engine core client
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )
    
    async def generate(
        self,
        inputs: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        ...
    ) -> AsyncGenerator[RequestOutput, None]:
        
        # 1. Process inputs
        processed_inputs = await self.processor.process_inputs_async(
            request_id, inputs, ...
        )
        
        # 2. Create engine core request
        engine_core_request = EngineCoreRequest(
            request_id=request_id,
            prompt=processed_inputs["prompt"],
            prompt_token_ids=processed_inputs["prompt_token_ids"],
            mm_inputs=processed_inputs.get("mm_inputs"),
            sampling_params=sampling_params,
            lora_request=lora_request,
            ...
        )
        
        # 3. Submit to engine core
        engine_core_outputs_generator = self.engine_core.generate(
            engine_core_request
        )
        
        # 4. Process outputs
        async for engine_core_outputs in engine_core_outputs_generator:
            request_output = self.output_processor.process_outputs(
                request_id,
                engine_core_outputs,
            )
            yield request_output
```

---

## 7. Engine Core Layer

**File:** `vllm/v1/engine/core.py`

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        ...
    ):
        # 1. Setup model executor
        self.model_executor = executor_class(vllm_config)
        
        # 2. Initialize KV caches
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
            self._initialize_kv_caches(vllm_config)
        
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        
        # 3. Initialize cache on workers
        self.collective_rpc(
            "initialize_cache", 
            args=(num_gpu_blocks, num_cpu_blocks)
        )
        
        # 4. Create scheduler
        self.scheduler = Scheduler(
            scheduler_config=vllm_config.scheduler_config,
            kv_cache_config=kv_cache_config,
            lora_config=vllm_config.lora_config,
            parallel_config=vllm_config.parallel_config,
            ...
        )
    
    def step(self) -> list[EngineCoreOutputs]:
        """Main execution loop - one step of inference."""
        
        # 1. SCHEDULING: Decide what to execute
        scheduler_output: SchedulerOutput = self.scheduler.schedule()
        
        if scheduler_output.is_empty():
            return []
        
        # 2. EXECUTE MODEL: Run inference
        model_output: ModelRunnerOutput = self.model_executor.execute_model(
            scheduler_output=scheduler_output
        )
        
        # 3. UPDATE SCHEDULER: Process results
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output=scheduler_output,
            model_output=model_output,
        )
        
        # 4. Return outputs
        return engine_core_outputs
    
    def add_request(self, request: EngineCoreRequest):
        """Add a new request to the scheduler."""
        
        # Convert to internal Request object
        req = Request(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            mm_inputs=request.mm_inputs,
            sampling_params=request.sampling_params,
            arrival_time=time.time(),
            ...
        )
        
        # Add to scheduler's waiting queue
        self.scheduler.add_request(req)
```

---

## 8. Scheduler Layer

**File:** `vllm/v1/core/sched/scheduler.py`

```python
class Scheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        kv_cache_config: KVCacheConfig,
        ...
    ):
        # Request queues
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: deque[Request] = deque(maxlen=1024)
        
        # KV cache manager
        self.kv_cache_manager = KVCacheManager(
            block_size=kv_cache_config.block_size,
            num_gpu_blocks=kv_cache_config.num_gpu_blocks,
            ...
        )
    
    def schedule(self) -> SchedulerOutput:
        """Schedule requests for execution."""
        
        # 1. Move waiting requests to running if possible
        while self.waiting:
            req = self.waiting[0]
            
            # Check if we can allocate KV cache blocks
            if not self.kv_cache_manager.can_allocate(req):
                break
            
            # Allocate blocks
            self.kv_cache_manager.allocate(req)
            
            # Move to running queue
            self.waiting.popleft()
            self.running.append(req)
        
        # 2. Prepare scheduler output
        scheduler_output = SchedulerOutput(
            scheduled_requests=self.running,
            num_scheduled_tokens=self._count_tokens(self.running),
            total_num_requests=len(self.waiting) + len(self.running),
        )
        
        return scheduler_output
    
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> list[EngineCoreOutputs]:
        """Update state after model execution."""
        
        outputs = []
        
        for req in scheduler_output.scheduled_requests:
            # Get sampled token for this request
            sampled_token_id = model_output.sampled_token_ids[req.request_id]
            
            # Append token to request
            req.append_output_token(sampled_token_id)
            
            # Check if finished
            if self._is_finished(req):
                self.running.remove(req)
                self.finished.append(req)
                self.kv_cache_manager.free(req)
            
            # Create output
            engine_output = EngineCoreOutputs(
                request_id=req.request_id,
                new_token_ids=[sampled_token_id],
                finished=req.finished,
                ...
            )
            outputs.append(engine_output)
        
        return outputs
```

---

## 9. Executor Layer

**File:** `vllm/v1/executor/*.py`

```python
class Executor:
    """Base class for executors."""
    
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config
        
        # Create workers
        self._init_workers()
    
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """Execute model on all workers."""
        
        # Broadcast to all workers
        output = self._run_workers(
            "execute_model",
            scheduler_output=scheduler_output,
        )
        
        return output


class GPUExecutor(Executor):
    """GPU executor for single or multi-GPU inference."""
    
    def _init_workers(self):
        # Initialize workers based on tensor parallel size
        self.workers = []
        for rank in range(self.parallel_config.tensor_parallel_size):
            worker = Worker(
                vllm_config=self.vllm_config,
                rank=rank,
                distributed_init_method=self.distributed_init_method,
            )
            self.workers.append(worker)
    
    def _run_workers(self, method: str, **kwargs):
        # Execute method on all workers
        outputs = []
        for worker in self.workers:
            output = getattr(worker, method)(**kwargs)
            outputs.append(output)
        
        # Return output from first worker (all should be same for single GPU)
        return outputs[0]
```

---

## 10. Worker Layer

**File:** `vllm/v1/worker/worker.py`

```python
class Worker:
    """GPU worker that executes the model."""
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        rank: int,
        distributed_init_method: str,
    ):
        self.vllm_config = vllm_config
        self.rank = rank
        
        # Initialize distributed environment
        self._init_distributed_environment()
        
        # Load model
        self.model_runner = ModelRunner(
            vllm_config=vllm_config,
            rank=rank,
        )
    
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """Execute model forward pass."""
        
        # Delegate to model runner
        return self.model_runner.execute_model(scheduler_output)
    
    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ):
        """Initialize KV cache."""
        self.model_runner.initialize_cache(num_gpu_blocks, num_cpu_blocks)
```

---

## 11. Model Runner Layer

**File:** `vllm/v1/worker/model_runner.py`

```python
class ModelRunner:
    """Runs the model on a single GPU."""
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        rank: int,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        
        # Load model
        self.model = self._load_model()
        
        # Create KV cache
        self.kv_cache = None  # Initialized later
    
    def _load_model(self):
        """Load the model."""
        from vllm.model_executor.model_loader import get_model
        
        model = get_model(
            model_config=self.model_config,
            device_config=self.vllm_config.device_config,
            ...
        )
        
        return model.eval()
    
    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ):
        """Initialize KV cache."""
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
        
        self.kv_cache = FlashAttentionBackend.create_kv_cache(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=self.model_config.cache_config.block_size,
            ...
        )
    
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """Execute the model forward pass."""
        
        # 1. Prepare input tensors
        input_tensors = self._prepare_inputs(scheduler_output)
        
        # 2. Execute model forward pass
        hidden_states = self.model(
            input_ids=input_tensors.input_ids,
            positions=input_tensors.positions,
            kv_caches=self.kv_cache,
            attn_metadata=input_tensors.attn_metadata,
        )
        
        # 3. Sample next tokens
        sampled_token_ids = self._sample(
            hidden_states=hidden_states,
            sampling_metadata=input_tensors.sampling_metadata,
        )
        
        # 4. Return output
        return ModelRunnerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs=...,
        )
    
    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> InputTensors:
        """Prepare input tensors from scheduler output."""
        
        # Collect input tokens and positions
        input_tokens = []
        input_positions = []
        
        for req in scheduler_output.scheduled_requests:
            if req.is_first_token:
                # Prefill: use all prompt tokens
                tokens = req.prompt_token_ids
                positions = list(range(len(tokens)))
            else:
                # Decode: use last generated token
                tokens = [req.output_token_ids[-1]]
                positions = [len(req.prompt_token_ids) + len(req.output_token_ids) - 1]
            
            input_tokens.extend(tokens)
            input_positions.extend(positions)
        
        # Convert to tensors
        input_ids = torch.tensor(input_tokens, dtype=torch.long, device="cuda")
        positions = torch.tensor(input_positions, dtype=torch.long, device="cuda")
        
        # Create attention metadata
        attn_metadata = self._create_attn_metadata(scheduler_output)
        
        return InputTensors(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_metadata=self._create_sampling_metadata(scheduler_output),
        )
```

---

## 12. Model Layer

**File:** `vllm/model_executor/models/*.py` (e.g., `llama.py`)

```python
class LlamaForCausalLM(nn.Module):
    """Llama model for causal language modeling."""
    
    def __init__(self, config: LlamaConfig, ...):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        
        # Output normalization
        self.norm = RMSNorm(config.hidden_size)
        
        # Language model head
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )
        
        # Sampler
        self.sampler = Sampler()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
        ...
    ) -> torch.Tensor:
        """Forward pass through the model."""
        
        # 1. EMBEDDING: Convert token IDs to embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # 2. TRANSFORMER LAYERS: Pass through all layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_caches[i],
                attn_metadata=attn_metadata,
            )
        
        # 3. NORMALIZATION: Final layer norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class LlamaDecoderLayer(nn.Module):
    """Single transformer layer."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        
        # Self-attention
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
        )
        
        # Feed-forward network
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        
        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        
        # 1. SELF-ATTENTION with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states
        
        # 2. FEED-FORWARD with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
```

---

## 13. Sampling Layer

**File:** `vllm/model_executor/layers/sampler.py`

```python
class Sampler(nn.Module):
    """Samples the next tokens from logits."""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sample next tokens."""
        
        # 1. Get logits for positions that need sampling
        logits_to_sample = self._get_logits_to_sample(
            logits, 
            sampling_metadata
        )
        
        # 2. Apply sampling parameters
        for i, sampling_params in enumerate(sampling_metadata.seq_groups):
            seq_logits = logits_to_sample[i]
            
            # Apply temperature
            if sampling_params.temperature != 1.0:
                seq_logits = seq_logits / sampling_params.temperature
            
            # Apply top-p
            if sampling_params.top_p < 1.0:
                seq_logits = self._apply_top_p(seq_logits, sampling_params.top_p)
            
            # Apply top-k
            if sampling_params.top_k > 0:
                seq_logits = self._apply_top_k(seq_logits, sampling_params.top_k)
            
            logits_to_sample[i] = seq_logits
        
        # 3. Convert to probabilities
        probs = torch.softmax(logits_to_sample, dim=-1)
        
        # 4. Sample tokens
        sampled_tokens = torch.multinomial(
            probs, 
            num_samples=1,
            replacement=True
        ).squeeze(-1)
        
        return sampled_tokens
    
    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus (top-p) sampling."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), 
            dim=-1
        )
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[0] = False  # Keep at least one token
        
        # Set logits to -inf for removed tokens
        logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
        
        return logits
    
    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Top-k sampling."""
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        
        # Create mask for top-k tokens
        mask = torch.full_like(logits, float('-inf'))
        mask[top_k_indices] = top_k_logits
        
        return mask
```

---

## 14. Complete Flow Summary

### Request Flow (Non-Streaming)

```
1. HTTP POST /v1/chat/completions
   ↓
2. create_chat_completion() handler
   ↓
3. OpenAIServingChat.create_chat_completion()
   - Parse request
   - Apply chat template
   - Create sampling params
   ↓
4. EngineClient.generate()
   ↓
5. LLMEngine.generate()
   - Process inputs (Processor)
   - Create EngineCoreRequest
   ↓
6. EngineCore.add_request()
   - Add to scheduler's waiting queue
   ↓
7. EngineCore.step() (continuous loop)
   ├─ Scheduler.schedule()
   │  - Move requests from waiting → running
   │  - Allocate KV cache blocks
   │  - Create SchedulerOutput
   ├─ Executor.execute_model()
   │  ├─ Worker.execute_model()
   │  │  └─ ModelRunner.execute_model()
   │  │     ├─ Prepare inputs
   │  │     ├─ Model.forward()
   │  │     │  ├─ Embedding
   │  │     │  ├─ Transformer layers (attention + FFN)
   │  │     │  └─ Final norm
   │  │     └─ Sampler.forward()
   │  │        ├─ Apply temperature
   │  │        ├─ Apply top-p/top-k
   │  │        ├─ Softmax → probabilities
   │  │        └─ Multinomial sampling
   │  └─ Return ModelRunnerOutput (sampled tokens)
   └─ Scheduler.update_from_output()
      - Append tokens to requests
      - Check if finished
      - Free KV cache for completed
      - Return EngineCoreOutputs
   ↓
8. OutputProcessor.process_outputs()
   - Convert EngineCoreOutputs → RequestOutput
   - Detokenize if needed
   ↓
9. OpenAIServingChat yields RequestOutput
   - Format as ChatCompletionResponse
   ↓
10. FastAPI returns JSON response to client
```

### Key Data Structures

| Structure | Description |
|-----------|-------------|
| `ChatCompletionRequest` | API request from client |
| `SamplingParams` | Temperature, top-p, top-k, etc. |
| `EngineCoreRequest` | Internal request to engine core |
| `Request` | Scheduler's request object with state |
| `SchedulerOutput` | What to execute in this step |
| `ModelRunnerOutput` | Sampled tokens + logprobs |
| `EngineCoreOutputs` | Per-request outputs from core |
| `RequestOutput` | Final output with text + metadata |
| `ChatCompletionResponse` | OpenAI-format response |

### Performance Optimizations

1. **Continuous Batching**: Dynamic batching of requests
2. **PagedAttention**: Efficient KV cache with paging
3. **Tensor Parallelism**: Model sharded across GPUs
4. **V1 Engine Architecture**: Decoupled scheduling and execution
5. **Multiprocessing**: Engine core in separate process
6. **Async I/O**: Non-blocking request handling
7. **Chunked Prefill**: Split long prompts into chunks
8. **Speculative Decoding**: Draft model for faster generation

### V0 vs V1 Engine Differences

| Feature | V0 Engine | V1 Engine |
|---------|-----------|-----------|
| Architecture | Monolithic | Decoupled (Engine Core) |
| Scheduling | Iteration-level | More flexible |
| Performance | Good | Better (lower latency) |
| Multiprocessing | Optional | Built-in |
| Activation | `VLLM_USE_V1=0` | `VLLM_USE_V1=1` |

---

## Conclusion

The vLLM architecture provides a sophisticated, high-performance serving system for LLMs with:

- **Clean separation of concerns**: API → Serving → Engine → Executor → Model
- **Efficient resource management**: Dynamic scheduling with KV cache paging
- **OpenAI compatibility**: Drop-in replacement for OpenAI API
- **Scalability**: Support for distributed inference and multiple API servers
- **Flexibility**: Pluggable schedulers, executors, and model implementations

This architecture enables vLLM to achieve industry-leading throughput and latency for LLM serving.
