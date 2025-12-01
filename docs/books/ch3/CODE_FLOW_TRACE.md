# vLLM Code Flow Trace: `vllm serve --model gpt2`

## Overview
This document traces the complete code flow when executing `vllm serve --model gpt2` in vLLM v0.11.0. The execution path shows how the CLI command is parsed, the engine is initialized, and the API server is started.

---

## 1. Entry Point: CLI Command Execution

**File**: `vllm/entrypoints/cli/main.py`

### Flow:
```python
def main():
    # 1. Lazy imports to avoid circular dependencies
    import vllm.entrypoints.cli.serve
    
    # 2. Load CLI environment
    cli_env_setup()
    
    # 3. Create main parser
    parser = FlexibleArgumentParser(description="vLLM CLI")
    
    # 4. Register all subcommands (openai, serve, benchmark, etc.)
    CMD_MODULES = [
        vllm.entrypoints.cli.openai,
        vllm.entrypoints.cli.serve,  # <-- Our target
        vllm.entrypoints.cli.benchmark.main,
        vllm.entrypoints.cli.collect_env,
        vllm.entrypoints.cli.run_batch,
    ]
    
    # 5. Create subparsers and register each command
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()  # Returns [ServeSubcommand()]
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(
                dispatch_function=cmd.cmd)
    
    # 6. Parse arguments
    args = parser.parse_args()  # args.subparser = "serve"
    
    # 7. Call dispatch function (ServeSubcommand.cmd)
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
```

**Key Entry Point**: `vllm/entrypoints/cli/serve.py:ServeSubcommand.cmd()`

---

## 2. Serve Subcommand Processing

**File**: `vllm/entrypoints/cli/serve.py`

### Class: `ServeSubcommand`

```python
class ServeSubcommand(CLISubcommand):
    name = "serve"
    
    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # Step 1: Handle model_tag from positional argument
        if hasattr(args, 'model_tag') and args.model_tag is not None:
            args.model = args.model_tag  # args.model = "gpt2"
        
        # Step 2: Check server configuration
        if args.headless or args.api_server_count < 1:
            run_headless(args)
        else:
            if args.api_server_count > 1:
                run_multi_api_server(args)
            else:
                # Single API server (default case)
                # This is the typical path for `vllm serve --model gpt2`
                uvloop.run(run_server(args))
```

**For single API server (default)**: Calls `run_server(args)` using uvloop event loop

---

## 3. API Server Setup

**File**: `vllm/entrypoints/openai/api_server.py`

### Step 3.1: `run_server(args)` - AsyncIO Setup

```python
async def run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""
    
    # 1. Add process-specific prefix to logs
    decorate_logs("APIServer")
    
    # 2. Setup server socket and listen address
    listen_address, sock = setup_server(args)
    # Returns: "http://0.0.0.0:8000" and socket
    
    # 3. Call actual server worker
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
```

### Step 3.2: `setup_server(args)` - Socket Binding

```python
def setup_server(args):
    """Validate API server args, set up signal handler, create socket ready to serve."""
    
    logger.info("vLLM API server version %s", VLLM_VERSION)
    log_non_default_args(args)
    
    # Validate tool parsers and reasoning parsers
    validate_api_server_args(args)
    
    # Create TCP socket bound to host:port (default 0.0.0.0:8000)
    sock_addr = (args.host or "", args.port)  # ("", 8000)
    sock = create_server_socket(sock_addr)
    
    # Set ulimits for concurrency
    set_ulimit()
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Construct listen address for uvicorn
    listen_address = f"http://{host_part}:{port}"  # "http://0.0.0.0:8000"
    
    return listen_address, sock
```

### Step 3.3: `run_server_worker(listen_address, sock, args)` - Core Engine Setup

```python
async def run_server_worker(listen_address, sock, args, 
                           client_config=None, **uvicorn_kwargs) -> None:
    """Run a single API server worker."""
    
    # 1. Import tool parsers if specified
    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)
    
    # 2. Load logging config for uvicorn
    log_config = load_log_config(args.log_config_file)
    
    # 3. BUILD ASYNC ENGINE CLIENT (Main initialization)
    async with build_async_engine_client(args, client_config=client_config) \
            as engine_client:
        
        # 4. Register optional tokenizer info endpoint
        maybe_register_tokenizer_info_endpoint(args)
        
        # 5. Build FastAPI application
        app = build_app(args)
        
        # 6. Get vllm config from engine client
        vllm_config = await engine_client.get_vllm_config()
        
        # 7. Initialize FastAPI app state
        await init_app_state(engine_client, vllm_config, app.state, args)
        
        logger.info("Starting vLLM API server on %s", listen_address)
        
        # 8. Start HTTP server using uvicorn
        shutdown_task = await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            **uvicorn_kwargs,
        )
    
    # 9. Wait for shutdown
    await shutdown_task
    sock.close()
```

---

## 4. Engine Client Initialization - **CRITICAL STEP**

**File**: `vllm/entrypoints/openai/api_server.py`

### Step 4.1: `build_async_engine_client(args)` - Engine Arguments

```python
@asynccontextmanager
async def build_async_engine_client(
    args: Namespace,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    disable_frontend_multiprocessing: Optional[bool] = None,
    client_config: Optional[dict[str, Any]] = None,
) -> AsyncIterator[EngineClient]:
    
    # 1. Setup forkserver for multiprocessing (if configured)
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        multiprocessing.set_start_method('forkserver')
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
    
    # 2. Convert CLI args to AsyncEngineArgs
    engine_args = AsyncEngineArgs.from_cli_args(args)
    # This extracts: model="gpt2", tensor_parallel_size, gpu_memory_fraction, etc.
    
    # 3. Delegate to engine args builder
    async with build_async_engine_client_from_engine_args(
            engine_args,
            usage_context=usage_context,
            disable_frontend_multiprocessing=disable_frontend_multiprocessing,
            client_config=client_config,
    ) as engine:
        yield engine
```

**File**: `vllm/engine/arg_utils.py`

```python
class AsyncEngineArgs:
    @staticmethod
    def from_cli_args(args):
        """Convert CLI args to AsyncEngineArgs instance"""
        # Extracts model, tensor_parallel_size, gpu_memory_fraction, 
        # quantization, load_format, etc. from args
        return AsyncEngineArgs(
            model=args.model,  # "gpt2"
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            # ... many other parameters ...
        )
```

### Step 4.2: `build_async_engine_client_from_engine_args()` - Config Creation

```python
@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient:
    - in-process using the AsyncLLMEngine Directly
    - multiprocess using AsyncLLMEngine RPC
    """
    
    # 1. CREATE VLLM CONFIG (Model configuration & resource allocation)
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    # This step:
    # - Loads model from HuggingFace ("gpt2")
    # - Determines model config (vocab_size, hidden_size, etc.)
    # - Sets up parallel config (tensor, pipeline, data parallel)
    # - Allocates GPU memory
    
    # 2. Check V1 is enabled
    assert envs.VLLM_USE_V1  # Version 1 engine must be enabled
    
    # 3. Create AsyncLLM instance
    from vllm.v1.engine.async_llm import AsyncLLM
    
    async_llm = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=usage_context,
        enable_log_requests=engine_args.enable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
        client_addresses=client_config,
        client_count=client_count,
        client_index=client_index
    )
    
    # 4. Reset multimodal cache to free memory
    await async_llm.reset_mm_cache()
    
    # 5. Yield the engine client
    yield async_llm
    
    # 6. Cleanup on exit
    if async_llm:
        async_llm.shutdown()
```

---

## 5. VllmConfig Creation - **MODEL LOADING**

**File**: `vllm/engine/arg_utils.py`

### Step 5.1: `AsyncEngineArgs.create_engine_config()`

```python
def create_engine_config(
    self,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    headless: bool = False,
) -> VllmConfig:
    """Create VllmConfig from AsyncEngineArgs"""
    
    # 1. Create ModelConfig (loads model from HF)
    model_config = ModelConfig(
        model=self.model,  # "gpt2"
        tokenizer=self.tokenizer,
        load_format=self.load_format,
        seed=self.seed,
        # ... other model config parameters ...
    )
    # This loads the model:
    # - Download from HF if not cached
    # - Load tokenizer
    # - Get model dimensions, vocab size, etc.
    
    # 2. Create ParallelConfig (device placement strategy)
    parallel_config = ParallelConfig(
        tensor_parallel_size=self.tensor_parallel_size,
        pipeline_parallel_size=self.pipeline_parallel_size,
        # ... other parallel config parameters ...
    )
    
    # 3. Create CacheConfig (KV cache allocation)
    cache_config = CacheConfig(
        block_size=self.block_size,
        gpu_memory_utilization=self.gpu_memory_fraction,
        # ... cache parameters ...
    )
    
    # 4. Create SchedulerConfig (request scheduling)
    scheduler_config = SchedulerConfig(
        max_num_seqs=self.max_num_seqs,
        max_model_len=self.max_model_len,
        # ... scheduler parameters ...
    )
    
    # 5. Load plugins (LoRA, tools, etc.)
    load_general_plugins()
    
    # 6. Assemble VllmConfig
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        # ... other configs ...
    )
    
    return vllm_config
```

---

## 6. AsyncLLM Engine Initialization

**File**: `vllm/v1/engine/async_llm.py`

### Step 6.1: `AsyncLLM.from_vllm_config()`

```python
@classmethod
def from_vllm_config(
    cls,
    vllm_config: VllmConfig,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    enable_log_requests: bool = True,
    disable_log_stats: bool = False,
    client_addresses: Optional[dict[str, str]] = None,
    client_count: int = 1,
    client_index: int = 0,
) -> "AsyncLLM":
    """Create AsyncLLM from VllmConfig"""
    
    # 1. Get appropriate executor class based on config
    executor_class = Executor.get_class(vllm_config)
    
    # 2. Setup multimodal registry
    mm_registry = MULTIMODAL_REGISTRY
    # If custom multimodal configs, register them here
    
    # 3. Create AsyncLLM instance
    return cls(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=(not disable_log_stats),
        usage_context=usage_context,
        mm_registry=mm_registry,
        use_cached_outputs=False,
        log_requests=enable_log_requests,
        start_engine_loop=True,
        client_addresses=client_addresses,
        client_count=client_count,
        client_index=client_index,
    )
```

### Step 6.2: `AsyncLLM.__init__()` - Engine Core Setup

```python
def __init__(
    self,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    use_cached_outputs: bool = False,
    log_requests: bool = True,
    start_engine_loop: bool = True,
    stat_loggers: Optional[list[StatLoggerFactory]] = None,
    client_addresses: Optional[dict[str, str]] = None,
    client_count: int = 1,
    client_index: int = 0,
) -> None:
    """Initialize AsyncLLM engine"""
    
    # 1. Validate V1 is enabled
    if not envs.VLLM_USE_V1:
        raise ValueError("V1 AsyncLLMEngine but VLLM_USE_V1=False")
    
    # 2. Serialize custom transformer configs
    maybe_register_config_serialize_by_value()
    
    # 3. Store configurations
    self.model_config = vllm_config.model_config
    self.vllm_config = vllm_config
    self.executor_class = executor_class
    self.log_stats = log_stats
    self.usage_context = usage_context
    
    # 4. Initialize multimodal registry
    self.mm_registry = mm_registry
    
    # 5. Create tokenizer
    self.tokenizer = init_tokenizer_from_configs(
        model_config=vllm_config.model_config,
        tokenizer_mode=vllm_config.scheduler_config.tokenizer_mode,
    )
    
    # 6. Create input preprocessor
    self.input_preprocessor = InputPreprocessor(
        model_config=vllm_config.model_config,
        tokenizer=self.tokenizer,
        mm_registry=mm_registry,
    )
    
    # 7. Initialize tracing (if enabled)
    tracing_flags = DetailedTraceModules()
    init_tracer(
        vllm_config.model_config.model,
        tracing_flags,
        usage_context,
    )
    
    # 8. Setup stat loggers
    if stat_loggers is None:
        stat_loggers = StatLoggerFactory.get_default_stat_loggers()
    self.stat_logger_manager = StatLoggerManager(stat_loggers)
    
    # 9. Create ProcessorManager (handles request queuing and output)
    self.processor = Processor(
        scheduler_config=vllm_config.scheduler_config,
        log_stats=log_stats,
        stat_loggers=stat_loggers,
    )
    
    # 10. Initialize core engine components (request queue, output processor)
    self._init_engine_and_request_queues()
    
    # 11. Start engine loop if requested
    if start_engine_loop:
        self.start_engine_loop()
```

---

## 7. FastAPI Application Setup

**File**: `vllm/entrypoints/openai/api_server.py`

### Step 7.1: `build_app(args)` - API Endpoints

```python
def build_app(args: Namespace) -> FastAPI:
    """Build FastAPI application with all OpenAI-compatible endpoints"""
    
    # 1. Create FastAPI app with lifespan context manager
    app = FastAPI(
        openapi_url=None if args.disable_fastapi_docs else "/openapi.json",
        docs_url=None if args.disable_fastapi_docs else "/docs",
        redoc_url=None if args.disable_fastapi_docs else "/redoc",
        lifespan=lifespan,  # Handles app startup/shutdown
    )
    app.include_router(router)
    app.root_path = args.root_path
    
    # 2. Mount Prometheus metrics endpoint
    mount_metrics(app)
    
    # 3. Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    
    # 4. Add exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException):
        # ... error response formatting ...
        pass
    
    # 5. Add authentication middleware (if API key provided)
    if tokens := [key for key in (args.api_key or [envs.VLLM_API_KEY]) if key]:
        app.add_middleware(AuthenticationMiddleware, tokens=tokens)
    
    # 6. Add request ID middleware (if enabled)
    if args.enable_request_id_headers:
        app.add_middleware(XRequestIdMiddleware)
    
    # 7. Add scaling middleware
    app.add_middleware(ScalingMiddleware)
    
    # 8. Add custom middleware from args
    for middleware in args.middleware:
        # Dynamic middleware loading...
        pass
    
    return app
```

**Registered Endpoints** (via `router` - APIRouter):
- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion
- `POST /v1/completions` - Text completion
- `POST /v1/embeddings` - Embeddings
- `POST /v1/audio/transcriptions` - Speech-to-text
- `POST /v1/audio/translations` - Speech translation
- `POST /tokenize` - Tokenization
- And many more...

### Step 7.2: `init_app_state()` - Serving Handlers

```python
async def init_app_state(
    engine_client: EngineClient,
    vllm_config: VllmConfig,
    state: State,
    args: Namespace,
) -> None:
    """Initialize FastAPI app state with OpenAI serving handlers"""
    
    # 1. Setup model paths
    served_model_names = args.served_model_name or [args.model]
    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]
    
    # 2. Setup request logger
    request_logger = RequestLogger(...) if args.enable_log_requests else None
    
    # 3. Store engine client and config
    state.engine_client = engine_client
    state.vllm_config = vllm_config
    state.log_stats = not args.disable_log_stats
    
    # 4. Get supported tasks from engine
    supported_tasks = await engine_client.get_supported_tasks()
    # For LLMs: ["generate", "encode"]
    
    # 5. Create serving handlers for each capability
    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=vllm_config.model_config,
        base_model_paths=base_model_paths,
    )
    
    state.openai_serving_chat = OpenAIServingChat(...) \
        if "generate" in supported_tasks else None
    
    state.openai_serving_completion = OpenAIServingCompletion(...) \
        if "generate" in supported_tasks else None
    
    state.openai_serving_embedding = OpenAIServingEmbedding(...) \
        if "embed" in supported_tasks else None
    
    state.openai_serving_tokenization = OpenAIServingTokenization(...)
    
    # ... more serving handlers ...
```

---

## 8. HTTP Server Startup

**File**: `vllm/entrypoints/launcher.py`

### Step 8.1: `serve_http()` - Uvicorn Launch

```python
async def serve_http(
    app: FastAPI,
    sock: socket.socket,
    host: str,
    port: int,
    log_level: str = "info",
    access_log: bool = True,
    timeout_keep_alive: int = 5,
    ssl_keyfile: Optional[str] = None,
    ssl_certfile: Optional[str] = None,
    **uvicorn_kwargs
) -> Awaitable[None]:
    """Start uvicorn HTTP server"""
    
    # 1. Create uvicorn configuration
    config = uvicorn.Config(
        app=app,
        sock=sock,
        loop="uvloop",  # Use uvloop for high performance
        log_level=log_level,
        access_log=access_log,
        timeout_keep_alive=timeout_keep_alive,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        # ... other config ...
    )
    
    # 2. Create uvicorn server
    server = uvicorn.Server(config)
    
    # 3. Start server in asyncio task
    server_task = asyncio.create_task(server.serve())
    
    # 4. Return shutdown coroutine
    async def shutdown():
        server.should_exit = True
        await server_task
    
    return shutdown()
```

---

## 9. Request Handling Flow

Once the server is running, here's what happens when a request comes in:

### Example: `/v1/completions` Request

```python
# Client sends:
POST /v1/completions
{
    "model": "gpt2",
    "prompt": "Hello, world!",
    "max_tokens": 100
}

# 1. FastAPI routing finds create_completion endpoint
@router.post("/v1/completions")
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    # 2. Get completion handler from app state
    handler = completion(raw_request)  # OpenAIServingCompletion
    
    # 3. Call handler to generate completions
    generator = await handler.create_completion(request, raw_request)
    
    # 4. Return response (streaming or non-streaming)
    if isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())
    else:
        # Streaming response
        return StreamingResponse(content=generator, media_type="text/event-stream")

# 5. Inside OpenAIServingCompletion.create_completion():
async def create_completion(self, request, raw_request):
    # - Validate request
    # - Preprocess prompt with tokenizer
    # - Create SamplingParams from request
    # - Send to engine_client (AsyncLLM)
    # - Collect outputs as they stream from the model
    # - Format into CompletionResponse
    # - Yield or return response

# 6. Engine processes in AsyncLLM.generate():
async def generate(self, prompts, sampling_params, ...):
    # - Tokenize prompts
    # - Queue requests in processor
    # - Run inference loop in engine
    # - Stream output tokens back to API handler
    # - Return final output
```

---

## 10. Complete Request Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   User Command                              │
│           vllm serve --model gpt2                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            CLI Parsing (main.py)                            │
│   - Parse "serve" subcommand                               │
│   - Parse "--model gpt2"                                   │
│   - Call ServeSubcommand.cmd()                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Setup & Server (api_server.py)                    │
│   - setup_server(): Bind socket to 0.0.0.0:8000           │
│   - Call run_server_worker()                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│       Engine Initialization (api_server.py)                │
│   - build_async_engine_client()                            │
│   - create_engine_config() [LOADS MODEL]                   │
│   - AsyncLLM.from_vllm_config()                            │
│   - AsyncLLM.__init__() [CREATES ENGINE]                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        FastAPI Application Setup                           │
│   - build_app(): Create FastAPI with routes               │
│   - init_app_state(): Setup serving handlers              │
│   - Mount endpoints (/v1/completions, etc.)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Uvicorn HTTP Server Started                        │
│   - serve_http(): Start listening on 0.0.0.0:8000         │
│   - Ready to accept requests                              │
└─────────────────────────────────────────────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ▼                           ▼
    ┌──────────────┐           ┌──────────────┐
    │ POST /v1/    │           │ POST /v1/    │
    │completions   │           │chat/         │
    │              │           │completions   │
    └──────────────┘           └──────────────┘
           │                           │
           └─────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │  OpenAI Serving Handler  │
            │  (OpenAIServingCompletion)
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │   AsyncLLM.generate()    │
            │   (Model Inference)      │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │   Engine Core Loop       │
            │   (Process batches)      │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │  Executor (GPU/CPU)      │
            │  (Run model forward)     │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │  Output Processor        │
            │  (Stream tokens back)    │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │  API Response Formatter  │
            │  (JSON/SSE)              │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │  Return to Client        │
            │  (HTTP response)         │
            └──────────────────────────┘
```

---

## Key Components Summary

| Component | Location | Responsibility |
|-----------|----------|-----------------|
| **CLI Parser** | `cli/main.py` | Parse command line arguments |
| **Serve Command** | `cli/serve.py:ServeSubcommand` | Route to appropriate server |
| **API Server** | `openai/api_server.py` | Setup FastAPI and endpoints |
| **Engine Builder** | `openai/api_server.py:build_async_engine_client` | Create AsyncLLM engine |
| **Config Creator** | `engine/arg_utils.py` | Create VllmConfig with model |
| **Model Config** | `config/` | Load model from HF |
| **AsyncLLM** | `v1/engine/async_llm.py` | Main inference engine |
| **Processor** | `v1/engine/processor.py` | Request queuing & output |
| **Executor** | `v1/executor/` | GPU/CPU model execution |
| **HTTP Server** | `launcher.py` | Uvicorn HTTP server |
| **Serving Handlers** | `openai/serving_*.py` | OpenAI API compatibility |

---

## Configuration Resolution Order

When `vllm serve --model gpt2` is executed:

1. **CLI Arguments** (`--model gpt2`)
2. **Environment Variables** (override CLI if set)
3. **Defaults** (hardcoded in code)
4. **Auto-detection** (GPU memory, best config, etc.)

---

## Performance Optimizations Used

1. **uvloop**: High-performance asyncio event loop
2. **Socket Reuse**: Pre-created socket before engine initialization (avoids race conditions with Ray)
3. **Multiprocessing**: Optional data parallelism across multiple API servers
4. **GPU Memory Management**: Smart allocation and reuse via caching
5. **Request Batching**: Schedules requests efficiently
6. **Streaming Responses**: SSE for long-running operations

---

## Common Configuration Options (for gpt2)

For `vllm serve --model gpt2`:

```bash
# Default: Uses GPT-2 from Hugging Face Hub
# GPU Memory: Automatically allocated
# Batch Size: Auto-tuned
# Tensor Parallel: 1 (single GPU)
# Pipeline Parallel: 1 (no pipeline)
```

Can be extended with:
```bash
vllm serve --model gpt2 \
    --tensor-parallel-size 2 \           # Multi-GPU
    --gpu-memory-utilization 0.9 \       # More aggressive memory use
    --max-model-len 2048 \               # Context length
    --max-num-seqs 256 \                 # Concurrent requests
    --enable-prefix-caching              # KV cache optimization
```

---

This trace provides a comprehensive view of the vLLM architecture and how all components work together to serve the gpt2 model via an OpenAI-compatible API.
