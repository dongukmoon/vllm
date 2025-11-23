# vLLM Execution Flow: "vllm serve gpt-2" to Model Execution

## Overview
This document traces the complete execution flow from running `vllm serve gpt-2` through to actual model inference on the GPU.

---

## Phase 1: CLI Entry Point

### 1.1 Command Execution
```bash
$ vllm serve gpt-2
```

### 1.2 Entry Point: `main.py`
**File:** `vllm/entrypoints/cli/main.py`

```python
def main():
    print("[TRACE] (main.py) vLLM CLI main() called")
    # 1. Lazy load all CLI modules
    import vllm.entrypoints.cli.serve
    import vllm.entrypoints.cli.benchmark.main
    # ... other modules
    
    # 2. Setup environment
    cli_env_setup()  # Sets VLLM_USE_V1, logging, etc.
    print("[TRACE] (main.py) cli_env_setup() completed")
    
    # 3. Create argument parser with subcommands
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")
    
    # 4. Register "serve" subcommand
    cmds["serve"] = ServeSubcommand()
    
    # 5. Parse arguments
    args = parser.parse_args()  # Parses: ["serve", "gpt-2"]
    print(f"[TRACE] (main.py) Arguments parsed, subparser: {args.subparser}")
    # args.subparser = "serve"
    # args.model = "gpt-2"
    
    # 6. Dispatch to ServeSubcommand
    args.dispatch_function(args)
```

**Key Variables:**
- `args.model = "gpt-2"` (the positional model argument)
- `args.subparser = "serve"`
- `args.api_server_count = 1` (default)
- `args.headless = False` (default)

---

## Phase 2: Serve Subcommand Dispatch

### 2.1 ServeSubcommand Handler
**File:** `vllm/entrypoints/cli/serve.py`

```python
class ServeSubcommand(CLISubcommand):
    name = "serve"
    
    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        print(f"[TRACE] ServeSubcommand.cmd called")
        
        # Prioritize model_tag from positional arg
        if hasattr(args, 'model_tag') and args.model_tag is not None:
            print(f"[TRACE] Using model_tag from positional arg: {args.model_tag}")
            args.model = args.model_tag
        # args.model = "gpt-2"
        
        print(f"[TRACE] (serve.py) headless={args.headless}, api_server_count={args.api_server_count}")
        
        # Decision Tree:
        if args.headless or args.api_server_count < 1:
            run_headless(args)  # Multiprocess mode without API server
        else:
            if args.api_server_count > 1:
                run_multi_api_server(args)  # Multiple API servers
            else:
                # ➜ Single API server in this process (COMMON PATH)
                uvloop.run(run_server(args))
```

**Execution Path for standard `vllm serve gpt-2`:**
```
ServeSubcommand.cmd(args)
  └─> uvloop.run(run_server(args))
      [Starts async event loop]
```

---

## Phase 3: API Server Initialization

### 3.1 `run_server()` - FastAPI App Setup
**File:** `vllm/entrypoints/openai/api_server.py`

```python
async def run_server(args: Namespace) -> None:
    print("[TRACE] (api_server.py) run_server() called")
    
    # 1. Create engine client (either in-process or multiprocess)
    async with build_async_engine_client(args, client_config) as engine_client:
        print("[TRACE] (api_server.py) Engine client created")
        
        # 2. Setup FastAPI application
        app = FastAPI(lifespan=lifespan)
        print("[TRACE] (api_server.py) FastAPI app created")
        
        # 3. Configure engine and endpoints
        await setup_server(args, app, engine_client)
        print("[TRACE] (api_server.py) Server setup completed")
        
        # 4. Run the server
        await run_server_worker(app, args)
```

### 3.2 Engine Client Creation: V1 Path
**File:** `vllm/entrypoints/openai/api_server.py`

```python
async def build_async_engine_client(args: Namespace) -> AsyncIterator[EngineClient]:
    print("[TRACE] (api_server.py) build_async_engine_client() called")
    
    # 1. Create AsyncEngineArgs from CLI arguments
    engine_args = AsyncEngineArgs.from_cli_args(args)
    # engine_args.model_id = "gpt-2"
    # engine_args.tensor_parallel_size = 1
    # engine_args.gpu_memory_utilization = 0.9
    # ... (many other configuration options)
    
    # 2. Delegate to engine args version
    async with build_async_engine_client_from_engine_args(
            engine_args, 
            args.disable_frontend_multiprocessing,
            client_config) as engine:
        print("[TRACE] (api_server.py) Engine client ready")
        yield engine
```

### 3.3 V1 AsyncLLM Engine Creation
**File:** `vllm/entrypoints/openai/api_server.py`

```python
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
    client_config: Optional[dict[str, Any]] = None,
) -> AsyncIterator[EngineClient]:
    print("[TRACE] (api_server.py) build_async_engine_client_from_engine_args() called")
    
    # 1. Create VllmConfig
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    print("[TRACE] (api_server.py) VllmConfig created")
    # vllm_config contains:
    #   - model_config (gpt-2 architecture, weights location, etc.)
    #   - scheduler_config (batch size, max tokens, etc.)
    #   - device_config (CUDA/CPU)
    #   - parallel_config (TP, DP, etc.)
    #   - quantization_config (if applicable)
    
    # 2. V1 Engine: Create AsyncLLM
    if envs.VLLM_USE_V1:  # ✓ True by default
        print("[TRACE] (api_server.py) Using V1 AsyncLLM engine")
        from vllm.v1.engine.async_llm import AsyncLLM
        
        # Create the async engine
        async_llm = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            disable_log_requests=engine_args.disable_log_requests,
            disable_log_stats=engine_args.disable_log_stats,
            client_addresses=client_config,
            client_index=0,
        )
        print("[TRACE] (api_server.py) AsyncLLM created")
        
        try:
            yield async_llm
        finally:
            async_llm.shutdown()
```

---

## Phase 4: AsyncLLM Initialization

### 4.1 AsyncLLM.__init__()
**File:** `vllm/v1/engine/async_llm.py`

```python
class AsyncLLM(EngineClient):
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        ...
    ) -> None:
        print("[TRACE] (async_llm.py) AsyncLLM.__init__() called - initializing V1 async engine")
        
        # ===== 1. Configuration Validation =====
        if not envs.VLLM_USE_V1:
            raise ValueError("V1 must be enabled")
        print("[TRACE] (async_llm.py) V1 enabled, proceeding with initialization")
        
        # ===== 2. Model Configuration =====
        self.model_config = vllm_config.model_config
        # model_config.model = "gpt-2"
        # model_config.dtype = torch.float16 (or specified)
        # model_config.max_model_len = 2048
        # model_config.hf_config = <huggingface config>
        
        print(f"[TRACE] (async_llm.py) Model: {self.model_config.model}, "
              f"dtype: {self.model_config.dtype}, max_len: {self.model_config.max_model_len}")
        
        # ===== 3. Tokenizer Initialization =====
        if self.model_config.skip_tokenizer_init:
            print("[TRACE] (async_llm.py) Skipping tokenizer init")
            self.tokenizer = None
        else:
            print("[TRACE] (async_llm.py) Initializing tokenizer from configs")
            self.tokenizer = init_tokenizer_from_configs(
                model_config=vllm_config.model_config,
                scheduler_config=vllm_config.scheduler_config,
                lora_config=vllm_config.lora_config)
            # Loads GPT-2 tokenizer (1 file, 50257 vocab size)
            print("[TRACE] (async_llm.py) Tokenizer initialized")
        
        # ===== 4. Input Processor =====
        print("[TRACE] (async_llm.py) Creating Processor for input conversion")
        self.processor = Processor(
            vllm_config=vllm_config,
            tokenizer=self.tokenizer,
            mm_registry=MULTIMODAL_REGISTRY,
        )
        # processor: Converts text prompts → EngineCoreRequest objects
        print("[TRACE] (async_llm.py) Processor created successfully")
        
        # ===== 5. Output Processor =====
        print("[TRACE] (async_llm.py) Creating OutputProcessor for output processing")
        self.output_processor = OutputProcessor(self.tokenizer, log_stats=self.log_stats)
        # output_processor: Converts EngineCoreOutput → RequestOutput
        print("[TRACE] (async_llm.py) OutputProcessor created successfully")
        
        # ===== 6. EngineCore Client (Multiprocess Backend) =====
        print("[TRACE] (async_llm.py) Creating EngineCoreClient for multiprocess backend")
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,  # e.g., CudaExecutor
            log_stats=self.log_stats,
            client_addresses=client_addresses,  # Multi-engine coordination
            client_index=client_index,
        )
        # engine_core: IPC client to separate process running EngineCore
        print(f"[TRACE] (async_llm.py) EngineCoreClient created, "
              f"managed engines: {self.engine_core.engine_ranks_managed}")
        
        # ===== 7. Statistics Logging =====
        self.logger_manager: Optional[StatLoggerManager] = None
        if self.log_stats:
            print("[TRACE] (async_llm.py) Creating StatLoggerManager for stats logging")
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=stat_loggers,
            )
            self.logger_manager.log_engine_initialized()
            print("[TRACE] (async_llm.py) StatLoggerManager initialized")
        
        # ===== 8. Output Handler Task =====
        self.output_handler: Optional[asyncio.Task] = None
        try:
            asyncio.get_running_loop()
            print("[TRACE] (async_llm.py) Event loop detected, starting output handler")
            self._run_output_handler()
        except RuntimeError:
            print("[TRACE] (async_llm.py) No event loop yet, deferring output handler")
```

---

## Phase 5: EngineCore Backend Initialization

### 5.1 EngineCoreClient Creation
**File:** `vllm/v1/engine/core_client.py`

```python
class EngineCoreClient:
    @staticmethod
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        ...
    ) -> EngineCoreClient:
        print("[TRACE] Creating async multiprocess EngineCore client")
        
        # 1. Spawn EngineCore process(es)
        #    For each data parallel rank:
        core_procs = []
        for rank in range(parallel_config.data_parallel_size):
            proc = multiprocessing.Process(
                target=EngineCoreProc.run_engine_core,
                args=(vllm_config, executor_class, rank, ...)
            )
            proc.start()
            core_procs.append(proc)
            print(f"[TRACE] Started EngineCore process {rank}")
        
        # 2. Create IPC channels for communication
        #    Uses Unix sockets or ZMQ for inter-process communication
        
        # 3. Return client for async communication
        return EngineCoreClient(core_procs, ipc_channels, ...)
```

### 5.2 EngineCore Process (Separate Process)
**File:** `vllm/v1/engine/core.py`

```python
class EngineCoreProc:
    @staticmethod
    def run_engine_core(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        rank: int,
        ...
    ):
        print(f"[TRACE] EngineCore process {rank} started")
        
        # ===== 1. Initialize GPU =====
        torch.cuda.set_device(rank)
        print(f"[TRACE] GPU device {rank} set")
        
        # ===== 2. Create Model Executor =====
        print(f"[TRACE] Creating executor for rank {rank}")
        executor = executor_class.create_executor(
            vllm_config=vllm_config,
            rank=rank,
        )
        # executor = CudaExecutor (or other)
        # - Loads GPT-2 model weights from HuggingFace
        # - Compiles CUDA kernels if needed
        # - Allocates GPU memory for KV cache
        print(f"[TRACE] Executor created for rank {rank}")
        
        # ===== 3. Create Scheduler =====
        print(f"[TRACE] Creating scheduler for rank {rank}")
        scheduler = Scheduler(vllm_config.scheduler_config)
        # scheduler: Manages request batching, scheduling policy
        print(f"[TRACE] Scheduler created")
        
        # ===== 4. Create EngineCore =====
        engine_core = EngineCore(
            vllm_config=vllm_config,
            executor=executor,
            scheduler=scheduler,
            rank=rank,
        )
        print(f"[TRACE] EngineCore initialized for rank {rank}")
        
        # ===== 5. Main Loop =====
        engine_core.run_loop()
        # Waits for requests via IPC
        # Executes batched requests
        # Returns outputs via IPC
```

---

## Phase 6: First Request - Chat Completion API Call

### 6.1 Client Makes Request
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-2",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7
  }'
```

### 6.2 FastAPI Endpoint Handler
**File:** `vllm/entrypoints/openai/api_server.py`

```python
@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request = None,
) -> ChatCompletionResponse | StreamingResponse:
    print(f"[TRACE] (api_server.py) POST /v1/chat/completions - model: {request.model}")
    
    # 1. Get the chat serving handler
    serving_chat: OpenAIServingChat = app.state.serving_chat
    
    # 2. Create chat completion (generates output or stream)
    result = await serving_chat.create_chat_completion(
        request=request,
        raw_request=raw_request,
    )
    print(f"[TRACE] (api_server.py) Chat completion request processed")
    
    return result
```

### 6.3 OpenAIServingChat Handler
**File:** `vllm/entrypoints/openai/serving_chat.py`

```python
class OpenAIServingChat(OpenAIServing):
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[RequestLogger] = None,
    ) -> Union[ChatCompletionResponse, StreamingResponse]:
        print("[TRACE] (serving_chat.py) OpenAIServingChat.create_chat_completion() called")
        
        # ===== 1. Validate Request =====
        print(f"[TRACE] (serving_chat.py) Model: {request.model}, "
              f"Messages: {len(request.messages)}")
        
        # ===== 2. Get Tokenizer =====
        print("[TRACE] (serving_chat.py) Retrieving tokenizer")
        tokenizer = await self.engine_client.get_tokenizer(request.lora_request)
        
        # ===== 3. Apply Chat Template =====
        print("[TRACE] (serving_chat.py) Applying chat template")
        prompt = self.apply_chat_template(
            request.messages,
            tokenizer=tokenizer,
            chat_template=self.chat_template,
        )
        # For GPT-2, simple concatenation of messages
        # prompt = "Hello"
        
        # ===== 4. Determine Response Mode =====
        use_beam_search = request.best_of is not None and request.best_of > 1
        use_streaming = request.stream
        print(f"[TRACE] (serving_chat.py) Streaming: {use_streaming}, Beam search: {use_beam_search}")
        
        # ===== 5. Create Sampling Params =====
        print("[TRACE] (serving_chat.py) Creating sampling parameters")
        sampling_params = SamplingParams(
            n=request.n,  # Number of completions (default 1)
            best_of=request.best_of,  # For beam search
            temperature=request.temperature,  # 0.7
            top_p=request.top_p,  # 1.0
            top_k=request.top_k,  # -1
            frequency_penalty=request.frequency_penalty,  # 0.0
            presence_penalty=request.presence_penalty,  # 0.0
            length_penalty=request.length_penalty,  # 1.0
            max_tokens=request.max_tokens,  # Entire input
            min_tokens=request.min_tokens,  # 0
            ...
        )
        print(f"[TRACE] (serving_chat.py) SamplingParams created: "
              f"temperature={sampling_params.temperature}, "
              f"max_tokens={sampling_params.max_tokens}")
        
        # ===== 6. Process Request =====
        if use_streaming:
            print("[TRACE] (serving_chat.py) Using streaming response")
            return StreamingResponse(
                self.chat_completion_stream(
                    request, sampling_params, prompt, ...
                ),
                media_type="text/event-stream",
            )
        else:
            print("[TRACE] (serving_chat.py) Using non-streaming response")
            result = await self.chat_completion_full_generator(
                request, sampling_params, prompt, ...
            )
            return result
```

---

## Phase 7: Request Processing in Engine

### 7.1 Generate Function (Non-streaming)
**File:** `vllm/v1/engine/async_llm.py`

```python
async def generate(
    self,
    prompt: PromptType,  # "Hello"
    sampling_params: SamplingParams,
    request_id: str,  # UUID
    ...
) -> AsyncGenerator[RequestOutput, None]:
    print(f"[TRACE] (async_llm.py) generate() called - request_id: {request_id}, "
          f"temperature: {sampling_params.temperature}, top_p: {sampling_params.top_p}")
    
    try:
        # ===== 1. Start Output Handler =====
        print(f"[TRACE] (async_llm.py) Starting output handler")
        self._run_output_handler()  # Background task to collect outputs
        
        # ===== 2. Add Request to Engine =====
        print(f"[TRACE] (async_llm.py) Adding request to engine")
        q = await self.add_request(
            request_id,
            prompt,
            sampling_params,
            lora_request=lora_request,
            ...
        )
        print(f"[TRACE] (async_llm.py) Request added, starting output streaming")
        
        # ===== 3. Stream Outputs =====
        finished = False
        output_count = 0
        while not finished:
            out = q.get_nowait() or await q.get()
            output_count += 1
            print(f"[TRACE] (async_llm.py) Yielding output #{output_count}, "
                  f"finished: {out.finished}")
            finished = out.finished
            yield out
            # out = RequestOutput(
            #     request_id=request_id,
            #     prompt=prompt,
            #     outputs=[CompletionOutput(text=" world", ...)]
            # )
        
        print(f"[TRACE] (async_llm.py) Generation complete - {output_count} outputs")
```

### 7.2 Add Request to Engine
**File:** `vllm/v1/engine/async_llm.py`

```python
async def add_request(
    self,
    request_id: str,
    prompt: PromptType,  # "Hello"
    params: SamplingParams,
    ...
) -> RequestOutputCollector:
    print(f"[TRACE] (async_llm.py) add_request() - request_id: {request_id}")
    
    # ===== 1. Check Engine Health =====
    if self.errored:
        raise EngineDeadError()
    
    # ===== 2. Create Output Collector =====
    queue = RequestOutputCollector(output_kind=params.output_kind)
    print(f"[TRACE] (async_llm.py) Created RequestOutputCollector")
    
    # ===== 3. Process Inputs (Tokenization) =====
    print(f"[TRACE] (async_llm.py) Processing inputs for request")
    prompt_str, request = self.processor.process_inputs(
        request_id, prompt, params, arrival_time=None, ...
    )
    # prompt_str = "Hello"
    # request = EngineCoreRequest(
    #     request_id=request_id,
    #     prompt=prompt_str,
    #     prompt_token_ids=[15496],  # GPT-2 tokenization
    #     sampling_params=params,
    # )
    print(f"[TRACE] (async_llm.py) Processed inputs")
    
    # ===== 4. Add to Internal Processor =====
    print(f"[TRACE] (async_llm.py) Adding to OutputProcessor")
    await self._add_request(request, prompt_str, None, 0, queue)
    
    return queue

async def _add_request(
    self,
    request: EngineCoreRequest,
    prompt: Optional[str],
    ...
) -> None:
    print(f"[TRACE] (async_llm.py) _add_request() - request_id: {request.request_id}")
    
    # ===== 1. Register in OutputProcessor =====
    # Tracks outputs for this request
    print(f"[TRACE] (async_llm.py) Adding to OutputProcessor")
    self.output_processor.add_request(request, prompt, parent_req, index, queue)
    
    # ===== 2. Send to EngineCore (IPC) =====
    # Sends to the separate EngineCore process
    print(f"[TRACE] (async_llm.py) Adding to EngineCore via IPC")
    await self.engine_core.add_request_async(request)
    print(f"[TRACE] (async_llm.py) Request sent to EngineCore")
```

---

## Phase 8: EngineCore Execution (Separate Process)

### 8.1 EngineCore Main Loop
**File:** `vllm/v1/engine/core.py`

```python
class EngineCore:
    async def run_loop(self):
        print("[TRACE] EngineCore main loop started")
        
        while True:
            # ===== 1. Wait for Requests =====
            requests = await self.ipc_server.get_requests()
            # Receives EngineCoreRequest from add_request_async()
            
            if requests:
                print(f"[TRACE] EngineCore received {len(requests)} requests")
                
                # ===== 2. Add to Scheduler =====
                for request in requests:
                    print(f"[TRACE] EngineCore: Scheduling request {request.request_id}")
                    self.scheduler.add_request(request)
                
                # ===== 3. Execute Scheduler Step =====
                print("[TRACE] EngineCore: Running scheduler step")
                scheduler_output = self.scheduler.schedule()
                # scheduler_output contains:
                # - requests_to_execute: list of requests to run
                # - requests_to_abort: list of requests to abort
                # - finished_requests: list of completed requests
                
                # ===== 4. Execute Model =====
                print("[TRACE] EngineCore: Executing model")
                model_output = self.executor.execute_model(
                    requests=scheduler_output.requests_to_execute,
                    shared_data={...},
                )
                # model_output = {
                #     "request_1": [token_ids_batch],
                #     ...
                # }
                print("[TRACE] EngineCore: Model execution completed")
                
                # ===== 5. Process Output =====
                print("[TRACE] EngineCore: Processing outputs")
                engine_core_output = self._process_outputs(
                    scheduler_output,
                    model_output,
                )
                
                # ===== 6. Send Results Back to API Server (IPC) =====
                print("[TRACE] EngineCore: Sending outputs back to API server")
                await self.ipc_server.send_outputs(engine_core_output)
```

### 8.2 Model Execution (GPU)
**File:** `vllm/v1/executor/cuda_executor.py`

```python
class CudaExecutor(Executor):
    def execute_model(
        self,
        requests: List[EngineCoreRequest],
        shared_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        print(f"[TRACE] CudaExecutor.execute_model() called with {len(requests)} requests")
        
        # ===== 1. Prepare Batch =====
        print("[TRACE] Preparing attention metadata and token IDs")
        attention_metadata = prepare_attention_metadata(requests)
        input_tokens = prepare_input_tokens(requests)
        # input_tokens = [15496]  # "Hello" tokenized for GPT-2
        
        # ===== 2. Load/Unload Weights (if needed) =====
        print("[TRACE] Verifying model weights are loaded")
        
        # ===== 3. Forward Pass =====
        print("[TRACE] Running forward pass on GPU")
        # ===== a. Token Embedding =====
        # input_ids [15496] -> embeddings [768] (GPT-2 embed size)
        print("[TRACE] Computing token embeddings")
        embeddings = self.model.gpt2.wte(input_tokens)  # Embedding lookup
        
        # ===== b. Apply Position Embeddings =====
        print("[TRACE] Adding positional embeddings")
        embeddings += self.model.gpt2.wpe(positions)  # Add position info
        
        # ===== c. Transformer Blocks =====
        print("[TRACE] Running 12 transformer blocks")
        for i, block in enumerate(self.model.gpt2.h):  # 12 layers
            print(f"[TRACE] Transformer block {i}: self-attention + FFN")
            # Self-attention over [15496] token
            # Uses cached KV from previous tokens
            # Feed-forward network
            hidden_states = block(embeddings, attention_metadata)
        
        # ===== d. Output Layer Norm =====
        print("[TRACE] Applying layer normalization")
        hidden_states = self.model.gpt2.ln_f(hidden_states)
        
        # ===== e. LM Head =====
        print("[TRACE] Computing logits via LM head")
        logits = self.model.lm_head(hidden_states)
        # logits shape: [1, 50257]  # All possible next tokens
        
        # ===== 4. Sampling =====
        print("[TRACE] Sampling next tokens")
        # Apply temperature scaling
        scaled_logits = logits / sampling_params.temperature  # 0.7
        
        # Apply top-p filtering
        print("[TRACE] Applying top-p filtering")
        filtered_logits = top_p_sampling(scaled_logits, top_p=1.0)
        
        # Sample from distribution
        print("[TRACE] Sampling from token distribution")
        next_tokens = torch.multinomial(
            F.softmax(filtered_logits, dim=-1),
            num_samples=1
        )
        # next_tokens = [5773]  # " world"
        
        print("[TRACE] Tokens sampled, adding to sequence")
        
        # ===== 5. Update KV Cache =====
        print("[TRACE] Updating KV cache")
        # Store computed attention keys/values for next forward pass
        
        # ===== 6. Return Results =====
        print("[TRACE] Executor returning results")
        return {
            "request_1": {
                "next_tokens": [5773],  # " world"
                "logits": logits,
                "kv_cache": {...},  # Updated cache
            }
        }
```

**GPU Operations Summary:**
```
Token ID 15496 → GPT-2 Embedding (768 dims)
              ↓
Add Position Embedding (pos 0)
              ↓
Pass through 12 Transformer Blocks:
  - Self-Attention (768→768)
  - Feed-Forward (768→3072→768)
  - Layer Norm
              ↓
Final Layer Norm (768→768)
              ↓
LM Head (768→50257 vocab)
              ↓
Apply Temperature (0.7)
              ↓
Top-P Sampling (keep top 100% probability)
              ↓
Sample next token: 5773 (" world")
```

---

## Phase 9: Output Processing & Response

### 9.1 Output Handler (Background Task)
**File:** `vllm/v1/engine/async_llm.py`

```python
def _run_output_handler(self):
    print("[TRACE] Creating output handler task")
    
    async def output_handler():
        print("[TRACE] Output handler task started")
        iteration_num = 0
        
        while True:
            iteration_num += 1
            
            # ===== 1. Pull Outputs from EngineCore =====
            print(f"[TRACE] Output handler iteration #{iteration_num}")
            outputs = await self.engine_core.get_output_async()
            num_outputs = len(outputs.outputs)
            print(f"[TRACE] Received {num_outputs} outputs from EngineCore")
            
            # ===== 2. Process EngineCoreOutputs =====
            print(f"[TRACE] Processing {num_outputs} outputs")
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                outputs.timestamp,
            )
            # Converts raw model outputs → RequestOutput objects
            
            # ===== 3. Update Request Queues =====
            # RequestOutput pushed to each request's queue
            # Wakes up waiting coroutine in generate()
            
            # ===== 4. Logging =====
            if self.logger_manager:
                self.logger_manager.record(...)
    
    self.output_handler = asyncio.create_task(output_handler())
```

### 9.2 Response Construction
**File:** `vllm/entrypoints/openai/serving_chat.py`

```python
# Back in API server process, generate() receives outputs

async def generate(prompt, sampling_params, request_id):
    # Yields RequestOutput objects from EngineCore
    
    # First output (token generation):
    output = RequestOutput(
        request_id=request_id,
        prompt="Hello",
        prompt_token_ids=[15496],
        outputs=[
            CompletionOutput(
                index=0,
                text=" world",
                token_ids=[5773],
                cumulative_logprob=-0.231,
                finish_reason=None,
            )
        ],
        finished=False,
    )
    yield output
    
    # Continue sampling more tokens until max_tokens reached
    # ...
    
    # Final output:
    output = RequestOutput(
        request_id=request_id,
        prompt="Hello",
        prompt_token_ids=[15496],
        outputs=[
            CompletionOutput(
                index=0,
                text=" world! How are you?",
                token_ids=[5773, 0, 1264, 389, 345, 30],
                cumulative_logprob=-1.234,
                finish_reason="length",  # or "stop_string"
            )
        ],
        finished=True,
    )
    yield output
```

### 9.3 API Response
**File:** `vllm/entrypoints/openai/api_server.py`

```python
# Convert RequestOutput to OpenAI-compatible ChatCompletionResponse

response = ChatCompletionResponse(
    id="chatcmpl-uuid",
    object="chat.completion",
    created=int(time.time()),
    model="gpt-2",
    choices=[
        ChatCompletionChoice(
            index=0,
            message=ChatMessage(
                role="assistant",
                content=" world! How are you?"
            ),
            finish_reason="length",
            logprobs=None,
        )
    ],
    usage=CompletionUsage(
        prompt_tokens=1,  # "Hello"
        completion_tokens=6,  # " world! How are you?"
        total_tokens=7,
    ),
)

# Return as JSON
return response
```

---

## Complete Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: CLI Entry                                              │
│                                                                 │
│  $ vllm serve gpt-2                                             │
│         ↓                                                       │
│  main.py:main() → cli_env_setup()                               │
│         ↓                                                       │
│  Parse args: model="gpt-2", api_server_count=1                 │
│         ↓                                                       │
│  ServeSubcommand.cmd(args)                                     │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: API Server Initialization                              │
│                                                                 │
│  uvloop.run(run_server(args))                                  │
│         ↓                                                       │
│  build_async_engine_client()                                   │
│         ↓                                                       │
│  AsyncEngineArgs.from_cli_args() → VllmConfig                  │
│         ↓                                                       │
│  AsyncLLM.from_vllm_config()                                   │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: Engine Core Initialization                             │
│                                                                 │
│  AsyncLLM.__init__()                                           │
│    ├─ Init Tokenizer (GPT-2)                                   │
│    ├─ Create Processor (input tokenization)                    │
│    ├─ Create OutputProcessor (output detokenization)           │
│    └─ Create EngineCoreClient                                  │
│           ↓                                                     │
│  EngineCoreClient.make_async_mp_client()                       │
│           ↓                                                     │
│  spawn EngineCore process(es)                                  │
│           ↓                                                     │
│  EngineCore.__init__()                                         │
│    ├─ Initialize GPU                                           │
│    ├─ Load GPT-2 model weights                                 │
│    ├─ Create Scheduler                                         │
│    └─ Create Executor                                          │
│           ↓                                                     │
│  EngineCore.run_loop()  [Waits for requests]                   │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: HTTP Request                                           │
│                                                                 │
│  User sends: POST /v1/chat/completions                         │
│    prompt: "Hello"                                             │
│    temperature: 0.7                                            │
│    max_tokens: 100                                             │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: Request Processing (API Server Process)               │
│                                                                 │
│  FastAPI endpoint handler                                      │
│         ↓                                                       │
│  OpenAIServingChat.create_chat_completion()                    │
│    ├─ Get Tokenizer                                            │
│    ├─ Apply Chat Template                                      │
│    ├─ Create SamplingParams                                    │
│    └─ Call engine_client.generate()                            │
│           ↓                                                     │
│  AsyncLLM.generate(prompt, sampling_params)                    │
│    ├─ Start output handler                                     │
│    └─ await add_request() [IPC to EngineCore]                  │
│           ↓                                                     │
│  AsyncLLM.add_request()                                        │
│    ├─ Processor.process_inputs()  [Tokenize "Hello"]           │
│    └─ engine_core.add_request_async()  [Send to EngineCore]    │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 6: GPU Model Execution (EngineCore Process)              │
│                                                                 │
│  EngineCore receives request via IPC                           │
│         ↓                                                       │
│  scheduler.add_request()                                       │
│         ↓                                                       │
│  scheduler.schedule()  [Decide batching]                       │
│         ↓                                                       │
│  executor.execute_model()  [GPU execution]                     │
│    ├─ Token embedding: [15496] → [768]                         │
│    ├─ 12 Transformer blocks                                    │
│    │  ├─ Self-attention over context                           │
│    │  ├─ Feed-forward network                                  │
│    │  └─ Layer normalization                                   │
│    ├─ Output layer norm                                        │
│    ├─ LM head: [768] → [50257]                                 │
│    ├─ Temperature scaling: 0.7                                 │
│    ├─ Top-p filtering: 1.0 (all tokens)                        │
│    ├─ Sampling: → token_id 5773 (" world")                     │
│    └─ Update KV cache                                          │
│           ↓                                                     │
│  Return model output (next tokens, logits, etc.)               │
│         ↓                                                       │
│  Send outputs back to API server via IPC                       │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 7: Output Processing (API Server Process)                │
│                                                                 │
│  output_handler() receives EngineCoreOutput                    │
│         ↓                                                       │
│  OutputProcessor.process_outputs()                             │
│    └─ Detokenize tokens → text (" world")                      │
│           ↓                                                     │
│  Push RequestOutput to request queue                           │
│         ↓                                                       │
│  AsyncLLM.generate() yields RequestOutput                      │
│         ↓                                                       │
│  OpenAIServingChat receives outputs                            │
│    └─ Convert to ChatCompletionResponse                        │
│           ↓                                                     │
│  Return JSON response to client                                │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 8: HTTP Response                                          │
│                                                                 │
│  200 OK                                                        │
│  {                                                             │
│    "id": "chatcmpl-...",                                       │
│    "object": "chat.completion",                                │
│    "model": "gpt-2",                                           │
│    "choices": [{                                               │
│      "message": {                                              │
│        "role": "assistant",                                    │
│        "content": " world! How are you?"                        │
│      },                                                        │
│      "finish_reason": "length"                                 │
│    }],                                                         │
│    "usage": {                                                  │
│      "prompt_tokens": 1,                                       │
│      "completion_tokens": 6,                                   │
│      "total_tokens": 7                                         │
│    }                                                           │
│  }                                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Architecture Components

### 1. **Tokenizer** (GPT-2)
- **Location:** `vllm/v1/engine/async_llm.py:100`
- **Purpose:** Convert text ↔ token IDs
- **GPT-2:** Uses BPE tokenizer with 50,257 vocabulary
- **Example:** "Hello" → [15496]

### 2. **Processor** (Input Processing)
- **Location:** `vllm/v1/engine/processor.py`
- **Purpose:** Text → EngineCoreRequest
- **Handles:** Tokenization, padding, attention mask

### 3. **Scheduler** (Request Batching)
- **Location:** `vllm/v1/engine/scheduler.py`
- **Purpose:** Decides which requests to execute together
- **Policy:** First-come-first-served, preemption, etc.

### 4. **Executor** (GPU Execution)
- **Location:** `vllm/v1/executor/cuda_executor.py`
- **Purpose:** Run model forward pass
- **Kernel:** CUDA kernels optimized for attention, quantization, etc.

### 5. **OutputProcessor** (Output Processing)
- **Location:** `vllm/v1/engine/output_processor.py`
- **Purpose:** EngineCoreOutput → RequestOutput
- **Handles:** Detokenization, token counting, finish reasons

---

## Trace Output Example

When running with TRACE enabled, you'd see:

```
[TRACE] (main.py) vLLM CLI main() called
[TRACE] Running cli_env_setup()
[TRACE] Creating argument parser
[TRACE] Arguments parsed, subparser: serve
[TRACE] ServeSubcommand.cmd called
[TRACE] Running single API server
[TRACE] (api_server.py) run_server() called
[TRACE] (api_server.py) build_async_engine_client() called
[TRACE] (async_llm.py) AsyncLLM.__init__() called
[TRACE] (async_llm.py) Initialized tokenizer from configs
[TRACE] (async_llm.py) Creating Processor for input conversion
[TRACE] (async_llm.py) Creating OutputProcessor for output processing
[TRACE] (async_llm.py) Creating EngineCoreClient for multiprocess backend
[TRACE] (async_llm.py) Event loop detected, starting output handler
[...server ready...]
[TRACE] (api_server.py) POST /v1/chat/completions
[TRACE] (serving_chat.py) create_chat_completion() called
[TRACE] (async_llm.py) generate() called - request_id: abc123
[TRACE] (async_llm.py) add_request() called - request_id: abc123
[TRACE] (async_llm.py) Processing inputs for request_id: abc123
[TRACE] (async_llm.py) Adding request to EngineCore via async
[...GPU execution...]
[TRACE] (async_llm.py) Output handler iteration #1 - pulling outputs
[TRACE] (async_llm.py) Received 1 outputs from EngineCore
[TRACE] (async_llm.py) Yielding output #1 - request_id: abc123, finished: False
[TRACE] (async_llm.py) Yielding output #2 - request_id: abc123, finished: True
[TRACE] (async_llm.py) Generation complete - 2 outputs
```

---

## Performance Hotspots

### 1. **Model Loading** (PHASE 3)
- Downloads GPT-2 weights (~500MB)
- Loads into GPU memory
- **Optimization:** Use weight sharing, lazy loading

### 2. **First Forward Pass** (PHASE 6)
- Slower due to GPU kernel launches
- KV cache not populated
- **Optimization:** Prefill optimization, batching

### 3. **Token Generation Loop** (PHASE 6)
- Repeated forward passes (autoregressive)
- Each token adds latency
- **Optimization:** Speculative decoding, parallel sampling

### 4. **IPC Overhead** (PHASE 5 & 7)
- Network serialization/deserialization
- Unix socket communication
- **Optimization:** Reduce copy, use shared memory

---

## Summary

**Full execution from CLI to GPU takes:**
- **Cold Start:** ~5-10 seconds (model loading)
- **Warm Start:** ~50-200ms (inference + IPC)
  - Input processing: ~1-5ms
  - GPU execution: ~30-100ms (depends on batch size, sequence length)
  - Output processing: ~1-5ms
  - IPC overhead: ~10-50ms

