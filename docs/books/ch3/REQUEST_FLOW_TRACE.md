# vLLM Request Flow: Complete Trace of cURL Completion Request

## Overview
This document traces the complete execution flow of a single HTTP completion request through vLLM, from the moment the cURL client sends the HTTP request until the server returns the response.

**Client Command**:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "Today'"'"'s weather is",
    "max_tokens": 50,
    "temperature": 0.5
  }'
```

---

## Stage 1: HTTP Request Transmission (Client → Server)

### 1.1 cURL Creates HTTP Request

**Client Side** (your terminal):
```
┌─────────────────────────────────────────┐
│ cURL HTTP Request Builder               │
├─────────────────────────────────────────┤
│ Method: POST                            │
│ URL: http://localhost:8000/v1/completions
│ Headers:                                │
│   Content-Type: application/json        │
│   Host: localhost:8000                  │
│   User-Agent: curl/7.x.x                │
│   Accept: */*                           │
│   Content-Length: 118                   │
│                                         │
│ Body (JSON):                            │
│ {                                       │
│   "model": "gpt2",                      │
│   "prompt": "Today's weather is",       │
│   "max_tokens": 50,                     │
│   "temperature": 0.5                    │
│ }                                       │
└─────────────────────────────────────────┘
```

### 1.2 TCP Connection

```
cURL (client)                          vLLM Server
     │                                      │
     │ TCP 3-Way Handshake                 │
     ├─────── SYN ────────────────────────→
     │                                      │
     │                        ← SYN-ACK ────┤
     │                                      │
     ├─────── ACK ────────────────────────→
     │ Connection Established ✓             │
     │                                      │
     ├─────── HTTP POST ───────────────────→
     │ (with request body)                  │
     │                                      │
```

### 1.3 Raw HTTP Request Sent

```
POST /v1/completions HTTP/1.1
Host: localhost:8000
Content-Type: application/json
Content-Length: 118
User-Agent: curl/7.x.x
Accept: */*

{"model":"gpt2","prompt":"Today's weather is","max_tokens":50,"temperature":0.5}
```

---

## Stage 2: Server Reception & Routing (Uvicorn/FastAPI)

### 2.1 Uvicorn Receives HTTP Request

**File**: `vllm/entrypoints/launcher.py` + Uvicorn (external library)

```python
# Uvicorn's ASGI server receives the raw HTTP bytes
# and parses them into a Scope dict:

scope = {
    "type": "http",
    "method": "POST",
    "path": "/v1/completions",
    "query_string": b"",
    "headers": [
        (b"content-type", b"application/json"),
        (b"content-length", b"118"),
        (b"host", b"localhost:8000"),
        # ... other headers ...
    ],
    "server": ("0.0.0.0", 8000),
    "client": ("127.0.0.1", <random_port>),
}
```

**Timeline**: < 1ms from cURL sending to Uvicorn receiving

### 2.2 FastAPI Router Matching

**File**: `vllm/entrypoints/openai/api_server.py:router`

```python
# FastAPI matches the path and HTTP method
# The router looks for: POST /v1/completions

# Router has registered:
@router.post("/v1/completions", ...)
async def create_completion(request: CompletionRequest, raw_request: Request):
    # ← This function will be called
    pass
```

**Matching Process**:
```
FastAPI Router
    ├─ Check HTTP method: POST ✓
    ├─ Check path: /v1/completions ✓
    ├─ Found matching route!
    └─ Route handler: create_completion()
```

### 2.3 Middleware Chain (Before Handler)

```python
# Request flows through middleware stack:

MIDDLEWARE CHAIN:
    ↓
    ├─ [1] Scaling Middleware
    │   └─ Check if server is currently scaling
    │       → Not scaling, continue ✓
    │
    ├─ [2] XRequestIdMiddleware (if enabled)
    │   └─ Generate X-Request-Id header
    │       → Generate UUID if not present ✓
    │
    ├─ [3] AuthenticationMiddleware (if API key set)
    │   └─ Check "Authorization: Bearer sk-xxx"
    │       → Path /v1/completions needs auth
    │       → Verify token ✓
    │
    ├─ [4] CORSMiddleware
    │   └─ Check Origin and methods
    │       → CORS headers added to response ✓
    │
    └─ [5] Request reaches handler
```

**Timeline**: < 2ms (middleware processing)

---

## Stage 3: HTTP Body Parsing

### 3.1 JSON Deserialization

**File**: `vllm/entrypoints/openai/api_server.py:create_completion()`

```python
@router.post("/v1/completions",
             dependencies=[Depends(validate_json_request)],
             ...)
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    # FastAPI automatically deserializes JSON → CompletionRequest
    
    # The raw HTTP body is parsed:
    # {"model":"gpt2","prompt":"Today's weather is","max_tokens":50,"temperature":0.5}
    #
    # Into CompletionRequest object:
    # CompletionRequest(
    #     model="gpt2",
    #     prompt="Today's weather is",
    #     max_tokens=50,
    #     temperature=0.5,
    #     top_p=1.0,  # default
    #     frequency_penalty=0.0,  # default
    #     # ... other defaults ...
    # )
    
    print(f"Model: {request.model}")  # "gpt2"
    print(f"Prompt: {request.prompt}")  # "Today's weather is"
    print(f"Max tokens: {request.max_tokens}")  # 50
    print(f"Temperature: {request.temperature}")  # 0.5
```

**Pydantic Validation**: FastAPI uses Pydantic to validate:
- ✅ "model" is a string (required)
- ✅ "prompt" is a string (required)
- ✅ "max_tokens" is an integer > 0 (optional, default 16)
- ✅ "temperature" is a float 0.0-2.0 (optional, default 1.0)

**If validation fails**:
```python
# FastAPI catches ValidationError
# Returns HTTP 422 Unprocessable Entity
raise HTTPException(
    status_code=422,
    detail={
        "error": "Invalid input",
        "messages": ["temperature must be between 0.0 and 2.0"]
    }
)
```

**Timeline**: < 5ms (JSON parsing + validation)

---

## Stage 4: Request Handler Execution

### 4.1 Entry Point: `create_completion()`

**File**: `vllm/entrypoints/openai/api_server.py`

```python
@router.post("/v1/completions", ...)
@with_cancellation        # Decorator: handle request cancellation
@load_aware_call          # Decorator: load-aware scheduling
async def create_completion(request: CompletionRequest, raw_request: Request):
    """
    Main handler for POST /v1/completions
    """
    
    # [Step 1] Get the serving handler from app state
    handler = completion(raw_request)  # → OpenAIServingCompletion instance
    
    # [Step 2] Verify handler exists (model supports text completion)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Completions API"
        )
    
    # [Step 3] Call handler to generate completions
    try:
        generator = await handler.create_completion(request, raw_request)
        # generator is either:
        # - CompletionResponse object (non-streaming)
        # - AsyncGenerator[CompletionStreamResponse] (streaming)
    except OverflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # [Step 4] Format response based on type
    if isinstance(generator, ErrorResponse):
        # Error occurred
        return JSONResponse(
            content=generator.model_dump(),
            status_code=generator.error.code
        )
    elif isinstance(generator, CompletionResponse):
        # Non-streaming: return complete response
        return JSONResponse(content=generator.model_dump())
    else:
        # Streaming: generator yields chunks
        return StreamingResponse(
            content=generator,
            media_type="text/event-stream"
        )
```

**For our request**:
- `request.stream` is not set or False → Non-streaming response
- `generator` will be a `CompletionResponse` object

**Timeline**: < 1ms (routing to handler)

---

## Stage 5: Request Handler (OpenAIServingCompletion)

### 5.1 Handler Processing

**File**: `vllm/entrypoints/openai/serving_completion.py`

```python
class OpenAIServingCompletion:
    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        """Handle completion request"""
        
        # [Step 1] Get model configuration
        model_config = self.model_config
        # → model_id: "gpt2"
        # → vocab_size: 50257
        # → max_position_embeddings: 1024
        
        # [Step 2] Validate model name
        if request.model != model_config.model or request.model not in self.served_model_names:
            return self.create_error_response(
                message=f"Model {request.model} not found"
            )
        # ✓ "gpt2" matches configured model
        
        # [Step 3] Get or format prompt(s)
        prompts = request.prompt  # "Today's weather is"
        if isinstance(prompts, str):
            prompts = [prompts]  # Convert to list: ["Today's weather is"]
        
        # [Step 4] Log request (if enabled)
        if self.request_logger:
            self.request_logger.log_request_start(
                request_id=request_id,
                prompt=prompts[0],
                model=request.model,
                sampling_params={
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    # ...
                }
            )
        
        # [Step 5] Prepare sampling parameters
        sampling_params = SamplingParams(
            n=request.n or 1,                           # 1
            best_of=request.best_of or 1,               # 1
            presence_penalty=request.presence_penalty or 0.0,  # 0.0
            frequency_penalty=request.frequency_penalty or 0.0,  # 0.0
            repetition_penalty=request.repetition_penalty or 1.0,  # 1.0
            temperature=request.temperature or 1.0,     # 0.5
            top_p=request.top_p or 1.0,                 # 1.0
            top_k=request.top_k or -1,                  # -1 (disabled)
            min_p=request.min_p or 0.0,                 # 0.0
            use_beam_search=request.use_beam_search or False,  # False
            length_penalty=request.length_penalty or 1.0,  # 1.0
            early_stopping=request.early_stopping or False,  # False
            stop=request.stop or [],                    # []
            stop_token_ids=request.stop_token_ids or [],  # []
            include_stop_str_in_output=True,
            skip_special_tokens=request.skip_special_tokens or True,
        )
        
        # [Step 6] Preprocess text/tokenize
        # → Uses the tokenizer to convert prompt to token IDs
        tokenized_results = self.tokenizer.encode_prompt(
            prompt=prompts[0],  # "Today's weather is"
            model_config=model_config,
            chat_template=self.chat_template,
        )
        
        # Result:
        # {
        #   "prompt_tokens": [2421, 594, 6193, 318],  # Token IDs for prompt
        #   "prompt": "Today's weather is",
        # }
        
        # [Step 7] Call engine to generate completions
        # This is the MAIN inference call
        outputs = await self.engine_client.generate(
            prompts=prompts,  # ["Today's weather is"]
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=None,
        )
        
        # outputs is a list of RequestOutput objects:
        # RequestOutput(
        #     request_id="...",
        #     prompt="Today's weather is",
        #     prompt_token_ids=[2421, 594, 6193, 318],
        #     outputs=[
        #         CompletionOutput(
        #             index=0,
        #             text="fine and sunny. I'm glad I took my umbrella just in case!",
        #             token_ids=[408, 290, 14503, 13, 40, 1101, 6568, 40, ...],
        #             cumulative_logprob=-5.234,
        #             logprobs=None,
        #             finish_reason="length",
        #         )
        #     ],
        #     finished=True,
        # )
        
        # [Step 8] Format outputs into OpenAI API response
        response = CompletionResponse(
            id=request_id,                           # "cmpl-abc123..."
            object="text_completion",
            created=int(time.time()),                # Unix timestamp
            model=request.model,                     # "gpt2"
            choices=[
                CompletionChoice(
                    index=0,
                    text=output.outputs[0].text,     # Generated text
                    logprobs=None,
                    finish_reason=output.outputs[0].finish_reason,  # "length"
                )
                for output in outputs
            ],
            usage=CompletionUsage(
                prompt_tokens=len(outputs[0].prompt_token_ids),     # 4
                completion_tokens=len(outputs[0].outputs[0].token_ids),  # ~50
                total_tokens=prompt_tokens + completion_tokens,     # ~54
            ),
        )
        
        # [Step 9] Log completion
        if self.request_logger:
            self.request_logger.log_request_end(
                request_id=request_id,
                prompt=prompts[0],
                prompt_token_ids=outputs[0].prompt_token_ids,
                outputs=outputs[0].outputs,
                total_time=elapsed_time,
            )
        
        # [Step 10] Return response
        return response
```

**Key Variable States at Step 7**:
```python
prompts = ["Today's weather is"]
sampling_params = SamplingParams(
    temperature=0.5,
    max_tokens=50,
    top_p=1.0,
    # ...
)
model_config = ModelConfig(
    model="gpt2",
    vocab_size=50257,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    # ...
)
```

**Timeline**: ~5ms (validation, preprocessing, parameter prep)

---

## Stage 6: Engine Inference - THE CORE

### 6.1 Engine Generation Call

**File**: `vllm/v1/engine/async_llm.py`

```python
class AsyncLLM(EngineClient):
    async def generate(
        self,
        prompts: list[str],              # ["Today's weather is"]
        sampling_params: SamplingParams,  # temperature=0.5, max_tokens=50
        request_id: str,                 # "cmpl-abc123"
        lora_request: Optional[LoRARequest] = None,
    ) -> list[RequestOutput]:
        """
        Generate completions for prompts
        """
        
        # [Step 1] Pre-process input
        # Verify inputs are valid
        assert prompts and len(prompts) > 0
        assert sampling_params.max_tokens > 0
        
        # [Step 2] Create engine request
        engine_request = EngineCoreRequest(
            request_id=request_id,
            prompts=prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            priority=0,
            arrival_time=time.time(),
        )
        
        # [Step 3] Pre-process prompts
        # Tokenize and prepare inputs
        processed_inputs = await self.input_preprocessor.preprocess(
            prompts=prompts,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        # Result: TokenIds, processed metadata
        
        # [Step 4] Queue request in processor
        # Add to the request queue waiting for batching
        request_output_queue = self.processor.add_request(
            request=engine_request,
            processed_inputs=processed_inputs,
        )
        
        # [Step 5] Wait for outputs from engine
        # The engine loop is running separately and will:
        # 1. Collect requests from queue
        # 2. Batch them together
        # 3. Send to executor (GPU)
        # 4. Collect outputs
        # 5. Put outputs in request_output_queue
        
        output_generator = self.output_processor.create_output_generator(
            request_output_queue
        )
        
        outputs = []
        async for output in output_generator:
            # Collect each output as it arrives
            outputs.append(output)
            # For our request, we get ONE output when finished
            # (since n=1 in sampling params)
        
        return outputs  # [RequestOutput(...)]
```

**Key State**:
```python
engine_request = {
    "request_id": "cmpl-abc123...",
    "prompts": ["Today's weather is"],
    "sampling_params": SamplingParams(temperature=0.5, max_tokens=50),
    "arrival_time": 1732122345.123,
}
```

**Timeline**: ~1ms (queueing request)

---

### 6.2 Processor Scheduling (Continuous in Background)

**File**: `vllm/v1/engine/processor.py`

While the request waits, the **processor** is continuously:

```python
class Processor:
    async def process(self):
        """Main processor loop - runs continuously"""
        
        while self.is_running:
            # [Every iteration]
            
            # [Step 1] Collect pending requests from queue
            pending_requests = self.request_queue.get_all()
            # Might be empty, or might contain our request
            
            # [Step 2] Group requests by compatibility
            # Requests that can run in the same batch:
            # - Same batch size
            # - Compatible sampling params
            batches = self.scheduler.schedule(pending_requests)
            
            # [Step 3] For each batch, create engine work
            for batch in batches:
                # Create batch metadata:
                # - Sequence group IDs
                # - Token budget
                # - KV cache allocation
                engine_work = self.prepare_engine_work(batch)
                
                # [Step 4] Send to executor queue
                self.executor_queue.put(engine_work)
            
            # [Step 5] Get results from executor
            results = self.executor_results_queue.get()
            # Results contain:
            # - New tokens generated
            # - Updated KV cache
            # - Sequence state
            
            # [Step 6] Process outputs
            for result in results:
                # Put outputs in corresponding request output queues
                self.request_output_queues[result.request_id].put(output)
```

**Our Request's Timeline in Processor**:
1. **t+1ms**: Request queued
2. **t+5ms**: Processor collects it (if no batch is ready)
3. **t+10ms**: Batched with other requests (or sent alone if none available)
4. **t+15ms**: Sent to executor

---

### 6.3 Executor GPU Execution (THE ACTUAL INFERENCE)

**File**: `vllm/v1/executor/*.py` (varies by GPU type)

This is where **the actual model runs on GPU**.

```python
class Executor:
    async def execute(self, batch_work):
        """
        Execute a batch of requests on GPU
        
        Input batch contains:
        - Sequence groups (requests)
        - Token IDs to process
        - KV cache information
        """
        
        # [Step 1] Load batch data to GPU memory
        batch_tokens = torch.tensor(
            [2421, 594, 6193, 318],  # "Today's weather is"
            device="cuda:0",
            dtype=torch.long,
        )
        
        # [Step 2] Run model forward pass
        # This is where GPT-2 actually processes the tokens
        
        with torch.no_grad():  # Disable gradient computation
            # Forward pass through 12 transformer layers
            output = self.model(
                input_ids=batch_tokens,
                past_key_values=cached_kv,  # From previous generation
                return_dict=True,
            )
            # output.logits shape: [batch_size, seq_len, vocab_size]
            # For us: [1, 4, 50257]
            #         (1 request, 4 tokens in prompt, 50257 vocab)
        
        # [Step 3] Get logits and apply preprocessing
        logits = output.logits[:, -1, :]  # Take last token's logits
        # Shape: [1, 50257]
        
        # [Step 4] Apply temperature scaling
        # temperature=0.5 → model will be more confident (less random)
        scaled_logits = logits / temperature  # Divide by 0.5
        # Lower temperature → higher values, sharper distribution
        
        # [Step 5] Apply sampling method (top-p)
        # top_p=1.0 → no filtering, use all tokens
        # For each token in batch, sample next token:
        
        probabilities = torch.softmax(scaled_logits, dim=-1)
        # Shape: [1, 50257]
        # Each value is between 0 and 1, sums to 1
        
        next_token = torch.multinomial(probabilities, num_samples=1)
        # Shape: [1, 1]
        # next_token[0, 0] might be 408 (token ID)
        
        # [Step 6] Update sequences
        # Add new token to the sequence
        sequence = [2421, 594, 6193, 318, 408]  # Added 408
        
        # [Step 7] Update KV cache
        # Save key/value tensors from this iteration
        cached_kv = output.past_key_values  # For next iteration
        
        # [Step 8] Check stopping conditions
        if next_token == tokenizer.eos_token_id:  # End of sequence
            finished = True
            finish_reason = "stop"
        elif len(sequence) >= max_tokens + len(prompt):  # Reached max
            finished = True
            finish_reason = "length"
        else:
            finished = False
            finish_reason = None
        
        # [Step 9] Return result for this iteration
        return ExecutorOutput(
            request_id="cmpl-abc123...",
            new_token=408,
            new_token_logprobs=-0.34,
            finished=finished,
            finish_reason=finish_reason,
        )
```

**This step repeats in a loop**:
```
Iteration 1:
  Input:  "Today's weather is"
  Output: 408 ("fine")
  
Iteration 2:
  Input:  "Today's weather is fine"
  Output: 290 ("and")
  
Iteration 3:
  Input:  "Today's weather is fine and"
  Output: 14503 ("sunny")
  
... (repeats until finished or max_tokens reached)

Iteration 50:
  Input:  "Today's weather is fine and sunny. I'm glad..."
  Output: [EOS] or max_tokens reached
  finish_reason: "length"
```

**GPU Execution Timeline**:
- Model loading: ~500ms (once at startup)
- Per-token forward pass: ~5-10ms (on RTX 3090)
- For 50 tokens: ~250-500ms total

---

### 6.4 Output Collection

**File**: `vllm/v1/engine/output_processor.py`

```python
class OutputProcessor:
    async def process_outputs(self, executor_results):
        """
        Convert executor raw outputs to RequestOutput
        """
        
        # [Step 1] Collect all tokens for this request
        all_tokens = []
        for executor_output in executor_results:  # One per iteration
            all_tokens.append(executor_output.new_token)
        
        # all_tokens = [408, 290, 14503, 13, 40, ...]  (50 tokens)
        
        # [Step 2] Decode tokens to text
        generated_text = self.tokenizer.decode(
            all_tokens,
            skip_special_tokens=True,
        )
        # "fine and sunny. I'm glad I took my umbrella!"
        
        # [Step 3] Create RequestOutput
        output = RequestOutput(
            request_id="cmpl-abc123...",
            prompt="Today's weather is",
            prompt_token_ids=[2421, 594, 6193, 318],
            outputs=[
                CompletionOutput(
                    index=0,
                    text=generated_text,
                    token_ids=all_tokens,
                    cumulative_logprob=-5.234,
                    logprobs=None,
                    finish_reason="length",
                )
            ],
            finished=True,
            created_time=1732122345.5,
        )
        
        # [Step 4] Put in output queue
        self.output_queues["cmpl-abc123..."].put(output)
        
        return output
```

**Timeline**: ~250-500ms (GPU computation for 50 tokens)

---

## Stage 7: Response Formatting

### 7.1 Convert RequestOutput to CompletionResponse

Back in `serving_completion.py`:

```python
# After awaiting engine outputs, we have:
request_output = RequestOutput(
    request_id="cmpl-abc123...",
    prompt="Today's weather is",
    prompt_token_ids=[2421, 594, 6193, 318],
    outputs=[...],
    finished=True,
)

# Format as OpenAI CompletionResponse
response = CompletionResponse(
    id="cmpl-abc123...",
    object="text_completion",
    created=1732122345,
    model="gpt2",
    choices=[
        CompletionChoice(
            index=0,
            text="fine and sunny. I'm glad I took my umbrella!",
            logprobs=None,
            finish_reason="length",
        )
    ],
    usage=CompletionUsage(
        prompt_tokens=4,              # "Today's weather is"
        completion_tokens=50,         # Generated tokens
        total_tokens=54,
    ),
)
```

**Timeline**: < 1ms (object creation)

---

## Stage 8: HTTP Response Serialization

### 8.1 FastAPI Serialization

Back in `api_server.py:create_completion()`:

```python
# Return CompletionResponse
return JSONResponse(content=response.model_dump())

# Pydantic serializes to dict:
response_dict = {
    "id": "cmpl-abc123...",
    "object": "text_completion",
    "created": 1732122345,
    "model": "gpt2",
    "choices": [
        {
            "text": "fine and sunny. I'm glad I took my umbrella!",
            "index": 0,
            "logprobs": None,
            "finish_reason": "length",
        }
    ],
    "usage": {
        "prompt_tokens": 4,
        "completion_tokens": 50,
        "total_tokens": 54,
    }
}

# FastAPI serializes to JSON
json_response = json.dumps(response_dict)

# Result:
json_response = '''
{
  "id": "cmpl-abc123...",
  "object": "text_completion",
  "created": 1732122345,
  "model": "gpt2",
  "choices": [
    {
      "text": "fine and sunny. I'm glad I took my umbrella!",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 50,
    "total_tokens": 54
  }
}
'''
```

**Timeline**: < 5ms (serialization)

---

## Stage 9: HTTP Response Transmission (Server → Client)

### 9.1 Uvicorn Sends Response

```
vLLM Server                        cURL (client)
     │                                  │
     │ HTTP/1.1 200 OK                 │
     ├────────────────────────────────→
     │ Content-Type: application/json   │
     │ Content-Length: 287              │
     │ X-Request-Id: <uuid>             │
     │                                  │
     │ {                                │
     │   "id": "cmpl-abc123...",       │
     │   "object": "text_completion",  │
     │   "created": 1732122345,        │
     │   "model": "gpt2",              │
     │   "choices": [...],             │
     │   "usage": {...}                │
     │ }                               │
     │                                  │
```

**HTTP Response Headers**:
```
HTTP/1.1 200 OK
date: Sun, 01 Dec 2024 20:12:25 GMT
server: uvicorn
content-length: 287
content-type: application/json
x-request-id: 550e8400-e29b-41d4-a716-446655440000
access-control-allow-origin: *
access-control-allow-credentials: true
```

**Timeline**: < 5ms (HTTP transmission over localhost)

---

## Stage 10: cURL Output Display

### 10.1 cURL Receives Response

```bash
# cURL receives the JSON response and prints to stdout:

$ curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "Today'"'"'s weather is",
    "max_tokens": 50,
    "temperature": 0.5
  }'

# Output:
{
  "id": "cmpl-abc123def456ghi789",
  "object": "text_completion",
  "created": 1732122345,
  "model": "gpt2",
  "choices": [
    {
      "text": "fine and sunny. I'm glad I took my umbrella!",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 50,
    "total_tokens": 54
  }
}
```

**Timeline**: < 2ms (displaying output)

---

## Complete Timeline Summary

```
Event                                          Duration      Cumulative
──────────────────────────────────────────────────────────────────────
[Client] cURL creates request                    0ms         0ms
[Network] TCP handshake                          1ms         1ms
[Network] HTTP transmission                      1ms         2ms
[Server] Uvicorn receives request                1ms         3ms
[Server] FastAPI routing                         1ms         4ms
[Server] Middleware processing                   2ms         6ms
[Server] JSON parsing & validation               5ms         11ms
[Server] Handler preprocessing                   5ms         16ms
[Engine] Request queueing                        1ms         17ms
[Engine] Processor scheduling                    10ms        27ms
[GPU] Model forward pass (50 tokens)             250-500ms   277-527ms
[Engine] Output processing                       5ms         282-532ms
[Server] Response formatting                     5ms         287-537ms
[Server] JSON serialization                      5ms         292-542ms
[Network] HTTP transmission                      1ms         293-543ms
[Client] cURL displays result                    2ms         295-545ms
──────────────────────────────────────────────────────────────────────
Total End-to-End Latency:                                    ~300-550ms

Breakdown:
- Network: ~5ms (negligible for localhost)
- Server processing: ~15ms
- GPU inference: 250-500ms (DOMINANT)
- Response formatting: ~10ms
```

---

## Data Structure Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ HTTP Request (JSON bytes)                                        │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ Uvicorn parses HTTP                                              │
│ Scope dict with headers, body                                   │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ FastAPI Router → matches POST /v1/completions                    │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ Pydantic: JSON → CompletionRequest                               │
│ {                                                                │
│   "model": "gpt2",                                               │
│   "prompt": "Today's weather is",                                │
│   "max_tokens": 50,                                              │
│   "temperature": 0.5,                                            │
│   ... (defaults for other fields)                               │
│ }                                                                │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ OpenAIServingCompletion.create_completion()                      │
│ - Validate model exists                                          │
│ - Create SamplingParams object                                   │
│ - Tokenize prompt                                                │
│ - Call engine.generate()                                         │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ AsyncLLM.generate()                                              │
│ - Create EngineCoreRequest                                       │
│ - Queue in Processor                                             │
│ - Wait for RequestOutput                                         │
└────────────────────┬─────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │ Processor scheduling  │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ Executor GPU forward  │
         │ (iterative, 50x)      │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ Token decoding        │
         │ + output formatting   │
         └───────────┬───────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ RequestOutput object:                                            │
│ {                                                                │
│   "request_id": "...",                                           │
│   "prompt": "Today's weather is",                                │
│   "prompt_token_ids": [2421, 594, 6193, 318],                    │
│   "outputs": [                                                   │
│     {                                                            │
│       "text": "fine and sunny...",                               │
│       "token_ids": [408, 290, 14503, ...],                       │
│       "finish_reason": "length",                                 │
│     }                                                            │
│   ],                                                             │
│   "finished": true,                                              │
│ }                                                                │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ Format as CompletionResponse (Pydantic)                          │
│ {                                                                │
│   "id": "cmpl-...",                                              │
│   "object": "text_completion",                                   │
│   "created": 1732122345,                                         │
│   "model": "gpt2",                                               │
│   "choices": [{"text": "...", "finish_reason": "length"}],       │
│   "usage": {"prompt_tokens": 4, "completion_tokens": 50}         │
│ }                                                                │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ Serialize to JSON (json.dumps)                                   │
│ {"id": "cmpl-...", "object": ..., ...}                          │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ JSONResponse in FastAPI                                          │
│ Wraps with HTTP headers and status code                          │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ Uvicorn HTTP Response (bytes)                                    │
│ HTTP/1.1 200 OK                                                  │
│ Content-Type: application/json                                   │
│ Content-Length: 287                                              │
│                                                                  │
│ {"id": "cmpl-...", ...}                                          │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ cURL receives HTTP response                                      │
│ Deserializes JSON and prints to stdout                           │
│ User sees the completion result                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Code Files Involved

| File | Function | Purpose |
|------|----------|---------|
| `vllm/entrypoints/openai/api_server.py` | `create_completion()` | Main request handler |
| `vllm/entrypoints/openai/serving_completion.py` | `create_completion()` | Completion handler logic |
| `vllm/v1/engine/async_llm.py` | `generate()` | Engine interface |
| `vllm/v1/engine/processor.py` | `process()` | Batching & scheduling |
| `vllm/v1/executor/abstract.py` | `execute()` | GPU execution |
| `vllm/v1/engine/output_processor.py` | `process_outputs()` | Output formatting |
| `vllm/transformers_utils/tokenizer.py` | `encode()` / `decode()` | Tokenization |

---

## Key Decision Points in Code

### 1. **Streaming vs Non-Streaming**
```python
if request.stream:
    # Return StreamingResponse with async generator
    return StreamingResponse(generator, media_type="text/event-stream")
else:
    # Return complete response
    return JSONResponse(response.model_dump())
```
**Our request**: `stream` not set → Non-streaming path

### 2. **Temperature Effect**
```python
# Temperature = 0.5 (low, deterministic)
logits = logits / temperature  # / 0.5 → higher values
probabilities = softmax(logits)
# → Higher confidence in top choices
# Result: More predictable output
```

### 3. **Stopping Conditions**
```python
if new_token == eos_token_id:
    finish_reason = "stop"
elif len(sequence) >= max_model_len + prompt_len:
    finish_reason = "length"  # Our request stops here
else:
    finish_reason = None  # Continue
```
**Our request**: Reaches `max_tokens=50` → `finish_reason="length"`

### 4. **Error Handling**
```python
if handler is None:
    return ErrorResponse(...)  # Model not found
elif validation_error:
    return HTTPException(422, ...)  # Invalid input
elif engine_error:
    return HTTPException(500, ...)  # Server error
```
**Our request**: All passes ✓

---

## Performance Bottleneck Analysis

```
For a typical completion request:

┌─────────────────────────────────────────┐
│ Total Latency: ~300-550ms               │
├─────────────────────────────────────────┤
│                                         │
│ GPU Forward Passes:  ~250-500ms (85%)   │ ← BOTTLENECK
│ ├─ Per-token latency: 5-10ms            │
│ └─ Number of tokens: 50                 │
│                                         │
│ Server Processing:   ~20-30ms (5%)      │
│ ├─ HTTP parsing: ~5ms                   │
│ ├─ Request handling: ~10ms              │
│ └─ Response serialization: ~5ms         │
│                                         │
│ Processor Scheduling: ~10-20ms (3%)     │
│ Network (localhost):  ~5ms (2%)         │
│ Other:                ~5-10ms (5%)      │
│                                         │
└─────────────────────────────────────────┘

Ways to Improve:
1. Increase batch size (use multiple requests)
2. Use prefix caching (if same prefix repeated)
3. Use longer max_model_len (process more tokens/batch)
4. Use higher temperature (more deterministic sampling)
5. Use multi-GPU (tensor parallelism)
```

---

## Error Cases & Handling

### Scenario 1: Invalid Model

```python
# Request:
{"model": "gpt3", "prompt": "Hello", ...}

# Handler:
if request.model not in self.served_model_names:
    return ErrorResponse(
        error=ErrorInfo(
            message="Model gpt3 not found. Available models: ['gpt2']",
            type="InvalidRequestError",
            code=404,
        )
    )

# HTTP Response:
HTTP/1.1 404 Not Found
{"error": {"message": "Model gpt3 not found...", ...}}
```

### Scenario 2: Invalid Temperature

```python
# Request:
{"model": "gpt2", "prompt": "Hello", "temperature": 2.5, ...}

# Pydantic Validation:
if not (0.0 <= temperature <= 2.0):
    raise ValidationError("temperature must be between 0.0 and 2.0")

# HTTP Response:
HTTP/1.1 422 Unprocessable Entity
{"error": {"message": "Invalid input", ...}}
```

### Scenario 3: Engine Crashed

```python
# Request comes in but engine is dead:

# In handler:
try:
    outputs = await engine_client.generate(...)
except EngineDeadError:
    return ErrorResponse(
        error=ErrorInfo(
            message="Engine is not running",
            type="InternalServerError",
            code=500,
        )
    )

# HTTP Response:
HTTP/1.1 500 Internal Server Error
{"error": {"message": "Engine is not running", ...}}
```

---

## Conclusion

When you execute the cURL command, here's what happens:

1. **cURL** sends JSON-formatted HTTP POST request to `localhost:8000/v1/completions`
2. **Uvicorn** receives the HTTP bytes and parses them
3. **FastAPI** routes to the `/v1/completions` endpoint handler
4. **Pydantic** validates JSON against `CompletionRequest` schema
5. **OpenAIServingCompletion** handler:
   - Validates model exists
   - Tokenizes prompt: "Today's weather is" → [2421, 594, 6193, 318]
   - Creates `SamplingParams` with temperature=0.5, max_tokens=50
6. **AsyncLLM Engine** queues request and waits for results
7. **Processor** batches the request (or runs immediately if alone)
8. **Executor** runs GPU forward passes (iteratively):
   - 50 iterations, each generating one token
   - Uses temperature 0.5 for deterministic sampling
   - Stops when reaching max_tokens=50
9. **Output Processor** decodes tokens to text: "fine and sunny..."
10. **Response Formatter** creates `CompletionResponse` object
11. **FastAPI** serializes to JSON
12. **Uvicorn** sends HTTP 200 with JSON body
13. **cURL** receives response and prints to stdout

**Total latency**: ~300-550ms (mostly GPU computation)

The entire process is optimized for concurrent requests through batching and continuous scheduling.
