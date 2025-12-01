# cURL Request Flow - Code-Level Execution Trace

This document provides the exact code executed at each step when the cURL completion request is processed.

---

## Step 1: HTTP Request Received by Uvicorn

**Network**: Uvicorn (external ASGI server) receives raw HTTP bytes and parses them.

**Pseudo-Code** (internal to Uvicorn):
```python
# Raw HTTP bytes received:
http_data = b"""POST /v1/completions HTTP/1.1\r
Host: localhost:8000\r
Content-Type: application/json\r
Content-Length: 118\r
\r
{"model":"gpt2","prompt":"Today's weather is","max_tokens":50,"temperature":0.5}"""

# Uvicorn parses into Scope dict
scope = {
    "type": "http",
    "method": "POST",
    "path": "/v1/completions",
    "query_string": b"",
    "headers": [
        (b"host", b"localhost:8000"),
        (b"content-type", b"application/json"),
        (b"content-length", b"118"),
    ],
    "server": ("0.0.0.0", 8000),
    "client": ("127.0.0.1", 54321),
    "scheme": "http",
}

# HTTP body is available as async stream
body = b'{"model":"gpt2","prompt":"Today\'s weather is","max_tokens":50,"temperature":0.5}'
```

---

## Step 2: FastAPI Router Matching

**File**: `vllm/entrypoints/openai/api_server.py`

```python
# The router object (APIRouter) defined at module level
router = APIRouter()

# Handler registered:
@router.post("/v1/completions",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
                 HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                 HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
             })
@with_cancellation  # Decorator for cancellation handling
@load_aware_call    # Decorator for load-aware scheduling
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Main entry point for completion requests"""
    handler = completion(raw_request)
    # ... rest of handler
```

**FastAPI's route matching**:
```python
# FastAPI looks up route in its route table:
# router.routes contains:
[
    Route(
        path="/v1/completions",
        endpoint=create_completion,
        methods=["POST"],
        ...
    ),
    # ... other routes ...
]

# Matches POST /v1/completions → endpoint = create_completion
```

---

## Step 3: JSON Validation (Pydantic)

**File**: `vllm/entrypoints/openai/protocol.py`

```python
# CompletionRequest is a Pydantic model
class CompletionRequest(BaseModel):
    model: str = None                           # Required
    prompt: Union[str, List[str]] = None        # Required
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16              # Default 16
    temperature: Optional[float] = 1.0          # Default 1.0, range [0-2]
    top_p: Optional[float] = 1.0                # Default 1.0, range [0-1]
    n: Optional[int] = 1
    # ... many more fields ...

# FastAPI/Pydantic validation process:
try:
    request = CompletionRequest(
        model="gpt2",
        prompt="Today's weather is",
        max_tokens=50,
        temperature=0.5,
        # All other fields get defaults
    )
    # ✓ All validations pass
except ValidationError as e:
    # ✗ If validation fails, raise HTTPException(422)
    raise HTTPException(status_code=422, detail=str(e))

# Result: request object is ready
# request.model == "gpt2"
# request.prompt == "Today's weather is"
# request.max_tokens == 50
# request.temperature == 0.5
```

---

## Step 4: Request Handler Execution

**File**: `vllm/entrypoints/openai/api_server.py`

```python
async def create_completion(request: CompletionRequest, raw_request: Request):
    """
    Handle POST /v1/completions
    """
    
    # [Line 1] Get the serving handler from app state
    handler = completion(raw_request)
    # Defined earlier as:
    # def completion(request: Request) -> Optional[OpenAIServingCompletion]:
    #     return request.app.state.openai_serving_completion
    # handler is an instance of OpenAIServingCompletion
    
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Completions API"
        )
    
    # [Line 10] Call handler
    try:
        generator = await handler.create_completion(request, raw_request)
        # This is async, so returns control to event loop
        # Execution continues here when result is available
    except OverflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # [Line 20] Check response type
    if isinstance(generator, ErrorResponse):
        # Error occurred
        return JSONResponse(
            content=generator.model_dump(),
            status_code=generator.error.code
        )
    elif isinstance(generator, CompletionResponse):
        # Non-streaming response (our case)
        return JSONResponse(content=generator.model_dump())
    else:
        # Streaming response
        return StreamingResponse(
            content=generator,
            media_type="text/event-stream"
        )
```

---

## Step 5: OpenAI Serving Handler

**File**: `vllm/entrypoints/openai/serving_completion.py`

```python
class OpenAIServingCompletion:
    
    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Request,
    ) -> Union[ErrorResponse, CompletionResponse]:
        """
        Handle completion request
        """
        
        # [Step 1] Validate model
        if request.model != self.model_config.model:
            return self.create_error_response(
                message=f"The model `{request.model}` does not exist. "
                        f"Available models: {self.served_model_names}"
            )
        # ✓ request.model == "gpt2" matches self.model_config.model
        
        # [Step 2] Get prompts
        prompts = request.prompt
        # prompts == "Today's weather is"
        
        if isinstance(prompts, str):
            prompts = [prompts]  # Convert to list
        # prompts == ["Today's weather is"]
        
        # [Step 3] Create request ID
        request_id = f"cmpl-{str(uuid.uuid4())}"
        # request_id == "cmpl-abc123def456..."
        
        # [Step 4] Log request (if enabled)
        if self.request_logger:
            self.request_logger.log_request_start(
                request_id=request_id,
                prompt=prompts[0],
                model=request.model,
                params={
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "top_p": request.top_p,
                }
            )
        
        # [Step 5] Create sampling parameters
        sampling_params = SamplingParams(
            n=request.n or 1,
            best_of=request.best_of or 1,
            presence_penalty=request.presence_penalty or 0.0,
            frequency_penalty=request.frequency_penalty or 0.0,
            repetition_penalty=request.repetition_penalty or 1.0,
            temperature=request.temperature or 1.0,  # ← 0.5
            top_p=request.top_p or 1.0,              # ← 1.0
            top_k=request.top_k or -1,
            min_p=request.min_p or 0.0,
            use_beam_search=request.use_beam_search or False,
            length_penalty=request.length_penalty or 1.0,
            early_stopping=request.early_stopping or False,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            include_stop_str_in_output=True,
            skip_special_tokens=request.skip_special_tokens or True,
        )
        # sampling_params.temperature == 0.5
        # sampling_params.top_p == 1.0
        
        # [Step 6] Preprocess input
        preprocessor_inputs = self._preprocess_inputs(
            prompts=prompts,
            sampling_params=sampling_params,
        )
        # Result: tokenized input ready for engine
        
        # [Step 7] Call engine to generate
        outputs = await self.engine_client.generate(
            prompts=prompts,                  # ["Today's weather is"]
            sampling_params=sampling_params,  # temperature=0.5, etc.
            request_id=request_id,
        )
        # outputs is list[RequestOutput]
        # RequestOutput contains the generated text
        
        # [Step 8] Format response
        response = CompletionResponse(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=request.model,  # "gpt2"
            choices=[
                CompletionChoice(
                    index=0,
                    text=output.outputs[0].text,      # Generated text
                    logprobs=None,
                    finish_reason=output.outputs[0].finish_reason,
                )
                for output in outputs
            ],
            usage=CompletionUsage(
                prompt_tokens=len(outputs[0].prompt_token_ids),
                completion_tokens=sum(
                    len(o.token_ids) for o in outputs[0].outputs
                ),
                total_tokens=len(outputs[0].prompt_token_ids) + sum(
                    len(o.token_ids) for o in outputs[0].outputs
                ),
            ),
        )
        # response is CompletionResponse object ready to return
        
        # [Step 9] Log completion
        if self.request_logger:
            self.request_logger.log_request_end(
                request_id=request_id,
                prompt=prompts[0],
                outputs=outputs[0].outputs,
            )
        
        # [Step 10] Return response
        return response
```

---

## Step 6: Engine Generation (THE CRITICAL PART)

**File**: `vllm/v1/engine/async_llm.py`

```python
class AsyncLLM(EngineClient):
    
    async def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
    ) -> List[RequestOutput]:
        """
        Generate completions
        """
        
        # [Step 1] Create engine request
        engine_request = EngineCoreRequest(
            request_id=request_id,
            prompts=prompts,  # ["Today's weather is"]
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        
        # [Step 2] Preprocess inputs
        processed_inputs = await self.input_preprocessor.preprocess(
            prompts=prompts,
            sampling_params=sampling_params,
        )
        # processed_inputs.prompt_token_ids = [2421, 594, 6193, 318]
        # (tokens for "Today's weather is")
        
        # [Step 3] Create output queue for this request
        output_queue: asyncio.Queue[RequestOutput] = asyncio.Queue()
        
        # [Step 4] Add request to processor
        self.processor.add_request(
            request_id=request_id,
            request=engine_request,
            processed_inputs=processed_inputs,
            output_queue=output_queue,
        )
        # Now the request is queued and waiting for the engine loop
        
        # [Step 5] Wait for outputs
        # The engine loop runs separately and processes requests
        # When done, it puts outputs in output_queue
        outputs = []
        while True:
            # Get output from queue (blocks until available)
            output = await output_queue.get()
            outputs.append(output)
            
            # Check if this is the final output
            if output.finished:
                break
        
        # [Step 6] Return all outputs
        return outputs
```

---

## Step 7: Processor Batching (Concurrent with Step 6)

**File**: `vllm/v1/engine/processor.py`

The processor runs continuously in a background task. When our request is queued, it gets batched:

```python
class Processor:
    
    async def process_requests(self):
        """
        Main loop that runs continuously
        """
        while self.is_running:
            # [Step 1] Collect requests from queue
            pending_requests = self.request_queue.get_all_pending()
            # pending_requests contains our request
            
            # [Step 2] Create batches
            batches = self._create_batches(pending_requests)
            # For our single request: batches = [Batch([our_request])]
            
            # [Step 3] For each batch, create execution work
            for batch in batches:
                exec_work = ExecutorWork(
                    batch_size=len(batch.requests),  # 1
                    request_ids=[request_id for _, request_id in batch.requests],
                    token_ids=[req.processed_inputs.token_ids for _, req in batch.requests],
                    # etc.
                )
                
                # [Step 4] Send to executor
                self.executor_queue.put(exec_work)
                
                # [Step 5] Wait for executor results
                executor_output = await self.executor_results_queue.get()
                
                # [Step 6] Process outputs and put in request queues
                for request_id, output in executor_output.items():
                    self.request_output_queues[request_id].put(output)
                    # output goes back to our async function's output_queue
```

---

## Step 8: GPU Execution (Executor)

**File**: `vllm/v1/executor/cuda_executor.py` (or CPU executor)

```python
class CudaExecutor(Executor):
    
    async def execute(self, batch_work: ExecutorWork) -> Dict[str, RequestOutput]:
        """
        Execute batch on GPU
        """
        
        # [Step 1] Prepare batch on GPU
        batch_tokens = torch.tensor(
            [2421, 594, 6193, 318],  # "Today's weather is"
            device="cuda:0",
            dtype=torch.long,
        )
        # Shape: [1, 4] (batch_size=1, seq_len=4)
        
        # [Step 2] Get cached KV (first iteration, this is empty)
        cached_kv = None  # First pass, no previous cache
        
        # [Step 3] Forward pass through model
        with torch.no_grad():
            model_output = self.model(
                input_ids=batch_tokens,
                past_key_values=cached_kv,
                return_dict=True,
                attention_mask=None,
            )
        # model_output.logits shape: [1, 4, 50257]
        #   (batch_size=1, seq_len=4, vocab_size=50257)
        
        # [Step 4] Get last token logits
        logits = model_output.logits[:, -1, :]  # [1, 50257]
        # These are raw scores for each vocab token
        
        # [Step 5] Apply temperature scaling
        temperature = 0.5
        logits = logits / temperature  # Divide by 0.5
        # Lower values become higher, sharper distribution
        
        # [Step 6] Apply top-p filtering (if top_p < 1.0)
        # In our case, top_p=1.0, so no filtering
        
        # [Step 7] Sample next token
        probabilities = torch.softmax(logits, dim=-1)  # [1, 50257]
        # Now probabilities sum to 1.0
        
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).squeeze()  # Shape: [1] → scalar
        # next_token = 408  (might be token ID for "fine")
        
        # [Step 8] Update KV cache for next iteration
        cached_kv = model_output.past_key_values
        # Save for next token prediction
        
        # [Step 9] Check stopping conditions
        max_tokens = 50
        num_tokens_generated = 1
        
        if next_token == self.tokenizer.eos_token_id:
            finished = True
            finish_reason = "stop"
        elif num_tokens_generated >= max_tokens:
            finished = True
            finish_reason = "length"
        else:
            finished = False
            finish_reason = None
        
        # [Step 10] Return output for this iteration
        return ExecutorOutput(
            request_id=request_id,
            new_token_id=next_token.item(),  # 408
            finished=finished,
            finish_reason=finish_reason,
        )

# This steps repeats in a loop:
# Iteration 1: generate 1st token (408)
# Iteration 2: generate 2nd token (290)
# Iteration 3: generate 3rd token (14503)
# ...
# Iteration 50: generate 50th token, finished=True
```

**The loop structure**:
```python
# In the engine loop, this executes repeatedly:
all_tokens = [2421, 594, 6193, 318]  # Initial prompt

for iteration in range(max_tokens):
    # Run executor forward pass
    next_token = execute_forward_pass(
        input_ids=all_tokens[-1:],  # Last token only
        cached_kv=cached_kv,
    )
    
    all_tokens.append(next_token)  # Add to sequence
    
    if next_token == eos_token_id or len(all_tokens) >= max_model_len:
        break

# After loop:
# all_tokens = [2421, 594, 6193, 318, 408, 290, 14503, 13, 40, ...]
```

---

## Step 9: Token Decoding

**File**: `vllm/v1/engine/output_processor.py`

```python
class OutputProcessor:
    
    async def process_output(self, executor_output) -> RequestOutput:
        """
        Convert executor output to RequestOutput
        """
        
        # [Step 1] Collect all tokens
        token_ids = [2421, 594, 6193, 318, 408, 290, 14503, ...]  # 50 tokens total
        
        # [Step 2] Decode tokens to text
        decoded_text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )
        # decoded_text = "Today's weather is fine and sunny. I'm glad I took my umbrella!"
        
        # [Step 3] Create RequestOutput
        request_output = RequestOutput(
            request_id="cmpl-abc123...",
            prompt="Today's weather is",
            prompt_token_ids=[2421, 594, 6193, 318],
            outputs=[
                CompletionOutput(
                    index=0,
                    text=decoded_text,
                    token_ids=token_ids[4:],  # Only generated tokens
                    cumulative_logprob=-5.234,
                    logprobs=None,
                    finish_reason="length",
                )
            ],
            finished=True,
            created_time=time.time(),
        )
        
        return request_output
```

---

## Step 10: Response Creation

**Back in serving_completion.py**:

```python
# After engine returns outputs
outputs = [request_output]  # List with one RequestOutput

# Create OpenAI response format
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
        prompt_tokens=4,       # len([2421, 594, 6193, 318])
        completion_tokens=50,  # Generated 50 tokens
        total_tokens=54,       # 4 + 50
    ),
)

return response
```

---

## Step 11: HTTP Response Serialization

**Back in api_server.py**:

```python
# In create_completion():
return JSONResponse(content=response.model_dump())

# response.model_dump() produces dict:
{
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

# JSONResponse serializes to JSON string
json_string = json.dumps({...})

# Uvicorn adds HTTP headers:
"""
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 287
X-Request-Id: 550e8400-e29b-41d4-a716-446655440000
Date: Sun, 01 Dec 2024 20:12:25 GMT

{"id":"cmpl-abc123...","object":"text_completion",...}
"""
```

---

## Step 12: Client Response Reception

```bash
# cURL receives the HTTP response

$ curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "Today'"'"'s weather is",
    "max_tokens": 50,
    "temperature": 0.5
  }'

# Output displayed:
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

---

## Variable State Changes Throughout Request

```
[HTTP Request] → [FastAPI Parser] → [Pydantic]
  ↓
CompletionRequest(
  model="gpt2",
  prompt="Today's weather is",
  max_tokens=50,
  temperature=0.5
)
  ↓
[OpenAI Handler] → [Engine Request Creation]
  ↓
EngineCoreRequest(
  request_id="cmpl-...",
  prompts=["Today's weather is"],
  sampling_params=SamplingParams(temperature=0.5, ...)
)
  ↓
[Input Preprocessing]
  ↓
Processed Input(
  prompt_token_ids=[2421, 594, 6193, 318],
  attention_mask=[1, 1, 1, 1]
)
  ↓
[Processor Queueing] → [GPU Execution Loop (50 iterations)]
  ↓
Generated Tokens:
  [408, 290, 14503, 13, 40, 1101, 6568, ...]
  ↓
[Token Decoding]
  ↓
RequestOutput(
  prompt_token_ids=[2421, 594, 6193, 318],
  output_text="fine and sunny. I'm glad I took my umbrella!",
  finished=True
)
  ↓
[Response Formatting]
  ↓
CompletionResponse(
  choices=[{"text": "fine and sunny...", "finish_reason": "length"}],
  usage={"prompt_tokens": 4, "completion_tokens": 50, "total_tokens": 54}
)
  ↓
[JSON Serialization]
  ↓
HTTP Response (JSON string)
  ↓
[Client Display]
```

---

## Key Functions Called in Order

| Step | File | Function | Input | Output |
|------|------|----------|-------|--------|
| 1 | api_server.py | `create_completion()` | CompletionRequest | CompletionResponse |
| 2 | serving_completion.py | `create_completion()` | CompletionRequest | CompletionResponse |
| 3 | async_llm.py | `generate()` | prompts, sampling_params | List[RequestOutput] |
| 4 | processor.py | `add_request()` | EngineCoreRequest | asyncio.Queue |
| 5 | executor.py | `execute()` | ExecutorWork | ExecutorOutput |
| 6 | output_processor.py | `process_output()` | ExecutorOutput | RequestOutput |
| 7 | serving_completion.py | response formatting | RequestOutput | CompletionResponse |

---

This detailed trace shows exactly what code runs when the cURL command executes, with actual variable values and data structures involved at each stage.
