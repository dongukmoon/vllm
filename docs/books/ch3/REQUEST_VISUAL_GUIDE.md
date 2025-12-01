# cURL Request: Visual Flow & Diagrams

Visual representations of the cURL request flow through vLLM.

---

## 1. Complete Request/Response Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│ CLIENT SIDE                                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ $ curl http://localhost:8000/v1/completions \                         │
│     -H "Content-Type: application/json" \                             │
│     -d '{"model":"gpt2","prompt":"Today'"'"'s weather is", ...}'       │
│                                                                         │
│ ▼                                                                       │
│ Create HTTP Request:                                                    │
│   POST /v1/completions HTTP/1.1                                        │
│   Host: localhost:8000                                                 │
│   Content-Type: application/json                                       │
│   Content-Length: 118                                                  │
│                                                                         │
│   {"model":"gpt2","prompt":"Today's weather is", "max_tokens":50, ...} │
│                                                                         │
│ ▼                                                                       │
│ TCP Connection (3-way handshake)                                        │
│ Send HTTP Request over network                                         │
│                                                                         │
│ ┌─────────────────────────────────────┐                                │
│ │ TIME: 0-5ms                         │                                │
│ │ Network latency for local request   │                                │
│ └─────────────────────────────────────┘                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP Request
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SERVER SIDE - UVICORN & FASTAPI                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Uvicorn (ASGI Server):                                                  │
│   Parse HTTP bytes → Scope dict                                        │
│   Extract headers, body, path, method                                  │
│                                                                         │
│ ▼                                                                       │
│ FastAPI Router:                                                         │
│   Match POST /v1/completions                                           │
│   Route to create_completion() handler                                 │
│                                                                         │
│ ▼                                                                       │
│ Dependency Injection:                                                   │
│   validate_json_request()                                              │
│                                                                         │
│ ▼                                                                       │
│ Pydantic Validation:                                                    │
│   Parse JSON {"model":"gpt2", "prompt":"...", "max_tokens":50, ...}   │
│   Validate against CompletionRequest schema                            │
│   ✓ All fields valid                                                   │
│                                                                         │
│ ▼                                                                       │
│ Handler Decorators:                                                     │
│   @with_cancellation    → Cancellation support                        │
│   @load_aware_call      → Load tracking                               │
│                                                                         │
│ ┌─────────────────────────────────────┐                                │
│ │ TIME: 5-15ms                        │                                │
│ │ HTTP parsing + validation           │                                │
│ └─────────────────────────────────────┘                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ CompletionRequest object
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SERVER SIDE - OPENAI SERVING HANDLER                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ OpenAIServingCompletion.create_completion():                           │
│                                                                         │
│   1. Validate model exists                                             │
│      ✓ "gpt2" is configured                                           │
│                                                                         │
│   2. Process prompt                                                     │
│      "Today's weather is" → ["Today's weather is"]                    │
│                                                                         │
│   3. Create sampling parameters                                         │
│      SamplingParams(                                                    │
│        temperature=0.5,                                                │
│        max_tokens=50,                                                  │
│        top_p=1.0,                                                      │
│        ...                                                             │
│      )                                                                 │
│                                                                         │
│   4. Tokenize prompt                                                    │
│      "Today's weather is"                                             │
│              ↓                                                         │
│      [2421, 594, 6193, 318]  (4 tokens)                               │
│                                                                         │
│   5. Call engine.generate()                                            │
│      (Async call - returns control to event loop)                      │
│                                                                         │
│ ┌─────────────────────────────────────┐                                │
│ │ TIME: 15-20ms                       │                                │
│ │ Preprocessing & parameter creation  │                                │
│ └─────────────────────────────────────┘                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ engine.generate() call
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SERVER SIDE - ENGINE & INFERENCE                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐   │
│ │ REQUEST QUEUEING (AsyncLLM)                                     │   │
│ │                                                                 │   │
│ │ 1. Create EngineCoreRequest                                    │   │
│ │    ├─ request_id: "cmpl-abc123..."                             │   │
│ │    ├─ prompts: ["Today's weather is"]                          │   │
│ │    └─ sampling_params: {temperature: 0.5, max_tokens: 50, ...} │   │
│ │                                                                 │   │
│ │ 2. Add to processor queue                                      │   │
│ │    ├─ Request joins waiting queue                              │   │
│ │    └─ Async wait for output_queue                              │   │
│ │                                                                 │   │
│ │ ┌───────────────────────────────────────┐                      │   │
│ │ │ TIME: 20-22ms                         │                      │   │
│ │ │ Request queueing                      │                      │   │
│ │ └───────────────────────────────────────┘                      │   │
│ └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│ ▼                                                                       │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐   │
│ │ PROCESSOR BATCHING (Continuous Loop)                            │   │
│ │                                                                 │   │
│ │ Processor runs independently:                                   │   │
│ │   1. Collect pending requests from queue                        │   │
│ │      └─ Our request is collected                                │   │
│ │                                                                 │   │
│ │   2. Group compatible requests                                  │   │
│ │      └─ Creates a batch: [our_request]                          │   │
│ │                                                                 │   │
│ │   3. Send batch to executor                                     │   │
│ │      └─ Executor queue receives: ExecutorWork                   │   │
│ │                                                                 │   │
│ │ ┌───────────────────────────────────────┐                      │   │
│ │ │ TIME: 22-32ms                         │                      │   │
│ │ │ Processor scheduling                  │                      │   │
│ │ └───────────────────────────────────────┘                      │   │
│ └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│ ▼                                                                       │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐   │
│ │ GPU EXECUTOR - ITERATIVE TOKEN GENERATION                       │   │
│ │                                                                 │   │
│ │ ╔═══════════════════════════════════════════════════════════╗   │   │
│ │ ║ ITERATION 1: Generate 1st token                           ║   │   │
│ │ ╚═══════════════════════════════════════════════════════════╝   │   │
│ │                                                                 │   │
│ │   Input:  [2421, 594, 6193, 318] (4 tokens)                   │   │
│ │   ├─ Load to GPU memory                                        │   │
│ │   ├─ Forward pass through GPT-2 (12 layers)                   │   │
│ │   │  └─ Output: logits [1, 50257]                              │   │
│ │   ├─ Apply temperature scaling: / 0.5                          │   │
│ │   ├─ Apply softmax: → probabilities                            │   │
│ │   ├─ Sample token: multinomial(probabilities, 1)              │   │
│ │   └─ Output: 408 ("fine")                                      │   │
│ │   Finished: False                                              │   │
│ │                                                                 │   │
│ │   ┌───────────────────────────────────────┐                    │   │
│ │   │ GPU Time: 5-10ms per iteration        │                    │   │
│ │   └───────────────────────────────────────┘                    │   │
│ │                                                                 │   │
│ │ ╔═══════════════════════════════════════════════════════════╗   │   │
│ │ ║ ITERATION 2: Generate 2nd token                           ║   │   │
│ │ ╚═══════════════════════════════════════════════════════════╝   │   │
│ │                                                                 │   │
│ │   Input:  [2421, 594, 6193, 318, 408] (5 tokens)              │   │
│ │   └─ Output: 290 ("and")                                       │   │
│ │   Finished: False                                              │   │
│ │                                                                 │   │
│ │ [... iterations 3-49 ...]                                     │   │
│ │                                                                 │   │
│ │ ╔═══════════════════════════════════════════════════════════╗   │   │
│ │ ║ ITERATION 50: Generate 50th token                         ║   │   │
│ │ ╚═══════════════════════════════════════════════════════════╝   │   │
│ │                                                                 │   │
│ │   Input:  [2421, 594, 6193, 318, 408, 290, ..., last]         │   │
│ │   └─ Output: [EOS] or last token                               │   │
│ │   Finished: True (reached max_tokens=50)                       │   │
│ │   finish_reason: "length"                                      │   │
│ │                                                                 │   │
│ │ ┌───────────────────────────────────────┐                      │   │
│ │ │ GPU TIME: 250-500ms                   │                      │   │
│ │ │ (50 iterations × 5-10ms each)         │                      │   │
│ │ │ DOMINANT BOTTLENECK!                  │                      │   │
│ │ └───────────────────────────────────────┘                      │   │
│ └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│ ▼                                                                       │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐   │
│ │ OUTPUT PROCESSING                                               │   │
│ │                                                                 │   │
│ │ 1. Collect all 50 generated tokens                             │   │
│ │    [408, 290, 14503, 13, 40, 1101, 6568, ...]                 │   │
│ │                                                                 │   │
│ │ 2. Decode to text                                              │   │
│ │    "fine and sunny. I'm glad I took my umbrella!"              │   │
│ │                                                                 │   │
│ │ 3. Create RequestOutput                                        │   │
│ │    ├─ request_id: "cmpl-abc123..."                             │   │
│ │    ├─ text: "fine and sunny. I'm..."                           │   │
│ │    ├─ token_ids: [408, 290, 14503, ...]                        │   │
│ │    ├─ finish_reason: "length"                                  │   │
│ │    └─ finished: True                                           │   │
│ │                                                                 │   │
│ │ 4. Put in output queue                                         │   │
│ │    └─ Unblocks the awaiting handler                            │   │
│ │                                                                 │   │
│ │ ┌───────────────────────────────────────┐                      │   │
│ │ │ TIME: 510-520ms                       │                      │   │
│ │ │ Output processing                     │                      │   │
│ │ └───────────────────────────────────────┘                      │   │
│ └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ RequestOutput
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SERVER SIDE - RESPONSE FORMATTING                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Back in OpenAIServingCompletion.create_completion():                   │
│                                                                         │
│ 1. Handler receives RequestOutput from engine                           │
│                                                                         │
│ 2. Format as OpenAI CompletionResponse:                                │
│    {                                                                   │
│      "id": "cmpl-abc123...",                                          │
│      "object": "text_completion",                                     │
│      "created": 1732122345,                                           │
│      "model": "gpt2",                                                 │
│      "choices": [{                                                    │
│        "text": "fine and sunny. I'm glad I took my umbrella!",         │
│        "index": 0,                                                    │
│        "finish_reason": "length",                                     │
│        "logprobs": null                                               │
│      }],                                                              │
│      "usage": {                                                       │
│        "prompt_tokens": 4,                                            │
│        "completion_tokens": 50,                                       │
│        "total_tokens": 54                                             │
│      }                                                                │
│    }                                                                  │
│                                                                         │
│ 3. Return CompletionResponse object                                    │
│                                                                         │
│ ┌─────────────────────────────────────┐                                │
│ │ TIME: 520-525ms                     │                                │
│ │ Response formatting                 │                                │
│ └─────────────────────────────────────┘                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ CompletionResponse
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SERVER SIDE - HTTP RESPONSE                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Back in api_server.create_completion():                                │
│                                                                         │
│ 1. Return JSONResponse(content=response.model_dump())                   │
│                                                                         │
│ 2. Pydantic serializes to dict                                          │
│                                                                         │
│ 3. FastAPI/Uvicorn converts to JSON                                     │
│                                                                         │
│ 4. Add HTTP headers and status code                                     │
│    HTTP/1.1 200 OK                                                     │
│    Content-Type: application/json                                      │
│    Content-Length: 287                                                 │
│    X-Request-Id: 550e8400-e29b-41d4-a716-446655440000                 │
│    Date: Sun, 01 Dec 2024 20:12:25 GMT                                │
│                                                                         │
│ 5. Send HTTP response over network                                      │
│                                                                         │
│ ┌─────────────────────────────────────┐                                │
│ │ TIME: 525-530ms                     │                                │
│ │ HTTP serialization + transmission   │                                │
│ └─────────────────────────────────────┘                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP Response
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ CLIENT SIDE                                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ cURL receives HTTP response over network                                │
│                                                                         │
│ Deserialize JSON                                                        │
│                                                                         │
│ Print to stdout:                                                        │
│                                                                         │
│ {                                                                       │
│   "id": "cmpl-abc123def456ghi789",                                     │
│   "object": "text_completion",                                         │
│   "created": 1732122345,                                               │
│   "model": "gpt2",                                                     │
│   "choices": [                                                         │
│     {                                                                  │
│       "text": "fine and sunny. I'm glad I took my umbrella!",          │
│       "index": 0,                                                      │
│       "logprobs": null,                                                │
│       "finish_reason": "length"                                        │
│     }                                                                  │
│   ],                                                                   │
│   "usage": {                                                           │
│     "prompt_tokens": 4,                                                │
│     "completion_tokens": 50,                                           │
│     "total_tokens": 54                                                 │
│   }                                                                    │
│ }                                                                       │
│                                                                         │
│ ┌─────────────────────────────────────┐                                │
│ │ TIME: 530-535ms                     │                                │
│ │ Network transmission + display      │                                │
│ └─────────────────────────────────────┘                                │
│                                                                         │
│ ✓ COMPLETE                                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

TOTAL TIME: ~530-535ms
```

---

## 2. Code Execution Timeline

```
TIMELINE:
┌─────────────────────────────────────────────────────────────┐
│ 0ms      cURL sends HTTP request                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 0-2ms    Network transmission (localhost)                   │
│ │                                                           │
│ ├─ 2ms   Uvicorn receives HTTP bytes                        │
│ │        Parse HTTP → Scope dict                           │
│ │                                                           │
│ ├─ 3ms   FastAPI router matching                           │
│ │        Match: POST /v1/completions                       │
│ │                                                           │
│ ├─ 8ms   Pydantic JSON validation                          │
│ │        CompletionRequest schema check                    │
│ │        ✓ All valid                                       │
│ │                                                           │
│ ├─ 12ms  OpenAI serving handler                            │
│ │        Model validation                                  │
│ │        Prompt tokenization                               │
│ │        SamplingParams creation                           │
│ │                                                           │
│ ├─ 15ms  Engine.generate() called                          │
│ │        Request queued in processor                       │
│ │        Async wait begins                                 │
│ │                                                           │
│ ├─ 20ms  Processor schedules batch                         │
│ │        Batch created: [our_request]                      │
│ │                                                           │
│ ├─ 25ms  Executor receives batch                           │
│ │        GPU forward pass iteration 1                      │
│ │        Token generated: 408                              │
│ │                                                           │
│ ├─ 35ms  GPU forward pass iteration 2                      │
│ │        Token generated: 290                              │
│ │                                                           │
│ ├─ 45ms  GPU forward pass iteration 3                      │
│ │        Token generated: 14503                            │
│ │                                                           │
│ ├─ ... [iterations 4-49, ~8ms each]                        │
│ │                                                           │
│ ├─ 505ms GPU forward pass iteration 50 (LAST)              │
│ │        Token generated, finished=True                    │
│ │                                                           │
│ ├─ 510ms Output processing                                 │
│ │        Token decoding: → text                            │
│ │        RequestOutput created                             │
│ │        Put in output queue                               │
│ │        Handler unblocked                                 │
│ │                                                           │
│ ├─ 515ms Response formatting                               │
│ │        CompletionResponse object                         │
│ │        JSONResponse wrapper                              │
│ │                                                           │
│ ├─ 520ms HTTP serialization                                │
│ │        JSON string created                               │
│ │        HTTP headers added                                │
│ │                                                           │
│ ├─ 525ms Network transmission (response)                   │
│ │                                                           │
│ └─ 530ms cURL receives and displays                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

BREAKDOWN:
Network: ~10ms (5% of total)
HTTP Parsing: ~10ms (2% of total)
Validation/Processing: ~15ms (3% of total)
Processor: ~10ms (2% of total)
GPU Computation: ~485ms (91% of total) ← BOTTLENECK
Response Formatting: ~10ms (2% of total)

Total: ~530ms
```

---

## 3. Data Structure Evolution

```
Initial HTTP Request:
┌──────────────────────────────┐
│ POST /v1/completions         │
│ {                            │
│   "model": "gpt2",          │
│   "prompt": "Today's..."    │
│   "max_tokens": 50,         │
│   "temperature": 0.5        │
│ }                            │
└────────────┬─────────────────┘
             │
             ▼
CompletionRequest (Pydantic):
┌──────────────────────────────┐
│ model="gpt2"                 │
│ prompt="Today's weather is"  │
│ max_tokens=50                │
│ temperature=0.5              │
│ top_p=1.0 (default)          │
│ ... (10+ other fields)       │
└────────────┬─────────────────┘
             │
             ▼
EngineCoreRequest:
┌──────────────────────────────┐
│ request_id="cmpl-abc..."     │
│ prompts=["Today's..."]       │
│ sampling_params={...}        │
│ arrival_time=timestamp       │
└────────────┬─────────────────┘
             │
             ▼
Tokenized Input:
┌──────────────────────────────┐
│ token_ids=[2421, 594, ...]   │
│ attention_mask=[1, 1, ...]   │
└────────────┬─────────────────┘
             │
             ▼
GPU Logits (50 iterations):
┌──────────────────────────────┐
│ Iteration 1: logits [50257]  │
│   → token_id: 408            │
│ Iteration 2: logits [50257]  │
│   → token_id: 290            │
│ ...                          │
│ Iteration 50: logits [50257] │
│   → token_id: eos            │
└────────────┬─────────────────┘
             │
             ▼
Generated Tokens:
┌──────────────────────────────┐
│ [408, 290, 14503, 13, 40,   │
│  1101, 6568, 40, 40, 464,   │
│  ...] (50 tokens)            │
└────────────┬─────────────────┘
             │
             ▼
Decoded Text:
┌──────────────────────────────┐
│ "fine and sunny. I'm glad    │
│  I took my umbrella!"        │
└────────────┬─────────────────┘
             │
             ▼
RequestOutput:
┌──────────────────────────────┐
│ request_id="cmpl-..."        │
│ text="fine and sunny..."     │
│ token_ids=[408, 290, ...]    │
│ finish_reason="length"       │
│ finished=True                │
└────────────┬─────────────────┘
             │
             ▼
CompletionResponse:
┌──────────────────────────────┐
│ {                            │
│   "id": "cmpl-...",         │
│   "object": "text_....",    │
│   "created": 1732122345,    │
│   "model": "gpt2",          │
│   "choices": [{             │
│     "text": "fine...",      │
│     "finish_reason": "length"
│   }],                        │
│   "usage": {                 │
│     "prompt_tokens": 4,     │
│     "completion_tokens": 50 │
│   }                          │
│ }                            │
└────────────┬─────────────────┘
             │
             ▼
HTTP Response (JSON):
┌──────────────────────────────┐
│ HTTP/1.1 200 OK              │
│ Content-Type: application/... │
│ {"id": "cmpl-...", ...}     │
└──────────────────────────────┘
```

---

## 4. Concurrency & Async Pattern

```
Async Event Loop Timeline:

T=0ms    Request 1 arrives
         │
         ├─ HTTP parsing (1ms)
         ├─ Validation (5ms)
         ├─ Handler preprocessing (5ms)
         │
         └─ engine.generate() → AWAIT
              │ (async call returns control)
              │
         
T=16ms   [Event Loop continues]
         Request 2 arrives
         │
         ├─ HTTP parsing (1ms)
         ├─ Validation (5ms)
         ├─ Handler preprocessing (5ms)
         │
         └─ engine.generate() → AWAIT
              │ (async call returns control)
         
         Meanwhile in background:
         - Processor collecting batches
         - GPU executing both batches efficiently

T=25ms   Request 1 queued in processor
         Request 2 queued in processor

T=30ms   Batch created: [Request 1, Request 2]
         GPU executes both in parallel!

T=530ms  GPU finishes iteration 50
         Request 1 output ready
         → Unblock Request 1 handler
         → Format response
         → Send to client

T=532ms  Request 2 output ready
         → Unblock Request 2 handler
         → Format response
         → Send to client

Both requests are processed efficiently through:
- Async I/O (no blocking on network/disk)
- GPU batching (multiple requests in same forward pass)
- Event loop scheduling (multiple requests concurrent)
```

---

## 5. GPU Memory Access Pattern

```
GPU Memory During Inference:

INITIAL STATE:
┌─────────────────────────────┐
│ GPT-2 Weights (500MB)       │ ← Loaded once
│ ├─ Embedding layer          │
│ ├─ 12 Transformer blocks    │
│ └─ Output layer             │
├─────────────────────────────┤
│ KV Cache Pool (3GB)         │ ← Allocated but empty
├─────────────────────────────┤
│ Working Buffers (1GB)       │ ← For forward pass
├─────────────────────────────┤
│ Free Memory                 │
└─────────────────────────────┘

DURING ITERATION 1:
┌─────────────────────────────┐
│ GPT-2 Weights              │
├─────────────────────────────┤
│ Input: [2421, 594, ...]    │ ← 4 tokens
│ Attention tensors          │
│ Hidden states              │
│ Logits: [50257]            │ ← For 4 tokens
├─────────────────────────────┤
│ KV Cache: [seq=4, ...]     │ ← Growing
├─────────────────────────────┤
│ Free Memory                 │
└─────────────────────────────┘

DURING ITERATION 50:
┌─────────────────────────────┐
│ GPT-2 Weights              │
├─────────────────────────────┤
│ Input: [full 54 tokens]    │ ← 4 + 50 tokens
│ Attention tensors          │
│ Hidden states              │
│ Logits: [50257]            │
├─────────────────────────────┤
│ KV Cache: [seq=54, ...]    │ ← Fully populated
├─────────────────────────────┤
│ Free Memory                 │
└─────────────────────────────┘

Access Pattern:
1. Read prompt tokens from CPU
2. Load to GPU memory
3. Read weights (CACHED in GPU)
4. Compute forward pass
5. Write KV cache (reused next iteration)
6. Read KV cache (from step 5)
7. Repeat until finished
```

---

## 6. Temperature Effect Visualization

```
Temperature Scaling Impact:

Original Logits (raw model output):
[2.3, 1.5, 0.8, -0.2, -1.1, ...]

TEMPERATURE = 0.5 (Low, Deterministic)
Scaled: [2.3/0.5, 1.5/0.5, 0.8/0.5, ...]
      = [4.6, 3.0, 1.6, -0.4, -2.2, ...]
       ↓ Softmax
Prob:  [0.98, 0.015, 0.003, 0.0001, 0.0001, ...]
       
       Distribution is SHARP
       → Sample: Almost always token 0
       → Output: Very predictable ✓

TEMPERATURE = 1.0 (Default, Balanced)
Scaled: [2.3, 1.5, 0.8, -0.2, -1.1, ...]
       ↓ Softmax
Prob:  [0.75, 0.15, 0.06, 0.02, 0.02, ...]
       
       Distribution is BALANCED
       → Sample: Mostly token 0, sometimes token 1
       → Output: Reasonably random

TEMPERATURE = 2.0 (High, Random)
Scaled: [1.15, 0.75, 0.4, -0.1, -0.55, ...]
       ↓ Softmax
Prob:  [0.35, 0.22, 0.18, 0.14, 0.11, ...]
       
       Distribution is FLAT
       → Sample: Could be any token
       → Output: Very random/creative

Our request: temperature=0.5 → Deterministic output
```

---

## 7. Error Handling Flow

```
If request had invalid JSON:

curl POST with invalid JSON
         │
         ▼
Uvicorn receives bytes
         │
         ▼
FastAPI tries to parse
         │
         ├─ JSON parse error!
         │
         ▼
FastAPI exception handler
         │
         ├─ Create ErrorResponse
         │  ├─ status_code: 422
         │  ├─ detail: "Invalid JSON"
         │
         ▼
Return JSONResponse(status=422)
         │
         ▼
HTTP/1.1 422 Unprocessable Entity
{"error": {"message": "Invalid JSON", ...}}
         │
         ▼
cURL displays error

If request had invalid temperature (e.g., 2.5):

Pydantic validation
         │
         ├─ temperature: 2.5
         ├─ Expected: 0.0 <= x <= 2.0
         ├─ Validation fails!
         │
         ▼
FastAPI exception handler
         │
         ├─ Create ErrorResponse
         │  ├─ status_code: 422
         │  ├─ detail: "temperature must be..."
         │
         ▼
Return JSONResponse(status=422)
         │
         ▼
HTTP/1.1 422 Unprocessable Entity
{"error": {"message": "temperature must be...", ...}}
```

---

This visual guide complements the other two detailed documents (REQUEST_FLOW_TRACE.md and REQUEST_CODE_EXECUTION.md).
