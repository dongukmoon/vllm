# vLLM Architecture & Flow Diagrams

## 1. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      vLLM Serve                                │
└────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │   CLI    │  │   HTTP   │  │  Engine  │
        │ Parsing  │  │  Server  │  │  Core    │
        └──────────┘  └──────────┘  └──────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                ┌────────────▼────────────┐
                │   Model Executor       │
                │  (GPU/CPU Inference)   │
                └────────────────────────┘
```

---

## 2. Initialization Pipeline

```
COMMAND: vllm serve --model gpt2
│
├─[1] CLI Parsing (main.py)
│    ├─ Parse subcommand: "serve"
│    ├─ Parse arguments: --model gpt2
│    └─ Create Namespace object
│
├─[2] Setup & Validation (serve.py)
│    ├─ Detect server mode (single/multi/headless)
│    └─ Route to run_server()
│
├─[3] Socket Binding (api_server.py)
│    ├─ Create TCP socket
│    ├─ Bind to 0.0.0.0:8000
│    └─ Set signal handlers
│
├─[4] Engine Initialization (api_server.py)
│    ├─ Convert CLI args → AsyncEngineArgs
│    ├─ Create VllmConfig
│    │  ├─ Load model (gpt2) from Hugging Face
│    │  ├─ Get model config (vocab, hidden_size, etc.)
│    │  ├─ Setup parallel strategy
│    │  └─ Allocate GPU memory
│    └─ Create AsyncLLM instance
│       ├─ Initialize tokenizer
│       ├─ Create input preprocessor
│       ├─ Create processor (request queue)
│       ├─ Create output processor
│       └─ Start engine loop
│
├─[5] FastAPI Setup (api_server.py)
│    ├─ Create FastAPI app
│    ├─ Mount all endpoints
│    ├─ Add middleware (CORS, auth, metrics)
│    └─ Setup error handlers
│
├─[6] App State Initialization (api_server.py)
│    ├─ Store engine client
│    ├─ Create OpenAIServingCompletion
│    ├─ Create OpenAIServingChat
│    ├─ Create OpenAIServingEmbedding
│    ├─ Create OpenAIServingTokenization
│    └─ Create other serving handlers
│
└─[7] HTTP Server Start (launcher.py)
     ├─ Create Uvicorn config
     ├─ Start server with uvloop
     └─ ✓ READY: Listening on 0.0.0.0:8000
```

---

## 3. Request Processing Pipeline

```
CLIENT REQUEST
│
├─ HTTP POST /v1/completions
│  └─ Body: {"model": "gpt2", "prompt": "Hello", "max_tokens": 10}
│
├─[FastAPI Layer]
│  ├─ Router matches endpoint: create_completion()
│  └─ Deserialize: CompletionRequest object
│
├─[Serving Handler Layer]
│  ├─ OpenAIServingCompletion.create_completion()
│  ├─ Validate request
│  ├─ Check model availability
│  ├─ Tokenize prompt: "Hello" → [31373]
│  ├─ Create SamplingParams
│  │  ├─ temperature
│  │  ├─ top_p
│  │  ├─ max_tokens
│  │  └─ ...other params
│  └─ Call: engine_client.generate()
│
├─[Engine Layer]
│  ├─ AsyncLLM.generate()
│  ├─ Preprocess inputs
│  ├─ Enqueue request: EngineRequest
│  ├─ Wait for outputs
│  └─ Receive: RequestOutput (token IDs, tokens, logprobs)
│
├─[Processor Layer]
│  ├─ Processor dequeues requests
│  ├─ Groups compatible requests
│  ├─ Creates execution batch
│  └─ Submits to executor
│
├─[Executor Layer]
│  ├─ Load batch to GPU
│  ├─ Run model forward pass
│  │  ├─ Embedding layer
│  │  ├─ Transformer layers (12 for gpt2)
│  │  ├─ Output layer
│  │  └─ Logits: shape [batch_size, vocab_size]
│  ├─ Sample tokens from logits
│  ├─ Update KV cache
│  └─ Return: token IDs, logits (optional)
│
├─[Output Processor]
│  ├─ Decode token IDs to text
│  ├─ Compute log probabilities
│  ├─ Format output
│  └─ Stream to serving handler
│
├─[Response Formatting]
│  ├─ Create CompletionResponse object
│  ├─ Include:
│  │  ├─ id
│  │  ├─ object
│  │  ├─ created
│  │  ├─ model
│  │  ├─ choices: [{text, finish_reason, logprobs}]
│  │  └─ usage: {prompt_tokens, completion_tokens, total_tokens}
│  └─ Serialize to JSON
│
└─ HTTP Response (200 OK)
   └─ Body: {"id": "cmpl-xxx", "choices": [...], "usage": {...}}
```

---

## 4. Internal Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       AsyncLLM Engine                           │
│                  (vllm/v1/engine/async_llm.py)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Input Preprocessing                            │   │
│  │  - Tokenize prompts                                      │   │
│  │  - Handle multimodal inputs                             │   │
│  │  - Validate context length                              │   │
│  └────────────────┬──────────────────────────────────────┬──┘   │
│                   │                                      │       │
│  ┌────────────────▼──┐                    ┌────────────▼──┐    │
│  │  Request Queue    │                    │  Output Queue │    │
│  │  (FIFO)           │◄──────────────────►│  (Results)    │    │
│  └────────────────┬──┘                    └────────────▲──┘    │
│                   │                                    │        │
│  ┌────────────────▼──────────────────────────────────┐│        │
│  │        Processor / Scheduler                       ││        │
│  │  - Batch requests intelligently                   ││        │
│  │  - Sequence-level scheduling                      ││        │
│  │  - Token budget management                        ││        │
│  │  - KV cache allocation                            ││        │
│  └────────────────┬──────────────────────────────────┘│        │
│                   │                                    │        │
│  ┌────────────────▼────────────────────────────────────┐        │
│  │          Engine Core Loop                          │        │
│  │  - Run continuously                                │        │
│  │  - Collect batches from processor                 │        │
│  │  - Call executor.execute()                        │        │
│  │  - Handle outputs (decoding, streaming)           │        │
│  └────────────────┬──────────────────────────────────┐        │
│                   │                                  │        │
│  ┌────────────────▼──────────────────────────────────┐        │
│  │       Executor (GPU/CPU)                          │        │
│  │  - Load batch to device                           │        │
│  │  - Run model forward pass                         │        │
│  │  - Apply sampling (temperature, top-p, etc.)      │        │
│  │  - Return token IDs                               │        │
│  └──────────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Shared Resources:
├─ Model Weights (on GPU)
├─ KV Cache (dynamic pool)
├─ Tokenizer
├─ Config (model, parallel, scheduler, cache)
└─ Metrics & Logging
```

---

## 5. Parallel Execution Timeline

```
Request Timeline (3 concurrent requests, 3 tokens each):

Req1: "He"      Req2: "Hi t"    Req3: "How a"
│     │         │    │          │     │
│     ▼         │    ▼          │     ▼
│   [Batch 1: Req1+Req2+Req3 forward pass - 10ms]
│              │              │
│              ▼              ▼
│           Decode        Decode
│           "He" → "l"    "i t" → "h"    "ow a" → "re"
│              │              │
│   Iteration 2: [Another forward pass for next tokens]
│
Result (after ~30ms):
Req1: "Hello"       ✓
Req2: "Hi there"    ✓
Req3: "How are"     ✓

Key: All 3 requests processed in parallel through same forward pass!
```

---

## 6. Memory Management

```
GPU Memory Allocation:

┌────────────────────────────────────────────┐
│ TOTAL GPU MEMORY: ~8GB (for RTX 3090)      │
├────────────────────────────────────────────┤
│                                            │
│  [Model Weights]                           │
│  GPT-2: ~500MB                             │
│  ├─ Embeddings                             │
│  ├─ Transformer blocks (12)                │
│  └─ Output layer                           │
│                                            │
│  [KV Cache Pool]                           │
│  ~3-4GB (dynamic)                          │
│  ├─ Key caches: [batch, seq, heads, d]     │
│  ├─ Value caches: [batch, seq, heads, d]   │
│  └─ Reused across iterations                │
│                                            │
│  [Inference Buffers]                       │
│  ~1-2GB                                     │
│  ├─ Batch inputs                           │
│  ├─ Attention matrices                     │
│  ├─ Hidden states                          │
│  └─ Logits                                 │
│                                            │
│  [CUDA/System Overhead]                    │
│  ~500MB                                     │
│  ├─ CUDA context                           │
│  ├─ Allocator overhead                     │
│  └─ Miscellaneous                          │
│                                            │
│  [Free/Reserved]                           │
│  Remaining space                            │
│                                            │
└────────────────────────────────────────────┘

Configuration can be adjusted via:
- --gpu-memory-utilization (0.0-1.0)
- --max-model-len (context length)
- --max-num-seqs (concurrent sequences)
```

---

## 7. Configuration Flow

```
CLI Arguments
   ↓
┌─ --model gpt2
├─ --tensor-parallel-size 1
├─ --gpu-memory-utilization 0.9
├─ --max-model-len 2048
├─ --max-num-seqs 256
└─ --port 8000
   ↓
Namespace object (argparse)
   ↓
AsyncEngineArgs.from_cli_args()
   ↓
VllmConfig objects:
├─ ModelConfig
│  ├─ model_id
│  ├─ tokenizer_id
│  ├─ vocab_size
│  ├─ hidden_size
│  ├─ num_layers
│  ├─ num_attention_heads
│  ├─ dtype (float16, float32, bfloat16)
│  └─ quantization_scheme
│
├─ ParallelConfig
│  ├─ tensor_parallel_size
│  ├─ pipeline_parallel_size
│  ├─ data_parallel_size
│  ├─ world_size
│  └─ rank (for distributed execution)
│
├─ CacheConfig
│  ├─ block_size
│  ├─ num_blocks
│  ├─ cache_dtype
│  └─ gpu_memory_utilization
│
└─ SchedulerConfig
   ├─ max_num_seqs
   ├─ max_model_len
   ├─ max_padded_seq_len
   └─ scheduler_delay_factor
   
   ↓
AsyncLLM initialization with configs
   ↓
Ready for inference
```

---

## 8. Error Handling Flow

```
Request arrives
   ↓
Try:
├─ Parse JSON ✓
│  └─ If invalid JSON → HTTPException(400, "JSON decode error")
│
├─ Validate schema ✓
│  └─ If invalid → HTTPException(400, "Validation error")
│
├─ Check model exists ✓
│  └─ If not found → HTTPException(404, "Model not found")
│
├─ Check engine health ✓
│  └─ If dead → HTTPException(503, "Engine not responding")
│
├─ Process request ✓
│  └─ If error → HTTPException(500, "Internal error")
│
└─ Format response ✓
   └─ Success!

Except:
├─ HTTPException → Return error response with status code
├─ EngineDeadError → Return 503 Service Unavailable
├─ OverflowError → Return 400 Bad Request
├─ RequestValidationError → Return 422 Unprocessable Entity
└─ Generic Exception → Return 500 Internal Server Error

All errors include:
- HTTP status code
- Error message
- Error type
- Error code (for programmatic handling)
```

---

## 9. Streaming Response Pipeline

```
Streaming request:
POST /v1/completions
{...,"stream": true,...}
   ↓
create_completion(request)
   ↓
handler.create_completion(streaming=True)
   ↓
AsyncGenerator yields chunks:
├─ [chunk_1] ChatCompletionStreamResponse(choice.delta.content="Hello")
├─ [chunk_2] ChatCompletionStreamResponse(choice.delta.content=" world")
├─ [chunk_3] ChatCompletionStreamResponse(choice.delta.content="!")
├─ [chunk_4] ChatCompletionStreamResponse(choice.finish_reason="stop")
└─ [done] "[DONE]"
   ↓
StreamingResponse formats as Server-Sent Events (SSE):
├─ data: {"choices":[{"delta":{"content":"Hello"}}]}
├─ data: {"choices":[{"delta":{"content":" world"}}]}
├─ data: {"choices":[{"delta":{"content":"!"}}]}
├─ data: {"choices":[{"finish_reason":"stop"}]}
└─ data: [DONE]
   ↓
HTTP Response (Transfer-Encoding: chunked)
   ↓
Client receives streaming chunks in real-time
```

---

## 10. Multi-GPU / Tensor Parallel Example

```
Command: vllm serve --model gpt2 --tensor-parallel-size 2

GPU 0                          GPU 1
│                              │
Model Layer 1 (split)     Model Layer 1 (split)
├─ Heads 0-6               ├─ Heads 7-11
└─ Linear1, Linear2        └─ Linear1, Linear2
│                              │
All-Reduce (synchronize)
│◄─────────────────────────────►│
│                              │
Model Layer 2 (split)     Model Layer 2 (split)
├─ Heads 0-6               ├─ Heads 7-11
└─ Linear1, Linear2        └─ Linear1, Linear2
│                              │
...repeat for all layers...
│                              │
Output Logits                Output Logits
(combined)
   ↓
Final output

Benefits:
- Model parallelism: Each GPU handles subset of model
- Higher throughput: Process multiple batches in parallel
- Reduced latency: More compute per forward pass
```

---

## 11. Key Metrics During Operation

```
┌─ Request Metrics
│  ├─ Total requests processed
│  ├─ Requests per second (RPS)
│  ├─ Average latency
│  ├─ P50/P95/P99 latencies
│  └─ Error rate
│
├─ Throughput Metrics
│  ├─ Tokens per second
│  ├─ Tokens/sec/GPU
│  └─ Batch size (average)
│
├─ Resource Metrics
│  ├─ GPU memory used
│  ├─ GPU utilization %
│  ├─ GPU power usage
│  └─ CPU usage
│
├─ Engine Metrics
│  ├─ Queue length
│  ├─ Active sequences
│  ├─ KV cache usage
│  ├─ Engine loop iterations/sec
│  └─ Batch sizes
│
└─ Prometheus Metrics
   ├─ vllm_request_count
   ├─ vllm_request_duration
   ├─ vllm_prompt_tokens_total
   ├─ vllm_completion_tokens_total
   └─ vllm_token_stream_duration
```

---

## 12. Shutdown Sequence

```
Signal: SIGTERM or SIGINT (Ctrl+C)
   ↓
Signal handler triggered
   ↓
uvicorn.should_exit = True
   ↓
Stop accepting new requests
   ↓
Wait for in-flight requests to complete
   ├─ Timeout configurable (default: graceful)
   └─ Force kill after timeout
   ↓
Shutdown engine
├─ Stop engine loop
├─ Flush remaining outputs
├─ Cleanup GPU memory
└─ Close engine client
   ↓
Shutdown FastAPI
├─ Clear app state
├─ Close connections
└─ Cleanup resources
   ↓
Close socket
   ↓
Exit process
   ↓
✓ Clean shutdown complete
```

---

These diagrams show the complete architecture and execution flow of `vllm serve --model gpt2`.
