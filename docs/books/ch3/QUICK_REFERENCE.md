# vLLM `serve` Command - Quick Reference

## Command Execution Path

```
vllm serve --model gpt2
    ↓
main() [cli/main.py]
    ↓
ServeSubcommand.cmd() [cli/serve.py]
    ↓
uvloop.run(run_server(args))
    ↓
run_server() → run_server_worker() [openai/api_server.py]
    ├─ setup_server() - Bind socket to 0.0.0.0:8000
    ├─ build_async_engine_client() - Initialize AsyncLLM engine
    ├─ build_app() - Create FastAPI application
    ├─ init_app_state() - Setup serving handlers
    └─ serve_http() - Start uvicorn server
    
    Server Ready → Accept HTTP requests on port 8000
```

---

## Critical Initialization Steps (In Order)

### 1. **CLI Parsing** (< 1ms)
   - File: `vllm/entrypoints/cli/main.py:main()`
   - Input: `serve --model gpt2`
   - Output: `args.model = "gpt2"`, `args.subparser = "serve"`

### 2. **Argument Conversion** (< 1ms)
   - File: `vllm/engine/arg_utils.py:AsyncEngineArgs.from_cli_args()`
   - Converts CLI args → `AsyncEngineArgs` object
   - Extracts: model, tensor_parallel_size, gpu_memory_fraction, etc.

### 3. **Engine Config Creation** (~1-5s) ⚠️ SLOW
   - File: `vllm/engine/arg_utils.py:AsyncEngineArgs.create_engine_config()`
   - **MODEL LOADING HAPPENS HERE**:
     - Downloads gpt2 from Hugging Face Hub (if not cached)
     - Loads model config (vocab_size=50257, hidden_size=768, etc.)
     - Sets up tokenizer
     - Allocates GPU memory for KV cache
     - Determines parallel strategy

### 4. **AsyncLLM Engine Creation** (~2-10s) ⚠️ SLOW
   - File: `vllm/v1/engine/async_llm.py:AsyncLLM.__init__()`
   - Creates:
     - Input preprocessor
     - Tokenizer
     - Processor (request queuing)
     - Output processor
     - Engine loop task
   - **MODEL WEIGHTS LOADING HAPPENS HERE**

### 5. **FastAPI Application Setup** (< 100ms)
   - File: `vllm/entrypoints/openai/api_server.py:build_app()`
   - Registers endpoints:
     - POST /v1/completions
     - POST /v1/chat/completions
     - POST /v1/embeddings
     - POST /v1/audio/transcriptions
     - GET /v1/models
     - GET /health
     - etc.

### 6. **App State Initialization** (< 100ms)
   - File: `vllm/entrypoints/openai/api_server.py:init_app_state()`
   - Creates serving handlers:
     - `OpenAIServingCompletion` - Text generation
     - `OpenAIServingChat` - Chat API
     - `OpenAIServingEmbedding` - Embeddings
     - `OpenAIServingTokenization` - Tokenization

### 7. **HTTP Server Startup** (< 100ms)
   - File: `vllm/entrypoints/launcher.py:serve_http()`
   - Starts uvicorn on 0.0.0.0:8000
   - **Ready to accept requests**

---

## Data Flow: From Request to Response

### Example: `POST /v1/completions`

```
HTTP Request (JSON)
    ↓
FastAPI Router matches endpoint
    ↓
create_completion() [api_server.py]
    ↓
OpenAIServingCompletion.create_completion() [openai/serving_completion.py]
    ├─ Validate request
    ├─ Tokenize prompt
    ├─ Create SamplingParams
    └─ Call engine_client.generate()
    ↓
AsyncLLM.generate() [v1/engine/async_llm.py]
    ├─ Preprocess inputs
    ├─ Queue request in processor
    └─ Wait for outputs
    ↓
Engine Core Loop (runs continuously)
    ├─ Collect requests from queue
    ├─ Batch together
    ├─ Call executor
    └─ Stream outputs
    ↓
Executor.execute() [v1/executor/*]
    ├─ Load batch to GPU
    ├─ Run model forward pass
    ├─ Sample tokens
    └─ Return token IDs
    ↓
Output Processor
    ├─ Decode tokens to text
    └─ Stream to API handler
    ↓
API Handler formats response
    ↓
HTTP Response (JSON/SSE)
    ↓
Client receives completion
```

---

## Configuration at Startup

**For `vllm serve --model gpt2`:**

| Setting | Value | Source |
|---------|-------|--------|
| Model | gpt2 | CLI arg |
| Device | CUDA (if available, else CPU) | Auto-detect |
| GPU Memory | 90% of available | Default |
| Tensor Parallel | 1 | Auto-detect |
| Pipeline Parallel | 1 | Default |
| Block Size | 16 tokens | Default |
| Max Concurrent Requests | 256 | Default |
| Max Context Length | Model's max | Auto-detect |
| Port | 8000 | Default |
| Host | 0.0.0.0 | Default |

---

## File Organization

```
vllm/
├── entrypoints/
│   ├── cli/
│   │   ├── main.py              ← Entry point for `vllm serve`
│   │   └── serve.py             ← ServeSubcommand implementation
│   ├── openai/
│   │   ├── api_server.py        ← HTTP server setup
│   │   ├── cli_args.py          ← Argument parsing
│   │   ├── serving_completion.py ← Handles /v1/completions
│   │   ├── serving_chat.py      ← Handles /v1/chat/completions
│   │   └── ...
│   └── launcher.py              ← Uvicorn HTTP server
├── engine/
│   ├── arg_utils.py             ← AsyncEngineArgs, config creation
│   └── protocol.py              ← EngineClient interface
├── v1/
│   └── engine/
│       ├── async_llm.py         ← Main engine class
│       ├── core.py              ← Engine core
│       ├── processor.py         ← Request processor
│       ├── output_processor.py  ← Output handling
│       └── executor/
│           └── abstract.py      ← Executor interface
├── config/                      ← Model, cache, scheduler configs
├── model_executor/              ← Model weight loading
└── ...
```

---

## Important Environment Variables

```bash
# Engine version
VLLM_USE_V1=1                           # Use V1 engine (default)

# GPU Configuration
VLLM_GPU_MEMORY_UTILIZATION=0.9        # GPU memory usage
CUDA_VISIBLE_DEVICES=0                 # Which GPUs to use

# Logging
VLLM_LOG_LEVEL=INFO                    # Log level
VLLM_LOG_STATS_INTERVAL=10             # Stats logging frequency

# Performance
VLLM_HTTP_TIMEOUT_KEEP_ALIVE=5         # HTTP keep-alive timeout
VLLM_LOG_STATS_INTERVAL=10             # Log statistics interval

# Development
VLLM_SERVER_DEV_MODE=0                 # Enable debug endpoints
VLLM_DEBUG_LOG_API_SERVER_RESPONSE=0   # Log full responses
```

---

## Key Classes and Their Roles

| Class | File | Purpose |
|-------|------|---------|
| `ServeSubcommand` | cli/serve.py | CLI subcommand dispatcher |
| `AsyncEngineArgs` | engine/arg_utils.py | CLI arguments wrapper |
| `ModelConfig` | config/models.py | Model configuration |
| `ParallelConfig` | config/parallel.py | Distributed execution config |
| `VllmConfig` | config/model.py | All configs combined |
| `AsyncLLM` | v1/engine/async_llm.py | Main inference engine |
| `Processor` | v1/engine/processor.py | Request queue manager |
| `Executor` | v1/executor/abstract.py | GPU/CPU execution |
| `OpenAIServing*` | openai/serving_*.py | API handler classes |

---

## Typical Startup Timeline

```
0ms   : Command: vllm serve --model gpt2
1ms   : CLI parsing complete
2ms   : Arguments validated
3ms   : Socket bound to 0.0.0.0:8000
5ms   : Engine config creation started
        └─ Download gpt2 from HF (if not cached)  ← Can take 30+ seconds
        └─ Load model config
        └─ Allocate GPU memory

500ms : Engine config ready
600ms : AsyncLLM engine created
        └─ Load model weights to GPU              ← Can take 10-60 seconds
        └─ Allocate inference buffers
        
1500ms: Engine ready, starting HTTP server
1600ms: Uvicorn listening on 0.0.0.0:8000
        ✓ Ready to accept requests

First request: ~200ms (tokenization + first inference)
Subsequent requests: ~50-100ms (depending on length and parameters)
```

---

## Memory Layout

**GPU Memory for gpt2 (~124M parameters):**

```
┌─────────────────────────────────────┐
│ Model Weights (~500MB)              │  ← Loaded once
├─────────────────────────────────────┤
│ KV Cache (~3GB for 256 sequences)   │  ← Dynamic, shared pool
├─────────────────────────────────────┤
│ Inference Buffers (~1GB)            │  ← For computations
├─────────────────────────────────────┤
│ Overhead (~500MB)                   │  ← CUDA, allocator, etc.
└─────────────────────────────────────┘
  Total: ~5GB for gpt2 with 256 seq batches
```

---

## Request Batching Example

```
Request 1: "Hello"        (Tokens: 1)
Request 2: "Hi there"     (Tokens: 2)
Request 3: "How are you?" (Tokens: 3)

Waiting time:
├─ Request 1: Gets processed in batch 1 (0ms additional wait)
├─ Request 2: Queued, waits for batch 1 to finish (~50ms)
└─ Request 3: Queued, waits for batch 1,2 to finish (~100ms)

All three processed together if they can fit in batch (scheduler decides).
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM (Out of Memory) | Too many sequences or long context | Reduce `--max-num-seqs` or `--gpu-memory-utilization` |
| Slow startup | Downloading model weights | Use `--download-dir` to cache locally |
| Port already in use | Another service on 8000 | Use `--port <other_port>` |
| Connection refused | Server not ready | Wait for "Uvicorn running on" message |
| Errors loading model | Incompatible model format | Ensure model is in Hugging Face format |

---

## Testing the Server

```bash
# Start server
vllm serve --model gpt2

# In another terminal:

# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Simple completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "Hello, world!",
    "max_tokens": 10
  }'

# Streaming completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "Hello, world!",
    "max_tokens": 10,
    "stream": true
  }'
```

---

This is a condensed reference guide. See **CODE_FLOW_TRACE.md** for the detailed flow.
