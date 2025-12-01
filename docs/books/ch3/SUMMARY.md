# vLLM Serve Code Flow - Executive Summary

## Overview
When you execute `vllm serve --model gpt2`, a sophisticated system orchestrates the loading of the model onto GPU memory, sets up an OpenAI-compatible REST API, and establishes continuous inference capabilities. This document summarizes what happens at each stage.

---

## The 7-Stage Startup Process

### Stage 1: Command Parsing (< 1ms)
**File**: `vllm/entrypoints/cli/main.py`
- System parses `serve` as the subcommand
- Extracts `--model gpt2` and other CLI arguments
- Routes to `ServeSubcommand.cmd()`

### Stage 2: Server Preparation (< 10ms)
**File**: `vllm/entrypoints/cli/serve.py`
- Decides execution mode (single/multi-GPU, headless)
- For single GPU (default): calls `uvloop.run(run_server(args))`

### Stage 3: Socket Binding (< 10ms)
**File**: `vllm/entrypoints/openai/api_server.py`
- Creates TCP socket and binds to `0.0.0.0:8000`
- Sets signal handlers for graceful shutdown
- Pre-binds socket to avoid race conditions with Ray

### Stage 4: Model & Configuration Loading (2-30 seconds) ⚠️ **CRITICAL**
**File**: `vllm/engine/arg_utils.py`
- **Downloads/loads gpt2 from Hugging Face Hub** (or from cache)
- Creates `VllmConfig` containing:
  - `ModelConfig`: vocab size (50257), hidden size (768), layers (12)
  - `ParallelConfig`: tensor/pipeline parallel strategy
  - `CacheConfig`: KV cache allocation (adjusts with GPU memory)
  - `SchedulerConfig`: max concurrent requests (256), context length

### Stage 5: Engine Initialization (1-10 seconds) ⚠️ **CRITICAL**
**File**: `vllm/v1/engine/async_llm.py`
- **Loads all gpt2 model weights to GPU memory** (~500MB for gpt2)
- Creates:
  - Tokenizer
  - Input preprocessor (handles tokenization and multimodal inputs)
  - Processor (request scheduling and batching)
  - Output processor (decoding and formatting)
  - Engine loop (continuous inference task)

### Stage 6: FastAPI & Handler Setup (< 100ms)
**File**: `vllm/entrypoints/openai/api_server.py`
- Creates FastAPI application
- Registers 20+ endpoints:
  - `/v1/completions` - Text generation
  - `/v1/chat/completions` - Chat API
  - `/v1/embeddings` - Embedding generation
  - `/v1/audio/transcriptions` - Speech-to-text
  - And more...
- Initializes serving handlers:
  - `OpenAIServingCompletion` - Handles completion requests
  - `OpenAIServingChat` - Handles chat requests
  - `OpenAIServingTokenization` - Handles tokenization

### Stage 7: HTTP Server Start (< 100ms)
**File**: `vllm/entrypoints/launcher.py`
- Starts Uvicorn HTTP server with uvloop (high-performance async event loop)
- Listens on `0.0.0.0:8000`
- ✅ **READY TO ACCEPT REQUESTS**

**Total Startup Time**: 5-60 seconds (mostly Stage 4-5)

---

## Request Processing Flow (Once Running)

When a client sends a completion request:

```
HTTP Request → FastAPI Router → OpenAIServingCompletion 
    → Tokenize input → Create SamplingParams 
    → AsyncLLM.generate() 
    → Processor batches request
    → Engine loop executes forward pass
    → Executor runs model on GPU
    → Output processor decodes tokens
    → Format as CompletionResponse
    → HTTP Response → Client
```

**Latency**: ~50-200ms (depending on context length and parameters)

---

## Architecture Overview

```
User Command
     │
     ├─ CLI Module (parses arguments)
     │
     ├─ Config Module (loads model details)
     │
     ├─ Engine Module (handles inference)
     │   ├─ AsyncLLM (main engine)
     │   ├─ Processor (scheduling)
     │   └─ Executor (GPU execution)
     │
     ├─ API Module (HTTP endpoints)
     │   ├─ FastAPI app
     │   ├─ OpenAI serving handlers
     │   └─ Request/Response formatters
     │
     └─ HTTP Server (Uvicorn)
         └─ Listens on port 8000
```

---

## Key Files & Their Roles

| Component | Primary File | Secondary Files | Responsibility |
|-----------|--------------|-----------------|-----------------|
| **CLI Entry** | `cli/main.py` | - | Parse `vllm serve` command |
| **Serve Command** | `cli/serve.py` | - | Route to appropriate server mode |
| **API Server** | `openai/api_server.py` | - | HTTP server setup & endpoints |
| **Config Builder** | `engine/arg_utils.py` | `config/` | Create configs, load model from HF |
| **Engine** | `v1/engine/async_llm.py` | `v1/engine/core.py` | Manage inference pipeline |
| **Request Processing** | `v1/engine/processor.py` | - | Queue, batch, schedule requests |
| **Model Execution** | `v1/executor/` | `model_executor/` | GPU forward pass, sampling |
| **API Handlers** | `openai/serving_*.py` | - | OpenAI compatibility layer |
| **HTTP Launch** | `launcher.py` | - | Start Uvicorn server |

---

## How It Serves Requests

### Single Request Example: `POST /v1/completions`

**Request**:
```json
{
  "model": "gpt2",
  "prompt": "Once upon a time",
  "max_tokens": 50,
  "temperature": 0.7
}
```

**Processing Steps**:

1. **FastAPI** receives request and routes to `/v1/completions` handler
2. **Request Validation** checks JSON schema and required fields
3. **OpenAIServingCompletion** (the handler):
   - Tokenizes prompt: "Once upon a time" → [4, 201, 1626, 257, 640]
   - Creates `SamplingParams` with temperature=0.7, max_tokens=50
   - Calls `engine_client.generate(tokens, sampling_params)`
4. **AsyncLLM Engine**:
   - Preprocesses tokens
   - Enqueues request in processor
   - Waits for outputs
5. **Processor/Scheduler**:
   - Groups compatible requests into batches
   - Allocates KV cache
   - Sends to executor
6. **Executor** (GPU):
   - Loads batch onto GPU
   - Runs GPT-2 forward pass (12 transformer layers)
   - Generates logits for next token
   - Applies temperature scaling
   - Samples token using top-p/temperature
   - Updates KV cache
   - Repeats for up to 50 tokens
7. **Output Processing**:
   - Decodes tokens back to text
   - Formats with completion metadata
8. **Response Formatting**:
   - Creates `CompletionResponse` with:
     - id: unique completion ID
     - object: "text_completion"
     - model: "gpt2"
     - choices: [{text, finish_reason, index}]
     - usage: {prompt_tokens, completion_tokens}
9. **FastAPI** returns as JSON response

**Response**:
```json
{
  "id": "cmpl-8XXXX",
  "object": "text_completion",
  "created": 1703001234,
  "model": "gpt2",
  "choices": [{
    "text": " in a land far far away, there lived a king and a queen. The king had a very large castle",
    "index": 0,
    "finish_reason": "length",
    "logprobs": null
  }],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 25,
    "total_tokens": 30
  }
}
```

---

## Memory & Performance Characteristics

### For GPT-2 (124M parameters):

**Memory Usage**:
- Model weights: ~500MB
- KV cache (256 concurrent sequences): ~3GB
- Inference buffers: ~1-2GB
- Overhead: ~500MB
- **Total: ~5-6GB GPU memory**

**Throughput**:
- Single request: ~50-100ms latency
- Batched requests: ~100-200ms (amortized lower per request)
- Peak throughput: ~1000+ tokens/second (on RTX 3090)

**Concurrency**:
- Default: 256 concurrent sequences
- Adjustable via `--max-num-seqs`
- Sequences with same length batched together for efficiency

---

## Configuration Options

You can customize the server startup with CLI flags:

```bash
# Model and computing
vllm serve --model gpt2 \
    --tensor-parallel-size 1 \           # Number of GPUs (model parallelism)
    --pipeline-parallel-size 1 \         # Pipeline parallelism
    --gpu-memory-utilization 0.9 \       # How much GPU memory to use (0-1)
    --max-model-len 2048 \              # Maximum context length
    --max-num-seqs 256                  # Max concurrent sequences

# Server configuration
vllm serve --model gpt2 \
    --host 0.0.0.0 \                    # Listen address
    --port 8000 \                       # Listen port
    --api-key sk-xxxx \                 # API authentication key
    --ssl-keyfile key.pem \             # HTTPS support
    --ssl-certfile cert.pem

# Performance tuning
vllm serve --model gpt2 \
    --enable-prefix-caching \           # Cache prompt prefixes
    --disable-log-stats \               # Reduce logging overhead
    --disable-uvicorn-access-log \      # Reduce HTTP logging
    --uvicorn-log-level warning         # Quieter logging
```

---

## Lifecycle States

```
START
  │
  ├─ [Loading Model] ◄─── (downloading from HF if needed)
  │
  ├─ [Initializing Engine] ◄─── (moving weights to GPU)
  │
  ├─ [Ready] ◄─── ✅ Accepting requests
  │   │
  │   ├─ [Processing Request 1]
  │   │
  │   ├─ [Processing Request 2]
  │   │
  │   └─ [Processing Request N]
  │
  ├─ [Shutdown Signal] ◄─── (Ctrl+C or SIGTERM)
  │
  ├─ [Draining] ◄─── (finishing in-flight requests)
  │
  └─ [Stopped] ◄─── Exited
```

---

## Common Modifications & Extensions

You can extend vLLM by:

1. **Adding Custom Models**: Load from local paths or other sources
2. **Custom Serving Handlers**: Create new OpenAIServing* classes
3. **Custom Executors**: Implement model-specific optimizations
4. **Middleware**: Add authentication, logging, rate limiting
5. **Plugins**: Load LoRA adapters, custom quantization, etc.

---

## Performance Optimization Techniques Used

1. **Batching**: Groups requests for better GPU utilization
2. **Prefix Caching**: Reuses KV cache for repeated prompts
3. **Continuous Batching**: Doesn't wait for all requests to complete
4. **Page Attention**: Efficient KV cache management
5. **Tensor Parallelism**: Splits model across multiple GPUs
6. **uvloop**: High-performance async event loop
7. **CUDA Graphs**: Pre-compiled CUDA kernels for better performance

---

## Monitoring & Debugging

The server provides several endpoints for monitoring:

```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/v1/models

# Get version
curl http://localhost:8000/version

# Prometheus metrics (if enabled)
curl http://localhost:8000/metrics

# Server info (development mode)
curl http://localhost:8000/server_info
```

---

## Common Issues & Solutions

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| OOM Error | Too many sequences or GPU too small | Reduce `--max-num-seqs` or `--gpu-memory-utilization` |
| Slow first request | Model loading not complete | Wait for "Ready" message in logs |
| Port already in use | Another process on 8000 | Use `--port <other>` or stop other service |
| Model download fails | No internet or HF token needed | Check internet or provide HF token via env |
| API timeouts | Server overloaded | Reduce batch size or add more GPUs |
| Memory leak | Engine not releasing memory | Check for stuck requests or restart server |

---

## Conclusion

The `vllm serve --model gpt2` command initiates a sophisticated system that:

1. **Loads** the gpt2 model from Hugging Face
2. **Allocates** GPU memory intelligently
3. **Creates** a scalable inference engine
4. **Provides** OpenAI-compatible REST API
5. **Processes** concurrent requests efficiently
6. **Streams** responses in real-time

All of this happens through careful coordination of CLI parsing, configuration management, engine initialization, and HTTP server setup—allowing high-throughput, low-latency language model serving.

---

**For Detailed Information**:
- See `CODE_FLOW_TRACE.md` for complete code flow
- See `QUICK_REFERENCE.md` for quick lookup
- See `ARCHITECTURE_DIAGRAMS.md` for visual representations
