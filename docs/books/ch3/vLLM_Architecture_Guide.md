# vLLM V1 Architecture Guide

## Overview

vLLM V1 is a high-throughput inference engine optimized for distributed deployment with multiple parallelism strategies. This guide covers the core architecture, execution modes, and how requests flow through the system.

---

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [Distributed Execution](#distributed-execution)
3. [Offline vs Online Modes](#offline-vs-online-modes)
4. [EngineCore & Client Pattern](#enginecore--client-pattern)
5. [Code Execution Paths](#code-execution-paths)
6. [Continuous Batching](#continuous-batching)
7. [Memory Management](#memory-management)

---

## Core Architecture

### EngineCore: The Heart of vLLM

**File**: `vllm/v1/engine/core.py`

The `EngineCore` class is the central computation engine responsible for:
- **Request Management**: Queuing and processing inference requests
- **Scheduling**: Continuous batching via the Scheduler component
- **Execution**: Coordinating model execution via Executor
- **KV Cache Management**: Managing key-value cache for efficient inference
- **Output Collection**: Gathering results from all workers

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""
    
    def __init__(self, vllm_config, executor_class, log_stats):
        self.model_executor = executor_class(vllm_config)
        self.scheduler = Scheduler(vllm_config, ...)
        self.structured_output_manager = StructuredOutputManager(...)
        
    def step_fn(self) -> EngineCoreOutputs:
        """Main execution loop - one inference step"""
        scheduler_output = self.scheduler.schedule()
        model_output = self.model_executor.execute_model(scheduler_output)
        return self.process_model_output(model_output)
```

### Key Components

#### 1. **Scheduler** (`vllm/v1/core/sched/scheduler.py`)
- Implements continuous batching
- Decides which requests to batch in current iteration
- Respects memory constraints (`max_num_scheduled_tokens`)
- Manages request state transitions

#### 2. **Executor** (`vllm/v1/executor/abstract.py`)
- Coordinates model execution across workers
- Broadcasts `SchedulerOutput` to all workers via `collective_rpc()`
- Aggregates results from workers

#### 3. **Workers** (`vllm/worker/worker_base.py`)
- Execute model forward pass on assigned device (GPU/TPU)
- Participate in collective operations (all_reduce, all_gather)
- Manage device memory and KV cache

#### 4. **KV Cache Manager** (`vllm/v1/core/kv_cache_utils.py`)
- Allocates GPU memory for KV cache (paged attention)
- Tracks block allocation and deallocation
- Supports sequence-level and token-level caching

---

## Distributed Execution

### Multi-Dimensional Parallelism

vLLM supports orthogonal parallelism dimensions forming a Cartesian product:

```
World Layout: ExternalDP × DP × PP × TP × EP

Where:
- ExternalDP: External data parallelism (multiple independent engines)
- DP: Data parallelism (synchronized within single engine)
- PP: Pipeline parallelism (sequential GPU stages)
- TP: Tensor parallelism (split model weights across GPUs)
- EP: Expert parallelism (for MoE models)
```

### Group Formation

**File**: `vllm/distributed/parallel_state.py` (lines 1132-1139)

Groups are formed via tensor reshaping creating orthogonal process groups:

```python
all_ranks = torch.arange(world_size).reshape(
    -1, data_parallel_size, pipeline_model_parallel_size,
    tensor_model_parallel_size
)

# TP Group (transpose last dim and unbind)
tp_groups = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)

# PP Group (transpose TP to end, reshape, unbind)
pp_groups = all_ranks.transpose(2, 3).reshape(
    -1, pipeline_model_parallel_size
).unbind(0)

# DP Group (transpose TP to end, reshape, unbind)
dp_groups = all_ranks.transpose(1, 3).reshape(
    -1, data_parallel_size
).unbind(0)
```

### Worker Count Calculation

For a given configuration:
- **Number of Workers per Executor**: `TP × PP`
- **Number of Executors**: `DP × ExternalDP`

**Example**: `TP=2, PP=2, DP=2, ExternalDP=1`
- Workers per executor: 2 × 2 = 4
- Total executors: 2 × 1 = 2
- Total workers: 4 × 2 = 8

### Collective Operations

**Control Plane**: RPC (Remote Procedure Call) via `collective_rpc()`
```python
def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
    """Broadcast method call to all workers, collect results"""
    output = self.collective_rpc("execute_model", args=(scheduler_output,))
    return output[0]  # Driver rank only
```

**Data Plane**: Direct communication via NCCL (GPU) or Gloo (CPU)
- All-reduce for gradient synchronization
- All-gather for collecting distributed tensors
- Reduce-scatter for distributing data

---

## Offline vs Online Modes

### Offline Mode (Batch Inference)

**Use Case**: Process pre-collected batches of prompts

**Entry Point**: `vllm/entrypoints/llm.py` - `LLM.generate()`

```python
# User code
llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
outputs = llm.generate(prompts, sampling_params)

# Internal flow
generate()
├─ _validate_and_add_requests()  # Add all prompts
│  └─ llm_engine.add_request() per prompt
└─ _run_engine()                 # Blocking loop
   ├─ while has_unfinished_requests():
   │  └─ llm_engine.step()
   └─ return sorted(outputs)
```

**Characteristics**:
- ✅ Blocking: `generate()` blocks until ALL requests complete
- ✅ Batch Processing: All prompts added before execution
- ✅ Synchronous: Single-threaded control
- ✅ Optimal Throughput: Best GPU utilization with batching

### Online Mode (Streaming API)

**Use Case**: Serve multiple concurrent clients via HTTP API

**Entry Point**: `vllm/entrypoints/openai/api_server.py` - FastAPI handlers

```python
# HTTP request
POST /v1/chat/completions

# Internal flow
async create_chat_completion(request)
├─ await engine_client.add_request_async()  # Non-blocking
│  └─ Returns immediately
└─ StreamingResponse(chat_completion_stream_generator())
   ├─ async for output in engine_client.get_output_stream():
   │  └─ Yield token to HTTP client
   └─ yield "[DONE]"
```

**Characteristics**:
- ✅ Non-Blocking: Returns immediately to caller
- ✅ Streaming: Yields results token-by-token
- ✅ Asynchronous: Concurrent request handling
- ✅ Dynamic: Requests arrive independently

### Comparison Table

| Feature | Offline | Online |
|---------|---------|--------|
| **Concurrency** | Sequential batches | Multiple concurrent |
| **Request Arrival** | All upfront | Streaming |
| **Blocking** | Caller blocks | Returns immediately |
| **Engine Control** | Caller-driven | Engine self-driven |
| **Response Mode** | Complete list | Streaming chunks |
| **Framework** | Pure Python | FastAPI + asyncio |
| **Latency Pattern** | Batch latency | Per-token latency |
| **Client Type** | Same process | Remote HTTP |

---

## EngineCore & Client Pattern

### Architecture

```
┌─────────────────────────────────┐
│      EngineCore (Shared)        │
│  • Scheduler (batching)         │
│  • Executor (inference)         │
│  • KV Cache management          │
│  • Request loop                 │
└─────────────────────────────────┘
            ▲
    ┌───────┼───────┬─────────────┐
    │       │       │             │
InprocClient │  AsyncMPClient  (other)
             │
        SyncMPClient
```

**Key Insight**: EngineCore logic is **shared** across all modes. The difference is in **how clients communicate** with EngineCore.

### Client Implementations

**File**: `vllm/v1/engine/core_client.py`

#### 1. **InprocClient** (Offline - Single Process)

```python
class InprocClient(EngineCoreClient):
    """In-process EngineCore for offline mode"""
    
    def __init__(self, vllm_config, executor_class, log_stats):
        self.engine_core = EngineCore(vllm_config, executor_class, log_stats)
    
    def add_request(self, request: EngineCoreRequest) -> None:
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)
    
    def get_output(self) -> EngineCoreOutputs:
        outputs, _ = self.engine_core.step_fn()  # Direct call
        return outputs and outputs.get(0) or EngineCoreOutputs()
```

**Communication**: Direct Python function calls (no IPC)

#### 2. **SyncMPClient** (Offline - Multiprocess)

```python
class SyncMPClient(MPClient):
    """Background EngineCore with ZMQ for distributed offline"""
    
    def __init__(self, vllm_config, executor_class, log_stats):
        # Launches background process running EngineCore
        self.engine_process = launch_core_engines(...)
        self.zmq_socket = make_zmq_socket(...)  # ZMQ communication
    
    def add_request(self, request: EngineCoreRequest) -> None:
        self.zmq_socket.send(request)  # Send via ZMQ
    
    def get_output(self) -> EngineCoreOutputs:
        return self.zmq_socket.recv()  # Receive via ZMQ (blocking)
```

**Communication**: ZMQ (TCP sockets) + background process

#### 3. **AsyncMPClient** (Online - Async)

```python
class AsyncMPClient(MPClient):
    """Background EngineCore with ZMQ for async online serving"""
    
    def __init__(self, vllm_config, executor_class, log_stats, ...):
        # Launches background process running EngineCore
        self.engine_process = launch_core_engines(...)
        self.zmq_socket = make_zmq_socket(...)  # ZMQ communication
    
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        await self.zmq_socket.send_async(request)  # Non-blocking send
    
    async def get_output_async(self) -> AsyncIterator[EngineCoreOutputs]:
        async for output in self.zmq_socket.recv_async():
            yield output  # Streaming receive
```

**Communication**: Async ZMQ + background process + asyncio event loop

#### 4. **Data-Parallel Variants**

- **DPAsyncMPClient**: Multiple EngineCore instances, external load balancing
- **DPLBAsyncMPClient**: Multiple EngineCore instances, internal load balancing

### Client Selection Logic

```python
@staticmethod
def make_client(
    multiprocess_mode: bool,
    asyncio_mode: bool,
    ...
) -> "EngineCoreClient":
    
    if multiprocess_mode and asyncio_mode:
        return EngineCoreClient.make_async_mp_client(...)  # Online API
    
    if multiprocess_mode and not asyncio_mode:
        return SyncMPClient(...)  # Offline distributed
    
    return InprocClient(...)  # Offline single-process
```

---

## Code Execution Paths

### Offline Execution Path

**File**: `vllm/entrypoints/llm.py`

```
1. User calls: llm.generate(prompts, sampling_params)
   └─ Location: LLM.__init__() creates LLMEngine

2. LLMEngine.from_engine_args()
   └─ Auto-selects V0 or V1 engine
   └─ Creates EngineCoreClient (usually InprocClient)

3. LLM.generate()
   ├─ _validate_and_add_requests()
   │  └─ Validates prompts and params
   │  └─ For each prompt:
   │     └─ _add_request()
   │        └─ llm_engine.add_request()
   │           └─ client.add_request(EngineCoreRequest)
   │              └─ engine_core.add_request() [InprocClient]
   │
   └─ _run_engine()
      ├─ Initialize progress bar
      └─ while llm_engine.has_unfinished_requests():
         └─ step_outputs = llm_engine.step()
            └─ client.get_output()
               └─ engine_core.step_fn() [InprocClient]
                  ├─ scheduler.schedule() → SchedulerOutput
                  ├─ executor.execute_model(scheduler_output)
                  │  └─ collective_rpc("execute_model", args=(scheduler_output,))
                  │     └─ All workers execute in parallel
                  └─ process_model_output()
         └─ For each finished output:
            └─ outputs.append(output)
      └─ return sorted(outputs, key=request_id)

4. Return list[RequestOutput] to caller
```

### Online Execution Path

**Files**: `vllm/entrypoints/openai/api_server.py`, `serving_chat.py`

```
1. HTTP Client: POST /v1/chat/completions
   └─ FastAPI route handler

2. create_chat_completion(request)
   ├─ Validate model and parameters
   ├─ Get tokenizer
   └─ request_id = await engine_client.add_request_async(
        prompt=prompt_tokens,
        sampling_params=sampling_params
      )
      └─ client.add_request_async(EngineCoreRequest)
         └─ ZMQ send to background process [AsyncMPClient]
         └─ Returns immediately to caller

3. Return StreamingResponse(chat_completion_stream_generator(...))
   └─ HTTP 200 OK sent immediately

4. Async generator: chat_completion_stream_generator()
   ├─ result_generator = await engine_client.get_output_stream(request_id)
   └─ async for output in result_generator:
      ├─ engine_client.get_output_async()
      │  └─ ZMQ recv from background process [AsyncMPClient]
      └─ For each token in output:
         └─ yield formatted chunk to HTTP client
   └─ yield "[DONE]"

5. Background Process (EngineCore loop):
   ├─ while True:  # Continuous loop
   │  ├─ Poll incoming requests from ZMQ queue
   │  └─ step_fn()
   │     ├─ scheduler.schedule() → SchedulerOutput
   │     ├─ executor.execute_model(scheduler_output)
   │     └─ Send outputs to clients via ZMQ

6. Results stream to client in real-time
   └─ Tokens arrive as they're generated
```

---

## Continuous Batching

### Three-Layer Architecture

```
┌─────────────────────────────────┐
│ Layer 1: Scheduler              │ Decides WHAT to batch
│ - Request queue                 │
│ - Memory constraints            │
│ - Produces SchedulerOutput      │
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Layer 2: Executor               │ Coordinates execution
│ - Broadcasts SchedulerOutput    │
│ - Collects results              │
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Layer 3: Workers                │ Executes model
│ - Forward pass                  │
│ - Collective ops (TP, PP, DP)   │
│ - Returns ModelRunnerOutput     │
└─────────────────────────────────┘
```

### Scheduling Example

```
Iteration 1:
  Pending: [req0, req1, req2, req3, req4]
  Scheduled: [req0, req1, req2, req3, req4]
  In-flight: {req0, req1, req2, req3, req4}
  
Iteration 2:
  Pending: [req5, req6]
  Scheduled: [req1, req2, req3, req4, req5]  # req0 done
  In-flight: {req1, req2, req3, req4, req5}
  
Iteration 3:
  Pending: [req7]
  Scheduled: [req2, req3, req4, req5, req6]  # req1 done
  In-flight: {req2, req3, req4, req5, req6}
```

**Key Feature**: Requests dynamically join/leave batch as they complete, enabling:
- **Higher GPU utilization**: No waiting for slowest request
- **Lower latency**: Fast requests finish immediately
- **Memory efficiency**: Process only active requests

---

## Memory Management

### PagedAttention

**Implementation Files**:
- GPU: `csrc/attention/paged_attention_v1.cu`, `paged_attention_v2.cu`
- CPU: `csrc/cpu/attention.cpp`
- Header: `csrc/attention/attention_kernels.cuh`

**Core Concepts**:
- KV cache divided into fixed-size blocks (typically 16 tokens/block)
- Blocks allocated/freed independently (no fragmentation)
- Non-contiguous block pointers supported
- Enables efficient context reuse and prefix caching

### KV Cache Configuration

**File**: `vllm/v1/core/kv_cache_utils.py`

```python
kv_cache_config = get_kv_cache_configs(
    vllm_config=vllm_config,
    num_gpu_blocks=num_gpu_blocks,
    num_cpu_blocks=num_cpu_blocks
)

# Tracks:
# - Block size
# - Block hash (for prefix matching)
# - Supported models/dtypes
# - Cache policies
```

### Block Allocation

```
GPU Memory Layout:
┌─────────────────────────┐
│ Model Weights           │ Fixed
├─────────────────────────┤
│ Activations             │ Dynamic (batch size)
├─────────────────────────┤
│ KV Cache Blocks         │ Managed by scheduler
│ [B0][B1][B2][B3][B4]... │ Free blocks
└─────────────────────────┘

Request req0:
├─ Prefill tokens: 0-15 → [B0]
├─ Generate: token 16 → [B1]
├─ Generate: token 17 → [B2]
└─ Block pointers: [0, 1, 2]

Request req1:
├─ Prefill tokens: 0-10 → [B3]
├─ Generate: token 11 → [B4]
└─ Block pointers: [3, 4]
```

---

## Request Lifecycle

### State Transitions

```
                   add_request()
                        │
                        ▼
┌─────────────────────────────────┐
│ WAITING                         │
│ • In request queue              │
│ • Waiting for scheduling        │
└─────────────────────────────────┘
                │
                ▼ schedule()
┌─────────────────────────────────┐
│ RUNNING                         │
│ • In current batch              │
│ • Being processed               │
│ • Accumulating tokens           │
└─────────────────────────────────┘
                │
                ├─ token == eos_token_id?
                │      └─ abort_request()
                │           ↓
                │      ABORTED
                │
                ├─ len(tokens) >= max_tokens?
                │      └─ FINISHED (length_finish)
                │
                └─ always_stop() == True?
                       └─ FINISHED (stop)
                
                       FINISHED (normal)
```

### Memory Cleanup

When request finishes:
1. KV cache blocks deallocated
2. Request removed from scheduler
3. Output sent to client
4. Blocks available for reuse

---

## Performance Characteristics

### Throughput Optimization

| Parallelism | Purpose | Throughput Impact |
|-------------|---------|------------------|
| **TP** | Reduce per-GPU memory | Modest (reduces batch size) |
| **PP** | Enable large models | Minor (pipelining reduces bubbles) |
| **DP** | Scale to more GPUs | Linear (each GPU adds capacity) |
| **Batching** | Amortize prefill | Quadratic (better utilization) |

### Latency Characteristics

```
Time-to-First-Token (TTFT):
  = Prefill time / (prefill parallelism)
  = max(TP collective ops, prefill computation)
  + network latency (if distributed)

Time-Per-Output-Token (TPOT):
  = Decode time / (decode parallelism)
  ≈ constant (independent of sequence length)
  
Total Latency:
  = TTFT + (num_output_tokens × TPOT)
```

### Memory Requirements

```
Model Weights:
  = model_size / (TP × PP parallelism)

KV Cache per GPU:
  = (num_gpu_blocks × block_size × num_heads × head_dim × bytes_per_token)
  
Activations (batch):
  = (batch_size × seq_len × hidden_dim × bytes_per_param)
  
Total per GPU:
  = Weights + KV Cache + Activations
```

---

## Key Files Reference

| Component | File | Purpose |
|-----------|------|---------|
| **EngineCore** | `vllm/v1/engine/core.py` | Main execution loop |
| **Clients** | `vllm/v1/engine/core_client.py` | Communication abstraction |
| **Scheduler** | `vllm/v1/core/sched/scheduler.py` | Continuous batching logic |
| **Executor** | `vllm/v1/executor/abstract.py` | Execution coordination |
| **Workers** | `vllm/worker/worker_base.py` | Device execution |
| **KV Cache** | `vllm/v1/core/kv_cache_utils.py` | Memory management |
| **Offline API** | `vllm/entrypoints/llm.py` | Batch inference |
| **Online API** | `vllm/entrypoints/openai/api_server.py` | REST API server |
| **Parallel State** | `vllm/distributed/parallel_state.py` | Group coordination |
| **PagedAttention** | `csrc/attention/paged_attention_*.cu` | Optimized attention kernels |

---

## Conclusion

vLLM V1 achieves high throughput through:

1. **Shared EngineCore**: Common computation engine for all modes
2. **Pluggable Clients**: Different communication strategies (in-proc, ZMQ async)
3. **Continuous Batching**: Dynamic request mixing for optimal utilization
4. **Multi-Dimensional Parallelism**: Orthogonal groups enabling distributed execution
5. **PagedAttention**: Efficient KV cache memory management
6. **Collective Operations**: Coordinated execution across workers

The architecture supports both offline batch processing and online streaming APIs with a single core implementation, reducing code duplication and maintenance burden.

