# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
from collections.abc import AsyncGenerator, Mapping
from copy import copy
from typing import Any, Optional, Union

import numpy as np

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.envs import VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
from vllm.inputs import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Device, cdiv
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.output_processor import (OutputProcessor,
                                             RequestOutputCollector)
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.v1.metrics.prometheus import shutdown_prometheus
from vllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)


class AsyncLLM(EngineClient):

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
        client_index: int = 0,
    ) -> None:
        """
        Create an AsyncLLM.

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            use_cached_outputs: Whether to use cached outputs.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        print("[TRACE] (async_llm.py) AsyncLLM.__init__() called - initializing V1 async engine")
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()
        print("[TRACE] (async_llm.py) Registered transformer configs for serialization")

        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.log_requests = log_requests
        self.log_stats = log_stats
        print(f"[TRACE] (async_llm.py) Configured model: {self.model_config.model}, log_requests: {self.log_requests}, log_stats: {self.log_stats}")

        if self.model_config.skip_tokenizer_init:
            print("[TRACE] (async_llm.py) Skipping tokenizer initialization (skip_tokenizer_init=True)")
            self.tokenizer = None
        else:
            # Tokenizer (+ ensure liveness if running in another process).
            print("[TRACE] (async_llm.py) Initializing tokenizer from configs")
            self.tokenizer = init_tokenizer_from_configs(
                model_config=vllm_config.model_config,
                scheduler_config=vllm_config.scheduler_config,
                lora_config=vllm_config.lora_config)
            print("[TRACE] (async_llm.py) Tokenizer initialization completed")

        # Processor (converts Inputs --> EngineCoreRequests).
        print("[TRACE] (async_llm.py) Creating Processor for input conversion")
        self.processor = Processor(
            vllm_config=vllm_config,
            tokenizer=self.tokenizer,
            mm_registry=mm_registry,
        )
        print("[TRACE] (async_llm.py) Processor created successfully")

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        print("[TRACE] (async_llm.py) Creating OutputProcessor for output processing")
        self.output_processor = OutputProcessor(self.tokenizer,
                                                log_stats=self.log_stats)
        print("[TRACE] (async_llm.py) OutputProcessor created successfully")

        # EngineCore (starts the engine in background process).
        print("[TRACE] (async_llm.py) Creating EngineCoreClient for multiprocess backend")
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_index=client_index,
        )
        print("[TRACE] (async_llm.py) EngineCoreClient created, engine ranks managed: {0}".format(self.engine_core.engine_ranks_managed))

        # Loggers.
        self.logger_manager: Optional[StatLoggerManager] = None
        if self.log_stats:
            print("[TRACE] (async_llm.py) Creating StatLoggerManager for stats logging")
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=stat_loggers,
            )
            self.logger_manager.log_engine_initialized()
            print("[TRACE] (async_llm.py) StatLoggerManager initialized and engine logged")
        else:
            print("[TRACE] (async_llm.py) Skipping stats logging (log_stats=False)")

        self.output_handler: Optional[asyncio.Task] = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            print("[TRACE] (async_llm.py) Event loop detected, starting output handler")
            self._run_output_handler()
        except RuntimeError:
            print("[TRACE] (async_llm.py) No event loop running yet, output handler will be started on first generate() call")

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
        client_addresses: Optional[dict[str, str]] = None,
        client_index: int = 0,
    ) -> "AsyncLLM":
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=not disable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_index=client_index,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shutdown, cleaning up the background proc and IPC."""

        shutdown_prometheus()

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown()

        if handler := getattr(self, "output_handler", None):
            handler.cancel()

    async def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""

        print(f"[TRACE] (async_llm.py) add_request() called - request_id: {request_id}, priority: {priority}")
        if self.errored:
            print(f"[TRACE] (async_llm.py) Engine is errored, raising EngineDeadError for request_id: {request_id}")
            raise EngineDeadError()

        is_pooling = isinstance(params, PoolingParams)
        print(f"[TRACE] (async_llm.py) Request type: {'pooling' if is_pooling else 'generation'}")

        # Create a new output collector for the request.
        queue = RequestOutputCollector(output_kind=params.output_kind)
        print(f"[TRACE] (async_llm.py) Created RequestOutputCollector for request_id: {request_id}")

        # Convert Input --> Request.
        print(f"[TRACE] (async_llm.py) Processing inputs for request_id: {request_id}")
        prompt_str, request = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            tokenization_kwargs, trace_headers, priority, data_parallel_rank)
        print(f"[TRACE] (async_llm.py) Processed inputs, prompt_tokens: {len(prompt_str.split()) if prompt_str else 0}")

        if is_pooling or params.n == 1:
            print(f"[TRACE] (async_llm.py) Single request case (pooling={is_pooling} or n=1), adding request directly")
            await self._add_request(request, prompt_str, None, 0, queue)
            return queue

        # Fan out child requests (for n>1).
        print(f"[TRACE] (async_llm.py) Fanning out {params.n} child requests for request_id: {request_id}")
        parent_request = ParentRequest(request_id, params)
        for idx in range(params.n):
            request_id, params = parent_request.get_child_info(idx)
            child_request = request if idx == params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = params
            print(f"[TRACE] (async_llm.py) Adding child request {idx + 1}/{params.n}, request_id: {request_id}")
            await self._add_request(child_request, prompt_str, parent_request,
                                    idx, queue)
        print(f"[TRACE] (async_llm.py) Completed fanning out {params.n} child requests")
        return queue

    async def _add_request(self, request: EngineCoreRequest,
                           prompt: Optional[str],
                           parent_req: Optional[ParentRequest], index: int,
                           queue: RequestOutputCollector):

        print(f"[TRACE] (async_llm.py) _add_request() called - request_id: {request.request_id}, index: {index}, has_parent: {parent_req is not None}")
        # Add the request to OutputProcessor (this process).
        print(f"[TRACE] (async_llm.py) Adding request to OutputProcessor - request_id: {request.request_id}")
        self.output_processor.add_request(request, prompt, parent_req, index,
                                          queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        print(f"[TRACE] (async_llm.py) Adding request to EngineCore via async - request_id: {request.request_id}")
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)
        print(f"[TRACE] (async_llm.py) Request added successfully - request_id: {request.request_id}")

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        print(f"[TRACE] (async_llm.py) generate() called - request_id: {request_id}, temperature: {sampling_params.temperature}, top_p: {sampling_params.top_p}")
        try:
            # We start the output_handler on the first call to generate() so
            # we can call __init__ before the event loop, which enables us
            # to handle startup failure gracefully in the OpenAI server.
            print(f"[TRACE] (async_llm.py) Starting output handler for generation - request_id: {request_id}")
            self._run_output_handler()

            print(f"[TRACE] (async_llm.py) Adding request to engine - request_id: {request_id}")
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
            )
            print(f"[TRACE] (async_llm.py) Request added to engine, starting output streaming - request_id: {request_id}")

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            output_count = 0
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()

                output_count += 1
                print(f"[TRACE] (async_llm.py) Yielding output #{output_count} - request_id: {request_id}, finished: {out.finished}")
                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                yield out

            print(f"[TRACE] (async_llm.py) Generation complete - request_id: {request_id}, total outputs: {output_count}")

        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        except (asyncio.CancelledError, GeneratorExit):
            print(f"[TRACE] (async_llm.py) Generation cancelled or generator exiting - request_id: {request_id}")
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            print(f"[TRACE] (async_llm.py) Engine is dead - request_id: {request_id}")
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error.
        except ValueError:
            print(f"[TRACE] (async_llm.py) Request validation error - request_id: {request_id}")
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e

    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            print("[TRACE] (async_llm.py) Output handler already running, skipping creation")
            return

        print("[TRACE] (async_llm.py) Creating and starting output handler task")
        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        logger_manager = self.logger_manager

        async def output_handler():
            print("[TRACE] (async_llm.py) Output handler task started")
            try:
                iteration_num = 0
                while True:
                    iteration_num += 1
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    print(f"[TRACE] (async_llm.py) Output handler iteration #{iteration_num} - pulling outputs from EngineCore")
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)
                    print(f"[TRACE] (async_llm.py) Received {num_outputs} outputs from EngineCore (engine_index: {outputs.engine_index})")

                    iteration_stats = IterationStats() if (
                        log_stats and num_outputs) else None

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    if num_outputs <= VLLM_V1_OUTPUT_PROC_CHUNK_SIZE:
                        slices = (outputs.outputs, )
                        print(f"[TRACE] (async_llm.py) Processing outputs in single chunk")
                    else:
                        num_slices = cdiv(num_outputs, VLLM_V1_OUTPUT_PROC_CHUNK_SIZE)
                        slices = np.array_split(
                            outputs.outputs,
                            num_slices)
                        print(f"[TRACE] (async_llm.py) Processing {num_outputs} outputs in {num_slices} chunks")

                    for i, outputs_slice in enumerate(slices):
                        # 2) Process EngineCoreOutputs.
                        print(f"[TRACE] (async_llm.py) Processing chunk {i + 1}/{len(slices)} - {len(outputs_slice)} outputs")
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats)
                        # NOTE: RequestOutputs are pushed to their queues.
                        assert not processed_outputs.request_outputs

                        # Allow other asyncio tasks to run between chunks
                        if i + 1 < len(slices):
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        if processed_outputs.reqs_to_abort:
                            print(f"[TRACE] (async_llm.py) Aborting {len(processed_outputs.reqs_to_abort)} requests due to stop strings")
                            await engine_core.abort_requests_async(
                                processed_outputs.reqs_to_abort)

                    # 4) Logging.
                    # TODO(rob): make into a coroutine and launch it in
                    # background thread once Prometheus overhead is non-trivial.
                    if logger_manager:
                        logger_manager.record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                        )
            except Exception as e:
                print(f"[TRACE] (async_llm.py) Output handler exception: {type(e).__name__}: {e}")
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())
        print("[TRACE] (async_llm.py) Output handler task created and scheduled")

    async def abort(self, request_id: str) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        print(f"[TRACE] (async_llm.py) abort() called - request_id: {request_id}")
        request_ids = self.output_processor.abort_requests((request_id, ))
        print(f"[TRACE] (async_llm.py) Aborted in OutputProcessor, aborting in EngineCore - request_id: {request_id}")
        await self.engine_core.abort_requests_async(request_ids)

        if self.log_requests:
            logger.info("Aborted request %s.", request_id)
        print(f"[TRACE] (async_llm.py) Request abort completed - request_id: {request_id}")

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        print(f"[TRACE] (async_llm.py) encode() called - request_id: {request_id}, pooling_type: {pooling_params.pooling_type}")
        try:
            # We start the output_handler on the first call to generate() so
            # we can call __init__ before the event loop, which enables us
            # to handle startup failure gracefully in the OpenAI server.
            print(f"[TRACE] (async_llm.py) Starting output handler for encoding - request_id: {request_id}")
            self._run_output_handler()

            print(f"[TRACE] (async_llm.py) Adding pooling request to engine - request_id: {request_id}")
            q = await self.add_request(
                request_id,
                prompt,
                pooling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
                tokenization_kwargs=tokenization_kwargs,
            )
            print(f"[TRACE] (async_llm.py) Pooling request added to engine, starting output streaming - request_id: {request_id}")

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            output_count = 0
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()
                assert isinstance(out, PoolingRequestOutput)
                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                output_count += 1
                print(f"[TRACE] (async_llm.py) Yielding pooling output #{output_count} - request_id: {request_id}, finished: {out.finished}")
                finished = out.finished
                yield out

            print(f"[TRACE] (async_llm.py) Encoding complete - request_id: {request_id}, total outputs: {output_count}")

        # If the request is disconnected by the client, generate()
        # is cancelled. So, we abort the request if we end up here.
        except asyncio.CancelledError:
            print(f"[TRACE] (async_llm.py) Encoding cancelled - request_id: {request_id}")
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            print(f"[TRACE] (async_llm.py) Engine is dead during encoding - request_id: {request_id}")
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error.
        except ValueError:
            print(f"[TRACE] (async_llm.py) Request validation error during encoding - request_id: {request_id}")
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            print(f"[TRACE] (async_llm.py) Encoding error - request_id: {request_id}, error: {type(e).__name__}")
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e

    async def get_vllm_config(self) -> VllmConfig:
        return self.vllm_config

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def get_decoding_config(self):
        raise ValueError("Not Supported on V1 yet.")

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return self.processor.input_preprocessor

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        if self.tokenizer is None:
            raise ValueError("Unable to get tokenizer because "
                             "skip_tokenizer_init is True")

        return self.tokenizer.get_lora_tokenizer(lora_request)

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(
        self,
        scheduler_outputs=None,
        model_output=None,
    ) -> None:
        if self.logger_manager:
            self.logger_manager.log()

    async def check_health(self) -> None:
        logger.debug("Called check_health.")
        if self.errored:
            raise self.dead_error

    async def start_profile(self) -> None:
        await self.engine_core.profile_async(True)

    async def stop_profile(self) -> None:
        await self.engine_core.profile_async(False)

    async def reset_mm_cache(self) -> None:
        self.processor.mm_registry.reset_processor_cache()
        self.processor.mm_input_cache_client.reset()
        await self.engine_core.reset_mm_cache_async()

    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        if device == Device.CPU:
            raise ValueError("Not supported on CPU.")
        await self.engine_core.reset_prefix_cache_async()

    async def sleep(self, level: int = 1) -> None:
        await self.engine_core.sleep_async(level)

    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        await self.engine_core.wake_up_async(tags)

    async def is_sleeping(self) -> bool:
        return await self.engine_core.is_sleeping_async()

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return await self.engine_core.add_lora_async(lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return await self.engine_core.remove_lora_async(lora_id)

    async def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return await self.engine_core.list_loras_async()

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return await self.engine_core.pin_lora_async(lora_id)

    async def collective_rpc(self,
                             method: str,
                             timeout: Optional[float] = None,
                             args: tuple = (),
                             kwargs: Optional[dict] = None):
        """
        Perform a collective RPC call to the given path.
        """
        return await self.engine_core.collective_rpc_async(
            method, timeout, args, kwargs)

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        """Wait for all requests to be drained."""
        start_time = time.time()
        while time.time() - start_time < drain_timeout:
            if not self.engine_core.dp_engines_running():
                logger.info("Engines are idle, requests have been drained")
                return

            logger.info(
                "Engines are still running, waiting for requests to drain...")
            await asyncio.sleep(1)  # Wait 1 second before checking again

        raise TimeoutError(f"Timeout reached after {drain_timeout} seconds "
                           "waiting for requests to drain.")

    async def scale_elastic_ep(self,
                               new_data_parallel_size: int,
                               drain_timeout: int = 300):
        """
        Scale up or down the data parallel size by adding or removing
        engine cores.
        Args:
            new_data_parallel_size: The new number of data parallel workers
            drain_timeout:
                Maximum time to wait for requests to drain (seconds)
        """
        old_data_parallel_size = \
            self.vllm_config.parallel_config.data_parallel_size
        if old_data_parallel_size == new_data_parallel_size:
            logger.info("Data parallel size is already %s, skipping scale",
                        new_data_parallel_size)
            return
        logger.info(
            "Waiting for requests to drain before "
            "scaling up to %s engines...", new_data_parallel_size)
        await self.wait_for_requests_to_drain(drain_timeout)
        logger.info(
            "Requests have been drained, proceeding with scale "
            "to %s engines", new_data_parallel_size)
        await self.engine_core.scale_elastic_ep(new_data_parallel_size)
        self.vllm_config.parallel_config.data_parallel_size = \
            new_data_parallel_size

        # recreate stat loggers
        if new_data_parallel_size > old_data_parallel_size and self.log_stats:
            # TODO(rob): fix this after talking with Ray team.
            # This resets all the prometheus metrics since we
            # unregister during initialization. Need to understand
            # the intended behavior here better.
            self.logger_manager = StatLoggerManager(
                vllm_config=self.vllm_config,
                engine_idxs=list(range(new_data_parallel_size)),
                custom_stat_loggers=None,
            )

    @property
    def is_running(self) -> bool:
        # Is None before the loop is started.
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return self.engine_core.resources.engine_dead or not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()
