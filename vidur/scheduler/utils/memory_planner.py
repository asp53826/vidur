from math import ceil

from vidur.config import ReplicaConfig
from vidur.entities.replica import Replica
from vidur.utils.param_counter import ParamCounter


class MemoryPlanner:
    def __init__(self, replica_config: ReplicaConfig, replica: Replica) -> None:
        self._param_counter = ParamCounter(replica_config)
        self._replica = replica

        # A draft model takes weight memory and keeps its own KV cache, so it
        # shrinks how many requests fit. Charged to every device, which matches
        # vLLM sizing the cache by the most constrained worker.
        spec_config = replica_config.speculative_decoding_config
        self._draft_replica_config = None
        if spec_config.enabled and spec_config.proposer == "draft_model":
            self._draft_replica_config = ReplicaConfig(
                model_name=spec_config.draft_model_name,
                tensor_parallel_size=spec_config.draft_tensor_parallel_size,
                num_pipeline_stages=1,
                device=replica_config.device,
                network_device=replica_config.network_device,
                memory_margin_fraction=replica_config.memory_margin_fraction,
            )

    def _get_draft_parameter_memory_per_device(self) -> int:
        if self._draft_replica_config is None:
            return 0
        return 2 * ParamCounter(
            self._draft_replica_config
        ).get_num_parameters_per_device()

    def _get_draft_kv_cache_memory_per_device_per_request(self) -> int:
        if self._draft_replica_config is None:
            return 0
        draft = self._draft_replica_config.model_config
        head_dim = draft.embedding_dim // draft.num_q_heads
        kv_heads = ceil(
            draft.num_kv_heads / self._draft_replica_config.tensor_parallel_size
        )
        return (
            2  # 2 bytes per float
            * 2  # key and value
            * head_dim
            * kv_heads
            * self._replica.max_request_tokens
            * draft.num_layers
        )

    def _get_kv_cache_memory_per_layer_per_request(self) -> int:
        return (
            2  # 2 bytes per float
            * 2  # one for key, one for value
            * self._replica.attention_head_dim
            * self._replica.kv_heads_per_tensor_parallel_worker
            * self._replica.max_request_tokens
        )

    def _get_parameter_memory_per_device(self) -> int:
        return 2 * self._param_counter.get_num_parameters_per_device()

    def _get_kv_cache_memory_per_device_per_request(self) -> int:
        return (
            self._get_kv_cache_memory_per_layer_per_request() * self._replica.num_layers
        )

    def get_max_batch_size(self) -> int:
        available_memory = (
            self._replica.total_memory_gb
            * 1024**3
            * (1 - self._replica.memory_margin_fraction)
        )
        parameter_memory_per_device = self._get_parameter_memory_per_device()
        kv_cache_memory_per_device_per_request = (
            self._get_kv_cache_memory_per_device_per_request()
        )

        memory_for_kv_cache = (
            available_memory
            - parameter_memory_per_device
            - self._get_draft_parameter_memory_per_device()
        )
        kv_cache_memory_per_device_per_request += (
            self._get_draft_kv_cache_memory_per_device_per_request()
        )
        number_of_requests = (
            memory_for_kv_cache // kv_cache_memory_per_device_per_request
        )

        assert (
            number_of_requests > 0
        ), "Not enough memory to store even a single request"

        return number_of_requests

    def get_max_request_slots(self) -> int:
        return self.get_max_batch_size() * self._replica.num_pipeline_stages
