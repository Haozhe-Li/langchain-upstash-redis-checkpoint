from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, List

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    SerializerProtocol,
    get_checkpoint_id,
)

from upstash_redis import Redis  # type: ignore[attr-defined]

from .base import (
    CHECKPOINT_THREADS_KEY,
    checkpoint_index_key,
    checkpoint_key,
    checkpoint_ns_index_key,
    checkpoint_writes_key,
    dump_checkpoint_hash,
    dump_write,
    filter_metadata,
    load_checkpoint_hash,
    load_write,
    new_writes_fields,
    next_version,
)


class UpstashRedisSaver(BaseCheckpointSaver[str]):
    """A LangGraph checkpoint saver backed by [Upstash Redis](https://upstash.com/docs/redis)
    over its REST API.

    Unlike the Redis Stack-based `langgraph-redis` package, this saver does not rely on
    RediSearch or RedisJSON (neither of which Upstash Redis supports). Checkpoints are
    stored as plain Redis hashes and indexed with sorted sets, so it works against any
    standard Upstash Redis database.

    Example:
        ```python
        from langgraph.checkpoint.upstash_redis import UpstashRedisSaver

        saver = UpstashRedisSaver.from_env()
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "thread-1"}}
        graph.invoke(inputs, config)
        ```
    """

    def __init__(
        self,
        client: Redis,
        *,
        ttl: int | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Args:
        client: An `upstash_redis.Redis` client instance.
        ttl: If set, checkpoints (and their indices) expire after this many seconds.
        serde: Optional custom serializer; defaults to LangGraph's `JsonPlusSerializer`.
        """
        super().__init__(serde=serde)
        self.client = client
        self.ttl = ttl

    @classmethod
    def from_conn_info(cls, *, url: str, token: str, ttl: int | None = None) -> UpstashRedisSaver:
        """Create a saver from an Upstash REST URL and token."""
        return cls(Redis(url=url, token=token), ttl=ttl)

    @classmethod
    def from_env(cls, *, ttl: int | None = None) -> UpstashRedisSaver:
        """Create a saver using `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN`
        environment variables."""
        return cls(Redis.from_env(), ttl=ttl)

    def get_next_version(self, current: str | None, channel: None) -> str:
        return next_version(current)

    # -- reads --------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        if checkpoint_id is None:
            ids = self.client.zrangebylex(checkpoint_index_key(thread_id, checkpoint_ns), "-", "+")
            if not ids:
                return None
            checkpoint_id = ids[-1]

        hash_ = self.client.hgetall(checkpoint_key(thread_id, checkpoint_ns, checkpoint_id))
        if not hash_:
            return None

        checkpoint, metadata, parent_checkpoint_id = load_checkpoint_hash(self.serde, hash_)
        pending_writes = self._get_pending_writes(thread_id, checkpoint_ns, checkpoint_id)

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        thread_ids = [config["configurable"]["thread_id"]] if config else self._all_thread_ids()
        config_checkpoint_ns = config["configurable"].get("checkpoint_ns") if config else None
        config_checkpoint_id = get_checkpoint_id(config) if config else None
        before_checkpoint_id = get_checkpoint_id(before) if before else None

        remaining = limit
        for thread_id in thread_ids:
            namespaces = (
                [config_checkpoint_ns]
                if config_checkpoint_ns is not None
                else self.client.smembers(checkpoint_ns_index_key(thread_id))
            )
            for checkpoint_ns in namespaces:
                ids = self.client.zrangebylex(checkpoint_index_key(thread_id, checkpoint_ns), "-", "+")
                for checkpoint_id in reversed(ids):
                    if remaining is not None and remaining <= 0:
                        return
                    if config_checkpoint_id and checkpoint_id != config_checkpoint_id:
                        continue
                    if before_checkpoint_id and checkpoint_id >= before_checkpoint_id:
                        continue

                    hash_ = self.client.hgetall(
                        checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
                    )
                    if not hash_:
                        continue
                    checkpoint, metadata, parent_checkpoint_id = load_checkpoint_hash(
                        self.serde, hash_
                    )
                    if not filter_metadata(metadata, filter):
                        continue

                    if remaining is not None:
                        remaining -= 1

                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_id,
                            }
                        },
                        checkpoint=checkpoint,
                        metadata=metadata,
                        parent_config=(
                            {
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_ns": checkpoint_ns,
                                    "checkpoint_id": parent_checkpoint_id,
                                }
                            }
                            if parent_checkpoint_id
                            else None
                        ),
                        pending_writes=self._get_pending_writes(thread_id, checkpoint_ns, checkpoint_id),
                    )

    def _get_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        writes_hash = self.client.hgetall(
            checkpoint_writes_key(thread_id, checkpoint_ns, checkpoint_id)
        )
        if not writes_hash:
            return []
        decoded = sorted(
            (load_write(self.serde, raw) for raw in writes_hash.values()),
            key=lambda item: (item[0], item[1]),
        )
        return [pending_write for _, _, pending_write in decoded]

    def _all_thread_ids(self) -> List[str]:
        return self.client.smembers(CHECKPOINT_THREADS_KEY)

    # -- writes ---------------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        hash_ = dump_checkpoint_hash(self.serde, config, checkpoint, metadata)
        key = checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        index_key = checkpoint_index_key(thread_id, checkpoint_ns)

        pipe = self.client.pipeline()
        pipe.hset(key, values=hash_)
        pipe.zadd(index_key, {checkpoint_id: 0})
        pipe.sadd(checkpoint_ns_index_key(thread_id), checkpoint_ns)
        pipe.sadd(CHECKPOINT_THREADS_KEY, thread_id)
        if self.ttl is not None:
            pipe.expire(key, self.ttl)
            pipe.expire(index_key, self.ttl)
        pipe.exec()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        key = checkpoint_writes_key(thread_id, checkpoint_ns, checkpoint_id)

        existing_fields = set(self.client.hgetall(key).keys())
        to_write = new_writes_fields(list(writes), task_id, existing_fields)
        if not to_write:
            return

        values = {
            field: dump_write(self.serde, task_id, idx, channel, value, task_path)
            for channel, idx, field, value in to_write
        }
        pipe = self.client.pipeline()
        pipe.hset(key, values=values)
        if self.ttl is not None:
            pipe.expire(key, self.ttl)
        pipe.exec()

    def delete_thread(self, thread_id: str) -> None:
        namespaces = self.client.smembers(checkpoint_ns_index_key(thread_id))
        pipe = self.client.pipeline()
        for checkpoint_ns in namespaces:
            index_key = checkpoint_index_key(thread_id, checkpoint_ns)
            ids = self.client.zrangebylex(index_key, "-", "+")
            for checkpoint_id in ids:
                pipe.delete(checkpoint_key(thread_id, checkpoint_ns, checkpoint_id))
                pipe.delete(checkpoint_writes_key(thread_id, checkpoint_ns, checkpoint_id))
            pipe.delete(index_key)
        pipe.delete(checkpoint_ns_index_key(thread_id))
        pipe.srem(CHECKPOINT_THREADS_KEY, thread_id)
        pipe.exec()
