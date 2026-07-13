from __future__ import annotations

import base64
import json
import random
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    Checkpoint,
    CheckpointMetadata,
    PendingWrite,
    SerializerProtocol,
)

# Redis key layout. All checkpoint state for a thread lives under keys
# namespaced by thread_id, so `delete_thread` only needs to know the set of
# checkpoint_ns values that thread has ever used.
CHECKPOINT_KEY_PREFIX = "checkpoint"
CHECKPOINT_WRITES_KEY_PREFIX = "checkpoint_writes"
CHECKPOINT_INDEX_KEY_PREFIX = "checkpoints_idx"
CHECKPOINT_NS_INDEX_KEY_PREFIX = "checkpoint_ns_idx"
CHECKPOINT_THREADS_KEY = "checkpoint_threads"

# Sentinel used in place of `None` for the parent_checkpoint_id hash field,
# since Redis hash fields cannot store a Python `None`.
_NO_PARENT = ""


def checkpoint_key(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
    return f"{CHECKPOINT_KEY_PREFIX}:{thread_id}:{checkpoint_ns}:{checkpoint_id}"


def checkpoint_writes_key(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
    return f"{CHECKPOINT_WRITES_KEY_PREFIX}:{thread_id}:{checkpoint_ns}:{checkpoint_id}"


def checkpoint_index_key(thread_id: str, checkpoint_ns: str) -> str:
    return f"{CHECKPOINT_INDEX_KEY_PREFIX}:{thread_id}:{checkpoint_ns}"


def checkpoint_ns_index_key(thread_id: str) -> str:
    return f"{CHECKPOINT_NS_INDEX_KEY_PREFIX}:{thread_id}"


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64decode(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def dump_typed(serde: SerializerProtocol, obj: Any) -> tuple[str, str]:
    """Serialize `obj` to a (type, base64-text) pair suitable for a Redis hash field."""
    type_, data = serde.dumps_typed(obj)
    return type_, _b64encode(data)


def load_typed(serde: SerializerProtocol, type_: str, data: str) -> Any:
    return serde.loads_typed((type_, _b64decode(data)))


def dump_checkpoint_hash(
    serde: SerializerProtocol,
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
) -> dict[str, str]:
    from langgraph.checkpoint.base import get_checkpoint_metadata

    checkpoint_type, checkpoint_data = dump_typed(serde, checkpoint)
    metadata_type, metadata_data = dump_typed(serde, get_checkpoint_metadata(config, metadata))
    parent_checkpoint_id = config["configurable"].get("checkpoint_id") or _NO_PARENT
    return {
        "checkpoint_type": checkpoint_type,
        "checkpoint": checkpoint_data,
        "metadata_type": metadata_type,
        "metadata": metadata_data,
        "parent_checkpoint_id": parent_checkpoint_id,
    }


def load_checkpoint_hash(
    serde: SerializerProtocol, hash_: dict[str, str]
) -> tuple[Checkpoint, CheckpointMetadata, str | None]:
    checkpoint = load_typed(serde, hash_["checkpoint_type"], hash_["checkpoint"])
    metadata = load_typed(serde, hash_["metadata_type"], hash_["metadata"])
    parent_checkpoint_id = hash_.get("parent_checkpoint_id") or None
    return checkpoint, metadata, parent_checkpoint_id


def dump_write(
    serde: SerializerProtocol, task_id: str, idx: int, channel: str, value: Any, task_path: str
) -> str:
    type_, data = dump_typed(serde, value)
    return json.dumps(
        {
            "task_id": task_id,
            "idx": idx,
            "channel": channel,
            "type": type_,
            "value": data,
            "task_path": task_path,
        }
    )


def load_write(serde: SerializerProtocol, raw: str) -> tuple[int, str, PendingWrite]:
    """Returns (idx, task_id, PendingWrite); the idx/task_id let callers sort
    writes into a deterministic order, since Redis hashes make no ordering
    guarantee across HSET calls (unlike a Python dict's insertion order)."""
    payload = json.loads(raw)
    value = load_typed(serde, payload["type"], payload["value"])
    return (payload["idx"], payload["task_id"], (payload["task_id"], payload["channel"], value))


def write_field(task_id: str, idx: int) -> str:
    return f"{task_id}:{idx}"


def new_writes_fields(
    writes: list[tuple[str, Any]], task_id: str, existing_fields: set[str]
) -> list[tuple[str, int, str, Any]]:
    """Compute (channel, idx, field, value) for writes that should actually be persisted.

    Mirrors `InMemorySaver.put_writes`: writes at a non-negative index are
    write-once (skipped if already present), while writes at the reserved
    negative indices (errors, interrupts, etc., see `WRITES_IDX_MAP`) always
    overwrite the previous value.
    """
    result = []
    for idx, (channel, value) in enumerate(writes):
        real_idx = WRITES_IDX_MAP.get(channel, idx)
        field = write_field(task_id, real_idx)
        if real_idx >= 0 and field in existing_fields:
            continue
        result.append((channel, real_idx, field, value))
    return result


def next_version(current: str | int | None) -> str:
    """Default version scheme shared with `InMemorySaver`: zero-padded monotonic
    counter plus a random suffix to disambiguate concurrent writers."""
    if current is None:
        current_v = 0
    elif isinstance(current, int):
        current_v = current
    else:
        current_v = int(current.split(".")[0])
    next_v = current_v + 1
    next_h = random.random()
    return f"{next_v:032}.{next_h:016}"


def filter_metadata(metadata: CheckpointMetadata, filter: dict[str, Any] | None) -> bool:
    if not filter:
        return True
    return all(metadata.get(k) == v for k, v in filter.items())
