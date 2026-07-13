"""Unit tests for pure key-building / serialization helpers. No network access required."""

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph.checkpoint.upstash_redis.base import (
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

SERDE = JsonPlusSerializer()


def test_key_builders():
    assert checkpoint_key("t1", "", "c1") == "checkpoint:t1::c1"
    assert checkpoint_writes_key("t1", "ns", "c1") == "checkpoint_writes:t1:ns:c1"
    assert checkpoint_index_key("t1", "ns") == "checkpoints_idx:t1:ns"
    assert checkpoint_ns_index_key("t1") == "checkpoint_ns_idx:t1"


def test_checkpoint_hash_roundtrip():
    config = {
        "configurable": {
            "thread_id": "t1",
            "checkpoint_ns": "",
            "checkpoint_id": "parent-id",
        }
    }
    checkpoint = {
        "v": 1,
        "id": "child-id",
        "ts": "2024-01-01T00:00:00+00:00",
        "channel_values": {"messages": ["hi"]},
        "channel_versions": {"messages": "1"},
        "versions_seen": {},
    }
    metadata = {"source": "loop", "step": 1}

    hash_ = dump_checkpoint_hash(SERDE, config, checkpoint, metadata)
    assert set(hash_.keys()) == {
        "checkpoint_type",
        "checkpoint",
        "metadata_type",
        "metadata",
        "parent_checkpoint_id",
    }
    assert hash_["parent_checkpoint_id"] == "parent-id"

    loaded_checkpoint, loaded_metadata, parent_id = load_checkpoint_hash(SERDE, hash_)
    assert loaded_checkpoint == checkpoint
    assert loaded_metadata["source"] == "loop"
    assert loaded_metadata["step"] == 1
    assert parent_id == "parent-id"


def test_checkpoint_hash_no_parent():
    config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "id": "root-id",
        "ts": "2024-01-01T00:00:00+00:00",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
    }
    hash_ = dump_checkpoint_hash(SERDE, config, checkpoint, {})
    assert hash_["parent_checkpoint_id"] == ""
    _, _, parent_id = load_checkpoint_hash(SERDE, hash_)
    assert parent_id is None


def test_write_roundtrip():
    raw = dump_write(SERDE, "task-1", 0, "messages", {"role": "user", "content": "hi"}, "path")
    idx, task_id, pending_write = load_write(SERDE, raw)
    assert idx == 0
    assert task_id == "task-1"
    assert pending_write == ("task-1", "messages", {"role": "user", "content": "hi"})


def test_new_writes_fields_dedups_regular_writes():
    writes = [("channel_a", "v1"), ("channel_b", "v2")]
    result = new_writes_fields(writes, "task-1", existing_fields=set())
    fields = {field for _, _, field, _ in result}
    assert fields == {"task-1:0", "task-1:1"}

    # Already-persisted regular writes are skipped on a retry.
    result_again = new_writes_fields(writes, "task-1", existing_fields=fields)
    assert result_again == []


def test_new_writes_fields_overwrites_special_channels():
    from langgraph.checkpoint.base import ERROR

    existing_fields = {f"task-1:{-1}"}  # ERROR maps to idx -1
    result = new_writes_fields([(ERROR, "boom")], "task-1", existing_fields=existing_fields)
    assert len(result) == 1
    channel, idx, field, value = result[0]
    assert channel == ERROR
    assert idx == -1
    assert value == "boom"


def test_next_version_monotonic():
    v1 = next_version(None)
    v2 = next_version(v1)
    v3 = next_version(v2)
    assert v1 < v2 < v3


def test_filter_metadata():
    metadata = {"source": "loop", "step": 2}
    assert filter_metadata(metadata, None) is True
    assert filter_metadata(metadata, {}) is True
    assert filter_metadata(metadata, {"source": "loop"}) is True
    assert filter_metadata(metadata, {"source": "input"}) is False
    assert filter_metadata(metadata, {"source": "loop", "step": 2}) is True
    assert filter_metadata(metadata, {"source": "loop", "step": 3}) is False
