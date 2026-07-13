"""Integration tests against a real Upstash Redis database (async saver).

Require UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN; skipped otherwise.
"""

from langgraph.checkpoint.base import ERROR
from langgraph.checkpoint.base.id import uuid6

from langgraph.checkpoint.upstash_redis.aio import AsyncUpstashRedisSaver

from .conftest import requires_upstash


def make_checkpoint(step: int) -> dict:
    return {
        "v": 1,
        "id": str(uuid6(clock_seq=step)),
        "ts": "2024-01-01T00:00:00+00:00",
        "channel_values": {"count": step},
        "channel_versions": {"count": str(step)},
        "versions_seen": {},
    }


@requires_upstash
async def test_aput_and_aget_tuple(thread_id):
    saver = AsyncUpstashRedisSaver.from_env()
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint = make_checkpoint(0)
    metadata = {"source": "loop", "step": 0}

    saved_config = await saver.aput(config, checkpoint, metadata, {})
    assert saved_config["configurable"]["checkpoint_id"] == checkpoint["id"]

    tup = await saver.aget_tuple(saved_config)
    assert tup is not None
    assert tup.checkpoint["id"] == checkpoint["id"]
    assert tup.checkpoint["channel_values"] == {"count": 0}
    assert tup.metadata["source"] == "loop"
    assert tup.parent_config is None

    await saver.adelete_thread(thread_id)
    assert await saver.aget_tuple(saved_config) is None


@requires_upstash
async def test_aget_tuple_latest_and_parent_chain(thread_id):
    saver = AsyncUpstashRedisSaver.from_env()
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    checkpoint_0 = make_checkpoint(0)
    config = await saver.aput(config, checkpoint_0, {"source": "input", "step": -1}, {})

    checkpoint_1 = make_checkpoint(1)
    config = await saver.aput(config, checkpoint_1, {"source": "loop", "step": 0}, {})

    latest = await saver.aget_tuple({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}})
    assert latest is not None
    assert latest.checkpoint["id"] == checkpoint_1["id"]
    assert latest.parent_config["configurable"]["checkpoint_id"] == checkpoint_0["id"]

    await saver.adelete_thread(thread_id)


@requires_upstash
async def test_aput_writes_and_pending_writes(thread_id):
    saver = AsyncUpstashRedisSaver.from_env()
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint = make_checkpoint(0)
    config = await saver.aput(config, checkpoint, {"source": "loop", "step": 0}, {})

    await saver.aput_writes(config, [("channel_a", "value_a"), ("channel_b", "value_b")], "task-1")

    tup = await saver.aget_tuple(config)
    assert tup is not None
    pending = {channel: value for _, channel, value in tup.pending_writes}
    assert pending == {"channel_a": "value_a", "channel_b": "value_b"}

    await saver.aput_writes(
        config, [("channel_a", "value_a_retry"), ("channel_b", "value_b")], "task-1"
    )
    tup2 = await saver.aget_tuple(config)
    pending2 = {channel: value for _, channel, value in tup2.pending_writes}
    assert pending2["channel_a"] == "value_a"  # unchanged

    await saver.aput_writes(config, [(ERROR, "first error")], "task-2")
    await saver.aput_writes(config, [(ERROR, "second error")], "task-2")
    tup3 = await saver.aget_tuple(config)
    errors = [value for _, channel, value in tup3.pending_writes if channel == ERROR]
    assert errors == ["second error"]

    await saver.adelete_thread(thread_id)


@requires_upstash
async def test_alist_with_filter_before_and_limit(thread_id):
    saver = AsyncUpstashRedisSaver.from_env()
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    ids = []
    for step in range(4):
        checkpoint = make_checkpoint(step)
        ids.append(checkpoint["id"])
        source = "input" if step == 0 else "loop"
        config = await saver.aput(config, checkpoint, {"source": source, "step": step - 1}, {})

    base_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    all_checkpoints = [c async for c in saver.alist(base_config)]
    assert [c.checkpoint["id"] for c in all_checkpoints] == list(reversed(ids))

    limited = [c async for c in saver.alist(base_config, limit=2)]
    assert [c.checkpoint["id"] for c in limited] == list(reversed(ids))[:2]

    before = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "", "checkpoint_id": ids[-1]}}
    before_results = [c async for c in saver.alist(base_config, before=before)]
    assert [c.checkpoint["id"] for c in before_results] == list(reversed(ids[:-1]))

    filtered = [c async for c in saver.alist(base_config, filter={"source": "input"})]
    assert len(filtered) == 1
    assert filtered[0].checkpoint["id"] == ids[0]

    await saver.adelete_thread(thread_id)


@requires_upstash
async def test_adelete_thread_removes_everything(thread_id):
    saver = AsyncUpstashRedisSaver.from_env()
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint = make_checkpoint(0)
    config = await saver.aput(config, checkpoint, {"source": "loop", "step": 0}, {})
    await saver.aput_writes(config, [("channel_a", "value_a")], "task-1")

    await saver.adelete_thread(thread_id)

    assert await saver.aget_tuple(config) is None
    results = [c async for c in saver.alist({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}})]
    assert results == []
