# langgraph-checkpoint-upstash-redis

A [LangGraph](https://github.com/langchain-ai/langgraph) checkpoint saver backed by
[Upstash Redis](https://upstash.com/docs/redis) over its REST API.

## Why this exists

The official [`langgraph-checkpoint-redis`](https://github.com/redis-developer/langgraph-redis)
package relies on RediSearch and RedisJSON to index checkpoints. Upstash Redis does not
support those modules, so that package doesn't work against an Upstash Redis database.

This package sidesteps that entirely: checkpoints are stored as plain Redis hashes and
indexed with sorted sets, so it only needs commands every Redis-compatible server
(including Upstash) supports. It talks to Upstash exclusively over REST via the
[`upstash-redis`](https://github.com/upstash/redis-py) SDK, which makes it a good fit for
serverless Python (AWS Lambda, Vercel, Cloud Run, etc.) where long-lived TCP connections
are awkward or unavailable.

## Install

```bash
pip install langgraph-checkpoint-upstash-redis
```

## Usage

### Sync

```python
from langgraph.checkpoint.upstash_redis import UpstashRedisSaver

# Reads UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN from the environment.
saver = UpstashRedisSaver.from_env()

# Or pass credentials explicitly:
saver = UpstashRedisSaver.from_conn_info(url="...", token="...")

graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "thread-1"}}
graph.invoke(inputs, config)
```

### Async

```python
from langgraph.checkpoint.upstash_redis.aio import AsyncUpstashRedisSaver

saver = AsyncUpstashRedisSaver.from_env()
graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "thread-1"}}
await graph.ainvoke(inputs, config)
```

Use `AsyncUpstashRedisSaver` with `graph.ainvoke`/`astream`/etc.; use `UpstashRedisSaver`
for sync graph execution. Each saver only implements the sync or async half of the
`BaseCheckpointSaver` interface, matching the pattern used by other single-mode
checkpointers (e.g. `AsyncSqliteSaver`).

### TTL

Both savers accept an optional `ttl` (seconds). When set, checkpoints and their indices
expire automatically:

```python
saver = UpstashRedisSaver.from_env(ttl=60 * 60 * 24)  # 1 day
```

## Design notes

- **No RediSearch / RedisJSON.** Checkpoints are stored as Redis hashes
  (`checkpoint:{thread_id}:{checkpoint_ns}:{checkpoint_id}`) and indexed per
  thread/namespace with a sorted set (`checkpoints_idx:{thread_id}:{checkpoint_ns}`).
  LangGraph's checkpoint IDs are monotonically increasing strings, so lexicographic
  order (`ZRANGEBYLEX`) is equivalent to chronological order — no search index needed.
- **REST-only.** All I/O goes through the `upstash-redis` HTTP client. Writes to a
  checkpoint (the hash, its index entry, and namespace/thread bookkeeping sets) are
  batched into a single pipelined HTTP request.
- **Metadata filtering** (the `filter` argument to `list`) is applied client-side after
  fetching candidate checkpoints for the relevant thread/namespace, since there's no
  secondary index to push it down to. This mirrors what LangGraph's own `InMemorySaver`
  does and is fine at the checkpoint volumes a single thread accumulates.
- Binary checkpoint/metadata/write payloads (from LangGraph's `JsonPlusSerializer`) are
  base64-encoded before being stored in Redis hash fields, since the REST protocol is
  text-based.

Out of scope for this package: a `BaseStore` implementation, vector search, and the
`DeltaChannel` / `prune` / `copy_thread` / `delete_for_runs` checkpointer extensions —
only the core checkpoint CRUD surface is implemented.

## Testing

Unit tests for serialization and key-building run without any network access:

```bash
pip install -e ".[dev]"
pytest tests/test_base.py
```

Integration tests exercise a real Upstash Redis database and require
`UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN` to be set (a free Upstash database
works fine); they're skipped automatically otherwise:

```bash
export UPSTASH_REDIS_REST_URL=...
export UPSTASH_REDIS_REST_TOKEN=...
pytest tests/
```

## License

MIT
