import os
import uuid

import pytest

requires_upstash = pytest.mark.skipif(
    not (os.environ.get("UPSTASH_REDIS_REST_URL") and os.environ.get("UPSTASH_REDIS_REST_TOKEN")),
    reason="UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN not set",
)


@pytest.fixture
def thread_id() -> str:
    """A fresh thread_id per test, so tests don't collide on a shared Upstash database."""
    return f"test-thread-{uuid.uuid4()}"
