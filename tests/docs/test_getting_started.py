from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

nbclient = pytest.importorskip("nbclient")
nbformat = pytest.importorskip("nbformat")
pytest.importorskip("ipykernel")

NOTEBOOK = Path(__file__).resolve().parents[2] / "examples" / "notebooks" / "getting_started.ipynb"
KERNEL = "immune-getting-started-test"
SECRETS = ("OPENAI_API_KEY", "TYPESAFE_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING")
KEY_SHAPES = re.compile(r"sk-[A-Za-z0-9_-]{20,}|lsv2_[A-Za-z0-9_]{20,}")


@pytest.fixture
def kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    spec = tmp_path / "kernels" / KERNEL
    spec.mkdir(parents=True)
    argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    environment = {"IMMUNE_NOTEBOOK_OFFLINE": "1", "DEMO_SPEND_FILE": str(tmp_path / "spend.json")}
    (spec / "kernel.json").write_text(
        json.dumps({"argv": argv, "display_name": KERNEL, "language": "python", "env": environment})
    )
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path))
    for name in SECRETS:
        monkeypatch.delenv(name, raising=False)
    return KERNEL


def saved_text() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return json.dumps([cell.get("outputs", []) for cell in notebook.cells if cell.cell_type == "code"])


def test_the_notebook_is_saved_with_outputs_from_one_clean_run() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
    assert all(output.get("output_type") != "error" for cell in code for output in cell.outputs)
    assert "running live" in saved_text()
    last = "".join(output.get("text", "") for output in code[-1].outputs)
    assert last.strip() == "done"


def test_the_saved_outputs_contain_no_secrets() -> None:
    text = saved_text()
    assert not KEY_SHAPES.search(text)
    for name in SECRETS:
        value = os.environ.get(name, "")
        if len(value) > 8:
            assert value not in text, name
    assert "/home/" not in text


def test_the_notebook_runs_offline_from_start_to_finish(kernel: str) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(
        notebook, timeout=300, kernel_name=kernel, resources={"metadata": {"path": str(NOTEBOOK.parent)}}
    )
    client.execute()
    printed = "".join(
        output.get("text", "") for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
    )
    assert "running offline" in printed
    assert "5 vaccines, 10 examples: all passed" in printed
    assert "2 passed" in printed
    assert printed.rstrip().endswith("done")
