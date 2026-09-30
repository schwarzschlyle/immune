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

TOUR = Path(__file__).resolve().parents[2] / "examples" / "notebooks" / "immune_tour.ipynb"
KERNEL = "immune-tour-test"


@pytest.fixture
def kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    spec = tmp_path / "kernels" / KERNEL
    spec.mkdir(parents=True)
    argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    (spec / "kernel.json").write_text(json.dumps({"argv": argv, "display_name": KERNEL, "language": "python"}))
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    return KERNEL


def test_the_tour_is_saved_with_outputs_from_one_clean_run() -> None:
    notebook = nbformat.read(TOUR, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
    assert all(output.get("output_type") != "error" for cell in code for output in cell.outputs)
    assert "".join(output.get("text", "") for output in code[-1].outputs).strip() == "done"


def test_the_saved_outputs_contain_no_secrets_or_personal_paths() -> None:
    notebook = nbformat.read(TOUR, as_version=4)
    text = json.dumps([cell.get("outputs", []) for cell in notebook.cells if cell.cell_type == "code"])
    assert not re.search(r"sk-[A-Za-z0-9_-]{20,}|lsv2_[A-Za-z0-9_]{20,}", text)
    for name in ("OPENAI_API_KEY", "TYPESAFE_API_KEY", "LANGSMITH_API_KEY"):
        value = os.environ.get(name, "")
        if len(value) > 8:
            assert value not in text, name
    assert "/home/" not in text


def test_the_tour_runs_offline_from_start_to_finish(kernel: str, tmp_path: Path) -> None:
    notebook = nbformat.read(TOUR, as_version=4)
    client = nbclient.NotebookClient(
        notebook, timeout=300, kernel_name=kernel, resources={"metadata": {"path": str(tmp_path)}}
    )
    client.execute()
    printed = "".join(
        output.get("text", "") for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
    )
    assert "set TYPESAFE_API_KEY and OPENAI_API_KEY" in printed
    assert printed.rstrip().endswith("done")
