"""The user guide notebooks: indexed, self-contained, saved from a clean run, and in sync with the API."""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

GUIDE = Path(__file__).resolve().parents[2] / "notebooks" / "user_guide"
NOTEBOOKS = sorted(GUIDE.glob("*.ipynb"))
_NOTEBOOK_NAMES = {"display", "get_ipython"}
_SECRET = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}|\bts_[A-Za-z0-9]{20,}")
_LOCAL_PATH = re.compile(r"/home/|/Users/|C:\\\\Users")
_SIGNATURE_ENTRY = re.compile(r"^### `(?P<name>[\w.]+)\(\)`\s+```python\n(?P<signature>.*?)\n```", re.S)
_HEADING = re.compile(r"^#{1,6} (?P<text>.+)$")
_INLINE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_LINK_TARGET = re.compile(r"\]\(((?:[^()\s]|\([^()\s]*\))+)\)")
_DEFAULT_OVERRIDES = {("FakeToolCall", "arguments"): "{}", ("FakeToolCall", "call_id"): "<generated>"}


def cells(path: Path) -> list[dict[str, Any]]:
    notebook: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return [{**cell, "source": "".join(cell["source"])} for cell in notebook["cells"]]


def code_cells(path: Path) -> list[dict[str, Any]]:
    return [cell for cell in cells(path) if cell["cell_type"] == "code"]


def without_shell_lines(source: str) -> str:
    lines: list[str] = []
    continued = False
    for line in source.splitlines():
        stripped = line.lstrip()
        if continued or stripped.startswith(("!", "%")):
            if not continued:
                lines.append(line[: len(line) - len(stripped)] + "pass")
            continued = line.rstrip().endswith("\\")
            continue
        lines.append(line)
    return "\n".join(lines)


def free_names(source: str) -> set[str]:
    flags = ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    tree = compile(without_shell_lines(source), "<cell>", "exec", flags=flags)
    defined: set[str] = set()
    loaded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else defined).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    return loaded - defined - set(dir(builtins)) - _NOTEBOOK_NAMES


def _default(owner: str, parameter: inspect.Parameter) -> str:
    override = _DEFAULT_OVERRIDES.get((owner, parameter.name))
    if override is not None:
        return override
    value = parameter.default
    return f'"{value}"' if isinstance(value, str) else repr(value)


def rendered_signature(qualified: str) -> str:
    module_name, _, attribute = qualified.rpartition(".")
    target = getattr(importlib.import_module(module_name), attribute)
    signature = inspect.signature(target)
    parts: list[str] = []
    star = False
    for parameter in signature.parameters.values():
        text = parameter.name
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            text, star = f"*{parameter.name}", True
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            text = f"**{parameter.name}"
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        if parameter.annotation is not inspect.Parameter.empty:
            annotation = parameter.annotation
            text += f": {annotation if isinstance(annotation, str) else annotation.__name__}"
        if parameter.default is not inspect.Parameter.empty:
            text += f" = {_default(attribute, parameter)}"
        parts.append(text)
    returns = ""
    if not inspect.isclass(target) and signature.return_annotation is not inspect.Signature.empty:
        returns = f" -> {signature.return_annotation}"
    return f"{qualified}({', '.join(parts)}){returns}"


def test_the_guide_has_notebooks() -> None:
    assert len(NOTEBOOKS) >= 15


def test_the_readme_lists_every_notebook_and_every_link_resolves() -> None:
    readme = (GUIDE / "README.md").read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\(([^)#]+\.ipynb)", readme))
    assert linked == {path.name for path in NOTEBOOKS}


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_every_code_cell_runs_on_its_own(path: Path) -> None:
    problems = {}
    for index, cell in enumerate(code_cells(path)):
        names = free_names(cell["source"])
        if names:
            problems[index] = sorted(names)
    assert not problems, f"cells use names defined elsewhere: {problems}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebooks_are_saved_from_one_clean_run(path: Path) -> None:
    runnable = code_cells(path)
    assert [cell["execution_count"] for cell in runnable] == list(range(1, len(runnable) + 1))
    for index, cell in enumerate(runnable):
        assert cell["outputs"], f"code cell {index} has no saved output"
        text = json.dumps(cell["outputs"])
        assert '"output_type": "error"' not in text, f"code cell {index} saved an error"
        assert not _SECRET.search(text), f"code cell {index} output looks like it contains a key"
        assert not _LOCAL_PATH.search(text), f"code cell {index} output contains a local path"


def heading_ids(path: Path) -> set[str]:
    """Jupyter's heading anchors: the heading's rendered text, with spaces replaced by hyphens."""
    ids = set()
    for cell in cells(path):
        if cell["cell_type"] != "markdown":
            continue
        fenced = False
        for line in cell["source"].splitlines():
            fenced ^= line.startswith("```")
            heading = _HEADING.match(line)
            if heading and not fenced:
                text = _INLINE_LINK.sub(r"\1", heading["text"]).replace("`", "").replace("**", "")
                ids.add(re.sub(r"\s+", "-", text.strip()))
    return ids


def test_links_between_guides_point_to_existing_notebooks_and_sections() -> None:
    anchors = {path.name: heading_ids(path) for path in NOTEBOOKS}
    for path in NOTEBOOKS:
        for cell in cells(path):
            if cell["cell_type"] != "markdown":
                continue
            for target in _LINK_TARGET.findall(cell["source"]):
                page, _, anchor = target.partition("#")
                page = page or path.name
                if not page.endswith(".ipynb") or ":" in page:
                    continue
                assert (GUIDE / page).exists(), f"{path.name} links to missing {target}"
                assert not anchor or anchor in anchors[page], f"{path.name} links to missing section {target}"


def test_the_api_reference_signatures_match_the_code() -> None:
    reference = next(path for path in NOTEBOOKS if "api_reference" in path.name)
    entries = [
        match
        for cell in cells(reference)
        if cell["cell_type"] == "markdown"
        for match in _SIGNATURE_ENTRY.finditer(cell["source"])
    ]
    assert len(entries) >= 20
    for entry in entries:
        documented = "".join(entry["signature"].split())
        actual = "".join(rendered_signature(entry["name"]).split())
        assert documented == actual, f"{entry['name']}: documented {entry['signature']!r}, code has {actual!r}"


@pytest.mark.skipif(os.environ.get("IMMUNE_RUN_USER_GUIDE") != "1", reason="set IMMUNE_RUN_USER_GUIDE=1 to run live")
@pytest.mark.live
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebooks_run_live(path: Path) -> None:
    nbclient = pytest.importorskip("nbclient")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("langgraph")
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("TYPESAFE_API_KEY")):
        pytest.skip("needs OPENAI_API_KEY and TYPESAFE_API_KEY")
    notebook = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(notebook, timeout=900, resources={"metadata": {"path": str(GUIDE)}}).execute()
