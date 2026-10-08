"""Checks the Python API docstrings (the generated stub and the `dsp_kitchen` wrappers).

1. numpydoc validation (`numpydoc lint`), keeping the checks that find missing or wrong
   documentation and skipping the ones this project does not follow (listed in `IGNORED`).
2. Constructors: a class's `Parameters` section (numpydoc puts constructor arguments there) names
   exactly the arguments of its `__new__` / `__init__`, unless the constructor documents them itself
   or (settings classes) every argument is a documented property of the same name.
3. Examples: every `>>>` example parses as Python.

Run: uv run --no-project --with numpydoc python docs/check_docstrings.py
Exits 1 and lists the problems when there are any.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUB = ROOT / "dsp_kitchen_py/dsp_kitchen_bindings/dsp_kitchen_bindings.pyi"
WRAPPERS = ROOT / "dsp_kitchen_py/dsp_kitchen"

IGNORED = {
    "SA01": "See Also sections are not used",
    "EX01": "examples are given where they help, not on every item",
    "ES01": "an extended summary is optional",
    "SS05": "summaries are descriptive ('Returns …', 'The …'), not imperative",
    "SS06": "a summary may wrap",
    "PR07": "type-only entries are deliberate for self-explanatory arguments (`probe : ProbeLayout`)",
    "RT03": "type-only return entries are deliberate (`SortingOutput`)",
    "PR08": "descriptions may start with a unit or a lowercase term (µV, k-means)",
    "RT04": "as PR08",
    "SS02": "as PR08",
}
# Checks that do not apply to some objects
DUNDER_WITHOUT_DOCS = {"__repr__", "__len__", "__str__", "__eq__", "__hash__", "__iter__"}
# Native entry points documented on their typed Python wrappers (`dsp_kitchen.synapse.ml.*.run`)
DOCUMENTED_ON_WRAPPER = {"run", "run_emusort"}


def qualified_names(tree: ast.Module) -> dict[int, tuple[str, ast.AST]]:
    """Line (of the definition and of its first decorator) → (qualified name, node)."""
    names: dict[int, tuple[str, ast.AST]] = {}

    def walk(nodes: list[ast.stmt], prefix: str) -> None:
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                entry = (prefix + node.name, node)
                names[node.lineno] = entry
                for decorator in node.decorator_list:
                    names[decorator.lineno] = entry
                if isinstance(node, ast.ClassDef):
                    walk(node.body, prefix + node.name + ".")

    walk(tree.body, "")
    return names


def is_setter(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef) and any(
        isinstance(d, ast.Attribute) and d.attr == "setter" for d in node.decorator_list
    )


def numpydoc_findings(path: Path, source: str) -> list[str]:
    """`numpydoc lint` on a `.py` copy (the linter skips `.pyi`), filtered."""
    tree = ast.parse(source)
    names = qualified_names(tree)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / (path.stem + ".py")
        shutil.copyfile(path, copy)
        result = subprocess.run(["numpydoc", "lint", str(copy)], capture_output=True, text=True)
    findings = []
    # numpydoc lint reports on stderr
    for line in (result.stdout + result.stderr).splitlines():
        m = re.match(r".*?:(\d+): (\w+) (.*)", line.strip())
        if not m:
            continue
        lineno, code, message = int(m.group(1)), m.group(2), m.group(3)
        if code in IGNORED:
            continue
        if lineno not in names:  # the module itself
            continue
        name, node = names[lineno]
        short = name.rsplit(".", 1)[-1]
        if short.startswith("_") and not short.startswith("__"):  # private
            continue
        if code == "GL08" and short in DUNDER_WITHOUT_DOCS:
            continue
        # Constructors without a docstring are documented by their class (checked below)
        if code == "GL08" and short in ("__new__", "__init__"):
            continue
        if name in DOCUMENTED_ON_WRAPPER:
            continue
        # Constructor arguments live in the class docstring; checked below instead
        if code == "PR02" and isinstance(node, ast.ClassDef):
            continue
        if code == "PR01" and short in ("__new__", "__init__"):
            continue
        # Property setters take `value`, documented by the property
        if code == "PR01" and is_setter(node) and "'value'" in message:
            continue
        findings.append(f"{path.relative_to(ROOT)}:{lineno} {name}: {code} {message}")
    return findings


def documented_parameters(doc: str) -> list[str] | None:
    """Names in a numpydoc `Parameters` section (None when there is none)."""
    m = re.search(r"^Parameters\n-+\n(.*?)(?:^\S[^\n]*\n-+\n|\Z)", doc, re.S | re.M)
    if not m:
        return None
    names = []
    for line in m.group(1).splitlines():
        if line and not line[0].isspace():
            for part in line.split(":")[0].split(","):
                names.append(part.strip().lstrip("*"))
    return names


def constructor_findings(path: Path, tree: ast.Module) -> list[str]:
    findings = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        ctor = next(
            (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("__new__", "__init__")),
            None,
        )
        if ctor is None:
            continue
        args = [a.arg for a in ctor.args.args + ctor.args.kwonlyargs if a.arg not in ("self", "cls")]
        if ctor.args.vararg:
            args.append(ctor.args.vararg.arg)
        if ctor.args.kwarg:
            args.append(ctor.args.kwarg.arg)
        own = documented_parameters(ast.get_docstring(ctor) or "")
        documented = own if own is not None else documented_parameters(ast.get_docstring(cls) or "")
        if not args and not documented:
            continue
        # Settings classes document each constructor argument as the property of the same name
        properties = {
            n.name
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and any(isinstance(d, ast.Name) and d.id == "property" for d in n.decorator_list)
            and ast.get_docstring(n)
        }
        if documented is None and args and set(args) <= properties:
            continue
        where = f"{cls.name}.{ctor.name}" if own is not None else cls.name
        if documented is None:
            findings.append(f"{path.relative_to(ROOT)}:{cls.lineno} {cls.name}: constructor arguments {args} not documented")
        elif set(documented) != set(args):
            missing, extra = sorted(set(args) - set(documented)), sorted(set(documented) - set(args))
            findings.append(
                f"{path.relative_to(ROOT)}:{cls.lineno} {where}: Parameters differ from the constructor"
                f" (undocumented: {missing}, unknown: {extra})"
            )
    return findings


def example_findings(path: Path, tree: ast.Module) -> list[str]:
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Module)):
            continue
        doc = ast.get_docstring(node)
        if not doc or ">>>" not in doc:
            continue
        # Consecutive `>>>` / `...` lines form one example
        block: list[str] = []
        blocks: list[list[str]] = []
        for line in doc.splitlines():
            stripped = line.strip()
            if stripped.startswith(">>> ") or stripped == ">>>":
                block.append(stripped[4:])
            elif stripped.startswith("... ") or stripped == "...":
                block.append(stripped[4:])
            elif block:
                blocks.append(block)
                block = []
        if block:
            blocks.append(block)
        for code in blocks:
            try:
                compile("\n".join(code), "<example>", "exec")
            except SyntaxError as e:
                name = getattr(node, "name", "module")
                findings.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}: example does not parse: {e.msg}")
    return findings


def main() -> int:
    files = [STUB, *sorted(p for p in WRAPPERS.rglob("*.py"))]
    findings = []
    for path in files:
        source = path.read_text()
        tree = ast.parse(source)
        findings += numpydoc_findings(path, source)
        findings += constructor_findings(path, tree)
        findings += example_findings(path, tree)
    for f in findings:
        print(f)
    print(f"{len(files)} files checked, {len(findings)} problems")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
