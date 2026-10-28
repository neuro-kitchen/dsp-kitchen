"""The generated stubs (`python/dspk/_native/**/*.pyi`) list exactly what the compiled extension has.

mypy's stubtest does not support nested native submodules, so this compares names: every public
function, class, constant and class member at run time against the stubs, both ways. On failure,
regenerate: `cargo run --bin stub_gen` (see bindings/python/README.md).
"""

import ast
import importlib
from pathlib import Path

import pytest

import dspk._native

STUBS = Path(dspk.__file__).parent / "_native"
MODULES = [
    "dspk._native", "dspk._native.io", "dspk._native.ops", "dspk._native.ops.filter", "dspk._native.pipeline",
    "dspk._native.runtime",
]


def runtime_module(name: str):
    module = importlib.import_module("dspk._native")
    for part in name.split(".")[2:]:
        module = getattr(module, part)
    return module


def stub_tree(name: str) -> ast.Module:
    path = STUBS.joinpath(*name.split(".")[2:], "__init__.pyi")
    return ast.parse(path.read_text())


def stub_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom) and node.module is None:  # from . import io
            names.update(alias.name for alias in node.names)
    return names


def public(names) -> set[str]:
    return {n for n in names if not n.startswith("_") or n == "__version__"}


@pytest.mark.parametrize("name", MODULES)
def test_module_names_match(name):
    module = runtime_module(name)
    assert public(dir(module)) == public(stub_names(stub_tree(name)))


@pytest.mark.parametrize("name", MODULES)
def test_class_members_match(name):
    module = runtime_module(name)
    for node in stub_tree(name).body:
        if not isinstance(node, ast.ClassDef):
            continue
        cls = getattr(module, node.name)
        declared = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
        actual = {n for n in vars(cls) if not n.startswith("_")}
        assert {n for n in declared if not n.startswith("_")} == actual, node.name
