from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UdfEntry:
    module: str
    name: str
    implemented: bool
    python_path: str | None = None
    stub_path: str | None = None
    signature: str | None = None
    docstring: str | None = None


@dataclass(frozen=True)
class UdfCatalog:
    entries: tuple[UdfEntry, ...]

    def get(self, module: str, name: str) -> UdfEntry | None:
        for entry in self.entries:
            if entry.module == module and entry.name == name:
                return entry
        return None


def discover_udfs(root: str | Path = "udfs") -> UdfCatalog:
    root_path = Path(root)
    discovered: dict[tuple[str, str], UdfEntry] = {}

    for suffix, implemented in ((".py", True), (".pyi", False)):
        for path in sorted(root_path.rglob(f"*{suffix}")):
            module = _module_name(root_path, path)
            for function in _read_functions(path):
                key = (module, function["name"])
                existing = discovered.get(key)
                entry = UdfEntry(
                    module=module,
                    name=function["name"],
                    implemented=implemented or (existing.implemented if existing else False),
                    python_path=str(path) if implemented else (existing.python_path if existing else None),
                    stub_path=str(path) if not implemented else (existing.stub_path if existing else None),
                    signature=function["signature"] if not implemented else (existing.signature if existing else None),
                    docstring=function["docstring"] or (existing.docstring if existing else None),
                )
                discovered[key] = entry

    entries = tuple(sorted(discovered.values(), key=lambda entry: (entry.module, entry.name)))
    return UdfCatalog(entries=entries)


def _module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root.parent).with_suffix("")
    return ".".join(relative.parts)


def _read_functions(path: Path) -> list[dict[str, str | None]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions: list[dict[str, str | None]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        rendered = ast.unparse(node)
        functions.append(
            {
                "name": node.name,
                "signature": rendered.splitlines()[0],
                "docstring": ast.get_docstring(node),
            }
        )
    return functions
