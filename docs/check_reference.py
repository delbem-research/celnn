from __future__ import annotations

import importlib
import re
from collections import Counter
from pathlib import Path
from types import ModuleType

import celnn

REFERENCE_ROOT = Path(__file__).parent / "source" / "reference"
DIRECTIVE = re.compile(
    r"^```\{auto\w+\}\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*$",
    re.MULTILINE,
)
PUBLIC_MODULES: dict[str, ModuleType] = {
    "celnn": celnn,
    "celnn.backends": importlib.import_module("celnn.backends"),
    "celnn.training": importlib.import_module("celnn.training"),
}


def _documented_symbols() -> dict[str, list[str]]:
    documented = {name: [] for name in PUBLIC_MODULES}
    module_names = sorted(PUBLIC_MODULES, key=len, reverse=True)

    for path in sorted(REFERENCE_ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for qualified_name in DIRECTIVE.findall(text):
            for module_name in module_names:
                prefix = f"{module_name}."
                if not qualified_name.startswith(prefix):
                    continue
                symbol = qualified_name[len(prefix) :]
                if "." not in symbol:
                    documented[module_name].append(symbol)
                break

    return documented


def _module_errors(
    module_name: str,
    module: ModuleType,
    documented: list[str],
) -> list[str]:
    counts = Counter(documented)
    exported = set(module.__all__)
    documented_set = set(documented)

    missing = sorted(exported - documented_set)
    unexpected = sorted(documented_set - exported)
    duplicates = sorted(name for name, count in counts.items() if count != 1)

    if not (missing or unexpected or duplicates):
        return []

    details = [f"{module_name} reference does not match {module_name}.__all__."]
    if missing:
        details.append(f"  missing: {', '.join(missing)}")
    if unexpected:
        details.append(f"  unexpected: {', '.join(unexpected)}")
    if duplicates:
        duplicate_names = ", ".join(duplicates)
        details.append(f"  not documented exactly once: {duplicate_names}")
    return details


def main() -> None:
    documented = _documented_symbols()
    errors: list[str] = []

    for module_name, module in PUBLIC_MODULES.items():
        errors.extend(
            _module_errors(module_name, module, documented[module_name])
        )

    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
