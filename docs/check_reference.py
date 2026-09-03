from __future__ import annotations

import argparse
import importlib
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import cast

from sphinx.util.inventory import InventoryFile

import celnn

PUBLIC_MODULES: dict[str, ModuleType] = {
    "celnn": celnn,
    "celnn.backends": importlib.import_module("celnn.backends"),
    "celnn.domains": importlib.import_module("celnn.domains"),
    "celnn.io": importlib.import_module("celnn.io"),
    "celnn.templates": importlib.import_module("celnn.templates"),
    "celnn.training": importlib.import_module("celnn.training"),
}
DEFAULT_INVENTORY = Path(__file__).parent / "_build" / "html" / "objects.inv"


def _load_inventory(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise SystemExit(f"Sphinx inventory not found: {path}")

    loaded = InventoryFile.loads(path.read_bytes(), uri="")
    return cast(dict[str, dict[str, object]], loaded.data)


def _python_entries(
    inventory: dict[str, dict[str, object]],
) -> dict[str, object]:
    entries: dict[str, object] = {}
    for object_type, objects in inventory.items():
        if object_type.startswith("py:"):
            entries.update(objects)
    return entries


def _allowed_public_prefixes() -> tuple[str, ...]:
    prefixes = [
        f"{module_name}.{symbol}"
        for module_name, module in PUBLIC_MODULES.items()
        for symbol in getattr(module, "__all__", ())
    ]
    return tuple(prefixes)


def _is_public_name(qualified_name: str, allowed: tuple[str, ...]) -> bool:
    return any(
        qualified_name == prefix
        or qualified_name.startswith(f"{prefix}.")
        for prefix in allowed
    )


def _entry_uri(entry: object) -> str:
    return str(getattr(entry, "uri", ""))


def _unexpected_internal_objects(
    inventory: dict[str, dict[str, object]],
) -> list[str]:
    allowed = _allowed_public_prefixes()
    entries = _python_entries(inventory)
    public_uris = {
        _entry_uri(entry)
        for qualified_name, entry in entries.items()
        if _is_public_name(qualified_name, allowed)
    }
    public_uris.discard("")

    unexpected: list[str] = []
    for qualified_name, entry in sorted(entries.items()):
        if not qualified_name.startswith("celnn."):
            continue
        if _is_public_name(qualified_name, allowed):
            continue
        # Autodoc may inventory an implementation name as an alias of the
        # public object. A shared URI means no extra private target was added.
        if _entry_uri(entry) in public_uris:
            continue
        unexpected.append(qualified_name)

    return unexpected


def _public_counts(
    inventory: dict[str, dict[str, object]],
    module_name: str,
) -> Counter[str]:
    prefix = f"{module_name}."
    counts: Counter[str] = Counter()

    for object_type, objects in inventory.items():
        if not object_type.startswith("py:"):
            continue
        for qualified_name in objects:
            if not qualified_name.startswith(prefix):
                continue
            symbol = qualified_name[len(prefix) :]
            if "." not in symbol:
                counts[symbol] += 1

    return counts


def _module_errors(
    module_name: str,
    module: ModuleType,
    inventory: dict[str, dict[str, object]],
) -> list[str]:
    counts = _public_counts(inventory, module_name)
    exported = set(getattr(module, "__all__", ()))
    rendered = set(counts)

    missing = sorted(exported - rendered)
    unexpected = sorted(rendered - exported)
    duplicates = sorted(name for name, count in counts.items() if count != 1)

    if not (missing or unexpected or duplicates):
        return []

    mismatch = (
        f"{module_name} rendered API does not match "
        f"{module_name}.__all__."
    )
    details = [mismatch]
    if missing:
        details.append(f"  missing: {', '.join(missing)}")
    if unexpected:
        details.append(f"  unexpected: {', '.join(unexpected)}")
    if duplicates:
        duplicate_names = ", ".join(duplicates)
        details.append(f"  not rendered exactly once: {duplicate_names}")
    return details


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify rendered public API objects in a Sphinx inventory."
    )
    parser.add_argument(
        "inventory",
        nargs="?",
        type=Path,
        default=DEFAULT_INVENTORY,
        help="Path to the generated objects.inv file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inventory = _load_inventory(args.inventory)
    errors: list[str] = []

    if not any(name.startswith("py:") for name in inventory):
        errors.append("Sphinx inventory contains no Python-domain objects.")

    unexpected_internal = _unexpected_internal_objects(inventory)
    if unexpected_internal:
        errors.append(
            "Sphinx inventory exposes non-public CELNN Python-domain objects:"
        )
        errors.extend(f"  unexpected: {name}" for name in unexpected_internal)

    for module_name, module in PUBLIC_MODULES.items():
        errors.extend(_module_errors(module_name, module, inventory))

    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
