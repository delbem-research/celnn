from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import celnn

REFERENCE = Path(__file__).parent / "source" / "reference" / "top-level.md"
DIRECTIVE = re.compile(
    r"^```\{auto(?:class|function|data)\}\s+celnn\.([A-Za-z_]\w*)\s*$",
    re.MULTILINE,
)


def main() -> None:
    documented = DIRECTIVE.findall(REFERENCE.read_text(encoding="utf-8"))
    counts = Counter(documented)
    exported = set(celnn.__all__)
    documented_set = set(documented)

    missing = sorted(exported - documented_set)
    unexpected = sorted(documented_set - exported)
    duplicates = sorted(name for name, count in counts.items() if count != 1)

    if missing or unexpected or duplicates:
        details = ["Top-level API reference does not match celnn.__all__."]
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        if duplicates:
            details.append(f"not documented exactly once: {', '.join(duplicates)}")
        raise SystemExit("\n".join(details))


if __name__ == "__main__":
    main()
