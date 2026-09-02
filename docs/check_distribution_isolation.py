from __future__ import annotations

import re
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

DOC_DEPENDENCIES = {"sphinx", "myst-nb", "furo"}


def _contains_docs(names: list[str]) -> bool:
    return any(
        "/docs/" in f"/{name}" or name.endswith("/docs") for name in names
    )


def _requirement_name(value: str) -> str:
    return (
        re.split(r"[ (<>=!~;@\[]", value.strip(), maxsplit=1)[0]
        .lower()
        .replace("_", "-")
    )


def main() -> None:
    wheel = next(Path("dist").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        metadata_name = next(
            name
            for name in wheel_names
            if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode("utf-8")

    sdist = next(Path("dist").glob("*.tar.gz"))
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_names = archive.getnames()

    assert not _contains_docs(wheel_names)
    assert not _contains_docs(sdist_names)

    metadata_fields = Parser().parsestr(metadata)
    extras = {
        value.lower()
        for value in metadata_fields.get_all("Provides-Extra", [])
    }
    requirement_names = {
        _requirement_name(value)
        for value in metadata_fields.get_all("Requires-Dist", [])
    }
    assert "docs" not in extras
    assert requirement_names.isdisjoint(DOC_DEPENDENCIES)


if __name__ == "__main__":
    main()
