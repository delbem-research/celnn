from __future__ import annotations

import re
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

DOC_DEPENDENCIES = {"sphinx", "myst-nb", "furo"}


def _docs_entries(names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if "/docs/" in f"/{name}" or name.endswith("/docs")
    ]


def _requirement_name(value: str) -> str:
    return (
        re.split(r"[ (<>=!~;@\[]", value.strip(), maxsplit=1)[0]
        .lower()
        .replace("_", "-")
    )


def _single_artifact(pattern: str, label: str) -> Path:
    artifacts = sorted(Path("dist").glob(pattern))
    if len(artifacts) != 1:
        raise SystemExit(
            f"Expected exactly one {label} in dist/, found {len(artifacts)}."
        )
    return artifacts[0]


def main() -> None:
    wheel = _single_artifact("*.whl", "wheel")
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        metadata_names = [
            name
            for name in wheel_names
            if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise SystemExit(
                "Expected exactly one .dist-info/METADATA file in the wheel."
            )
        metadata = archive.read(metadata_names[0]).decode("utf-8")

    sdist = _single_artifact("*.tar.gz", "source distribution")
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_names = archive.getnames()

    errors: list[str] = []
    wheel_docs = _docs_entries(wheel_names)
    if wheel_docs:
        errors.append(f"wheel contains docs content: {wheel_docs[0]}")

    sdist_docs = _docs_entries(sdist_names)
    if sdist_docs:
        errors.append(f"sdist contains docs content: {sdist_docs[0]}")

    metadata_fields = Parser().parsestr(metadata)
    extras = {
        value.lower()
        for value in metadata_fields.get_all("Provides-Extra", [])
    }
    if "docs" in extras:
        errors.append("wheel metadata exposes a docs extra")

    requirement_names = {
        _requirement_name(value)
        for value in metadata_fields.get_all("Requires-Dist", [])
    }
    leaked_dependencies = sorted(requirement_names & DOC_DEPENDENCIES)
    if leaked_dependencies:
        errors.append(
            "wheel metadata depends on docs tooling: "
            + ", ".join(leaked_dependencies)
        )

    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
