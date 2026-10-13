"""Export the diagnostic and fresh validation's named reports only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.namuwiki.pack_context_fixed_results import FILES as FRESH_FILES


DIAGNOSTIC_FILES = (
    "registration.json", "dev_report.json", "test_report.json",
    "dev_detail.csv", "test_detail.csv", "dev_checkpoint.json", "test_checkpoint.json",
    "diagnostic_report.json", "diagnostic_report.md", "failure_report.json",
)


def pack(root: Path, archive: Path) -> dict:
    sources = [(root / phase / name, f"{phase}/{name}")
               for phase, names in (("diagnostic", DIAGNOSTIC_FILES), ("fresh", FRESH_FILES))
               for name in names if (root / phase / name).is_file()]
    if not sources:
        raise ValueError("No diagnostic/fresh reports to export")
    if any(p.is_symlink() or root.resolve() not in p.resolve().parents for p, _ in sources):
        raise ValueError("Report export does not follow links or leave its output root")
    if archive.resolve() in {p.resolve() for p, _ in sources}:
        raise ValueError("Archive must not overwrite a report")
    manifest = dict(schema="context_fixed_followup_feedback_v1", files={})
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as z:
        for file, name in sources:
            content = file.read_bytes()
            manifest["files"][name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            z.writestr(name, content)
        z.writestr("export_manifest.json", json.dumps(manifest, indent=2) + "\n")
    temporary.replace(archive)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/context_fixed_followup_20261005"))
    args = parser.parse_args()
    archive = args.output_dir.with_name(args.output_dir.name + "_feedback.zip")
    manifest = pack(args.output_dir, archive)
    print(f"Follow-up feedback ZIP: {archive} ({len(manifest['files'])} files, {archive.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
