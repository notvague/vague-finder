"""Export only the fixed evaluation's reports and registered private labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


FILES = (
    "registration.json", "corpus_identity.json", "registered_new_labels.csv", "registered_new_review.json",
    "dev_report.json", "dev_detail.csv", "test_report.json", "test_detail.csv",
    "independent_report.json", "independent_detail.csv", "repeat_report.json",
    "validation_report.json", "validation_report.md", "failure_report.json",
    "dev_checkpoint.json", "test_checkpoint.json", "independent_checkpoint.json", "repeat_checkpoint.json",
    "dev_analysis_cache.json", "test_analysis_cache.json", "independent_analysis_cache.json", "repeat_analysis_cache.json",
)


def pack(root: Path, archive: Path) -> dict:
    sources = [root / name for name in FILES if (root / name).is_file()]
    if not sources:
        raise ValueError("No evaluation reports to export")
    if any(p.is_symlink() for p in sources):
        raise ValueError("Report export does not follow symbolic links")
    if archive.resolve() in {p.resolve() for p in sources}:
        raise ValueError("Archive must not overwrite an input report")
    report = dict(schema="context_fixed_feedback_v1", files={})
    archive.parent.mkdir(parents=True, exist_ok=True)
    temp = archive.with_suffix(archive.suffix + ".tmp")
    with ZipFile(temp, "w", compression=ZIP_DEFLATED) as zip_file:
        for file in sources:
            content = file.read_bytes()
            report["files"][file.name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            zip_file.writestr(file.name, content)
        zip_file.writestr("export_manifest.json", json.dumps(report, indent=2) + "\n")
    temp.replace(archive)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/context_fixed_validation_20261005"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    archive = args.archive or args.output_dir.with_name(args.output_dir.name + "_feedback.zip")
    packed = pack(args.output_dir, archive)
    print(f"Feedback ZIP: {archive} ({len(packed['files'])} files, {archive.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
