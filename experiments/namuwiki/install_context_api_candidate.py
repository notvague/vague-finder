"""Source-checked installation of experiment-only API candidate verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.namuwiki.install_context_review_cleanup import atomic_write, safe_file


PAYLOAD = "experiments/namuwiki/context_api_candidate_payload"
TARGETS = (
    "experiments/namuwiki/verify_context_api_ready.py",
    "tests/context/test_context_validated_activation.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data.decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")).hexdigest()


def preflight(project: Path) -> list[tuple[str, bytes, bytes]]:
    manifest = json.loads(safe_file(project, PAYLOAD + "/manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != "context_api_candidate_payload_v1" or set(manifest.get("targets", {})) != set(TARGETS):
        raise ValueError("Invalid API candidate verifier manifest")
    for name, expected in manifest["required_files"].items():
        if ("\\" in name or ":" in name or ".." in name.split("/")
                or not name.startswith(("experiments/namuwiki/", "tests/context/"))):
            raise ValueError("Invalid API verifier companion path")
        if digest(safe_file(project, name).read_bytes()) != expected:
            raise ValueError("Extract the complete API candidate fix ZIP: " + name)
    prepared = []
    for name in TARGETS:
        item = manifest["targets"][name]
        if item.get("payload") != Path(name).name + ".payload":
            raise ValueError("Unexpected API verifier payload name")
        data = safe_file(project, PAYLOAD + "/" + item["payload"]).read_bytes()
        if digest(data) != item["after"]:
            raise ValueError("API verifier payload integrity mismatch")
        compile(data.decode("utf-8-sig"), name, "exec")
        before = safe_file(project, name).read_bytes()
        if digest(before) == item["after"]:
            prepared.append((name, before, before))
            continue
        if digest(before) != item["before"]:
            raise ValueError("Experiment source differs from supplied revision; preserved: " + name)
        text = data.decode("utf-8-sig").replace("\r\n", "\n")
        if b"\r\n" in before:
            text = text.replace("\n", "\r\n")
        after = (b"\xef\xbb\xbf" if before.startswith(b"\xef\xbb\xbf") else b"") + text.encode("utf-8")
        prepared.append((name, before, after))
    return prepared


def install(project: Path, root: Path) -> dict:
    from experiments.namuwiki.resume_context_api_check import passed_binding
    project, root = project.resolve(), root.resolve()
    if project not in root.parents or root.relative_to(project).parts[0] != "artifacts":
        raise ValueError("API installation reports/backups must stay inside project artifacts")
    # Refuse unfinished evaluation before replacing even experiment code.
    passed_binding(project, root)
    prepared = preflight(project)
    changed = []
    try:
        for name, before, after in prepared:
            source = safe_file(project, name)
            if source.read_bytes() != before:
                raise ValueError("API experiment source changed during installation")
            if before != after:
                backup = root / "api_candidate_install/backups" / name
                if backup.is_file() and backup.read_bytes() != before:
                    raise ValueError("Existing API experiment backup differs; preserved")
                if not backup.exists():
                    atomic_write(backup, before)
                atomic_write(source, after)
                changed.append((name, before, after))
        report = dict(schema="context_api_candidate_install_v1", status="applied",
                      changed_files=[name for name, _, _ in changed],
                      normalized_source_sha256={name: dict(before=digest(before), after=digest(after))
                                                for name, before, after in prepared},
                      production_sources_settings_and_labels_changed=False)
        atomic_write(root / "api_candidate_install/install_report.json",
                     (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    except Exception:
        for name, before, after in reversed(changed):
            source = safe_file(project, name)
            if source.read_bytes() == after:
                atomic_write(source, before)
        raise
    print("[PASS] API candidate verifier installed; no production source/settings/labels changed", flush=True)
    return report


def main() -> int:
    from experiments.namuwiki.context_review_cleanup_runner import output_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    try:
        install(project, output_path(project, args.output_dir))
        return 0
    except Exception as exc:
        print(f"API verifier installation stopped: {type(exc).__name__}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
