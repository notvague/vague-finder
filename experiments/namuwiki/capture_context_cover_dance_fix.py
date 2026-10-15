"""Capture fixed evaluation output or package an existing log without rerunning."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile


OLD_OUTPUT = 'artifacts/context_fixed_followup_20261005'
DEFAULT_OUTPUT = OLD_OUTPUT + '_cover_dance_fix'


def output_path(value: str) -> str:
    value = value.replace('\\', '/').rstrip('/')
    if not re.fullmatch(r'artifacts/[A-Za-z0-9_./-]+', value) or '..' in value.split('/'):
        raise ValueError('OutputDir must stay inside artifacts without parent traversal')
    if value == OLD_OUTPUT:
        raise ValueError('Use a new output directory after changing the analyzer')
    return value


def capture(command: list[str], log_path: Path, *, cwd: Path) -> int:
    """Merge native stderr/stdout, stream to the terminal, and flush UTF-8 logs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', newline='\n') as log:
        # A resumed evaluation must not erase the log of its previous failure.
        log.write('\nCapture started: ' + datetime.now(timezone.utc).isoformat() + '\n')
        log.write('Command: ' + json.dumps(command, ensure_ascii=True) + '\n')
        log.flush()
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding='utf-8', errors='replace') as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            code = process.wait()
        log.write(f'Child process exit: {code}\n')
    return code


def recorded_child_exit(log_path: Path) -> int | None:
    """Return the last captured exit code; do not infer one from report status."""
    text = log_path.read_text(encoding='utf-8', errors='replace')
    values = re.findall(r'^Child process exit: (-?\d+)[ \t]*$', text, re.MULTILINE)
    return int(values[-1]) if values else None


def attach_console(archive: Path, log_path: Path, *, child_exit: int | None) -> Path:
    """Add logs to a new uniquely named ZIP, keeping the source ZIP read-only."""
    if not archive.is_file():
        raise FileNotFoundError('Evaluation did not export reports; use the saved console log')
    # Read a stable snapshot so the recorded SHA identifies these exact bytes.
    # Windows may allow reads while denying replacement of an open ZIP.
    from io import BytesIO

    source_bytes = archive.read_bytes()
    with ZipFile(BytesIO(source_bytes)) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        manifest = json.loads(source.read('export_manifest.json'))
        payloads = {name: source.read(name) for name in names if name != 'export_manifest.json'}
    if set(payloads) != set(manifest.get('files', {})):
        raise ValueError('Feedback manifest does not cover every member')
    for name, data in payloads.items():
        info = manifest['files'][name]
        if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
            raise ValueError(f'Feedback checksum mismatch: {name}')

    payloads['console_run.log'] = log_path.read_bytes()
    # Only these named post-validation reports are added. Backups, private
    # source files and arbitrary directory contents are never swept into ZIPs.
    for name in ('activation/activation_report.json', 'activation/api_smoke_report.json'):
        extra = log_path.parent / name
        if extra.is_file():
            if extra.is_symlink() or log_path.parent.resolve() not in extra.resolve().parents:
                raise ValueError('Activation report must stay inside the output root')
            payloads[name] = extra.read_bytes()
    info = dict(schema='context_cover_dance_capture_v2', child_exit=child_exit,
                captured_at=datetime.now(timezone.utc).isoformat(),
                reports_modified=False, report_verdict_is_authoritative=True,
                source_archive=archive.name,
                source_archive_sha256=hashlib.sha256(source_bytes).hexdigest(),
                source_archive_modified=False, packaging_reran_evaluation=False,
                analysis_policy='Packaging does not change analyses, settings, labels or result reports.')
    payloads['capture_info.json'] = (json.dumps(info, indent=2) + '\n').encode('utf-8')
    manifest['files'] = {name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                         for name, data in payloads.items()}
    manifest['console_capture'] = info
    # This is the final output path, reserved exclusively with a unique suffix.
    # Do not replace the original ZIP or an earlier combined ZIP: either may
    # be open in Explorer or held by another process on Windows.
    handle, destination = tempfile.mkstemp(prefix=archive.stem + '_with_log_',
                                           suffix='.zip', dir=archive.parent)
    destination = Path(destination)
    try:
        with os.fdopen(handle, 'w+b') as stream:
            with ZipFile(stream, 'w', compression=ZIP_DEFLATED) as target:
                for name, data in payloads.items():
                    target.writestr(name, data)
                target.writestr('export_manifest.json', json.dumps(manifest, indent=2).encode('utf-8'))
        with ZipFile(destination) as verified:
            for name, data in payloads.items():
                if verified.read(name) != data:
                    raise ValueError(f'Combined ZIP verification failed: {name}')
            if json.loads(verified.read('export_manifest.json')) != manifest:
                raise ValueError('Combined ZIP manifest verification failed')
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def inspect_boundary() -> list[dict]:
    """Inspect real parser behavior with synthetic model JSON; no API or DB."""
    from src.retrieval.query_analyzer import QueryAnalyzer

    cases = [
        ('radio_and_bass', '라디오에서 커버댄스한 곡은? 펑키한 베이스 소리가 기억나.',
         'Funky bass groove.', '', False, True, True),
        ('dance_history', '방송에서 댄스 커버한 곡이 뭐였지?', '', '', False, False, True),
        ('cover_and_history', '라디오에서 커버댄스한 곡이야. 앨범 표지에 빨간 꽃이 그려졌어.',
         '', 'Album artwork with red flowers.', True, False, True),
        ('dance_genre', '드럼 소리가 강하고 빠른 비트가 있는 댄스곡',
         'A fast dance track with strong drums.', '', False, True, False),
    ]
    analyzer = QueryAnalyzer(configured=True)
    rows = []
    for name, query, audio, image, want_image, want_audio, want_context in cases:
        raw = dict(intent_type='mixed', confidence=0.8, image_english_query=image,
                   audio_english_query=audio, context_clues=[],
                   modality_weights=dict(text=.7, image=.15 if image else 0., audio=.15 if audio else 0.))
        result = analyzer._parse(query, SimpleNamespace(text=json.dumps(raw)))
        observed = (bool(result.image_english_query), bool(result.audio_english_query), bool(result.context_clues))
        if observed != (want_image, want_audio, want_context) or result.confidence != .8:
            raise AssertionError(f'{name}: unexpected analyzer parse {observed}')
        row = dict(case=name, image=observed[0], audio=observed[1], context=observed[2],
                   context_clues=len(result.context_clues), confidence=result.confidence)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    print('[PASS] Offline analyzer parse; no Gemini call and no Qdrant access.', flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        # Keep the UTF-8 file accurate even if a redirected Windows terminal
        # cannot display an individual Docker progress character.
        sys.stdout.reconfigure(errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--inspect', action='store_true')
    mode.add_argument('--package-only', action='store_true',
                      help='Package saved console_run.log with existing feedback; no Docker, API or evaluation')
    mode.add_argument('--activate-only', action='store_true',
                      help='Verify/apply a completed passed evaluation and run real API checks; no evaluation rerun')
    parser.add_argument('--activate-after-pass', action='store_true',
                        help='Apply measured settings and verify HTTP only after full frozen evaluation passes')
    args = parser.parse_args(argv)
    if args.activate_after_pass and (args.check_only or args.inspect or args.package_only or args.activate_only):
        parser.error('--activate-after-pass requires the full evaluation mode')
    if args.inspect:
        inspect_boundary()
        return 0
    root = Path(__file__).resolve().parents[2]
    output = output_path(args.output_dir)
    log = root / output / ('console_check.log' if args.check_only else 'console_run.log')
    if args.package_only:
        try:
            archive = attach_console(root / (output + '_feedback.zip'), log,
                                     child_exit=recorded_child_exit(log))
        except Exception as exc:
            print(f'Log packaging failed: {type(exc).__name__}: {exc}. Send {log}', file=sys.stderr)
            return 2
        print('Packaging complete. No evaluation was run; saved verdicts are unchanged.', flush=True)
        print(f'Send this file: {archive}', flush=True)
        return 0
    if os.name != 'nt':
        parser.error('The capture launcher is for Windows; --inspect and --package-only also run on other systems')
    script = root / 'experiments/namuwiki/run_context_cover_dance_fix.ps1'
    command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'RemoteSigned',
               '-File', str(script), '-Worker', '-OutputDir', output]
    if args.check_only:
        command.append('-CheckOnly')
    if args.activate_after_pass:
        command.append('-ApplyAfterPass')
    if args.activate_only:
        command.append('-ActivateOnly')
    try:
        code = capture(command, log, cwd=root)
    except OSError as exc:
        print(f'Capture failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    print(f'Console log: {log}', flush=True)
    if not args.check_only:
        archive = root / (output + '_feedback.zip')
        try:
            archive = attach_console(archive, log, child_exit=code)
        except Exception as exc:
            print(f'Console attachment failed: {type(exc).__name__}: {exc}. Send {log}', file=sys.stderr)
            return code or 2
        print('Original feedback ZIP preserved; combined ZIP includes the captured log.', flush=True)
        print(f'Send this file: {archive}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
