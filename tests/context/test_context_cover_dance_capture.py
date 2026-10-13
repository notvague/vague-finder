"""Native output capture preserves failure codes and untouched report bytes."""

import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

from experiments.namuwiki.capture_context_cover_dance_fix import (
    attach_console, capture, inspect_boundary, main, output_path, recorded_child_exit,
)
from experiments.namuwiki import capture_context_cover_dance_fix as capture_module


def make_feedback(path: Path, *, tamper=False):
    report = json.dumps({'status': 'in_progress', 'completed_phases': ['dev', 'test']}).encode()
    manifest = {'schema': 'context_fixed_followup_feedback_v1', 'files': {
        'fresh/validation_report.json': {'bytes': len(report),
                                        'sha256': hashlib.sha256(report).hexdigest()}
    }}
    if tamper:
        manifest['files']['fresh/validation_report.json']['sha256'] = 'wrong'
    with ZipFile(path, 'w') as archive:
        archive.writestr('fresh/validation_report.json', report)
        archive.writestr('export_manifest.json', json.dumps(manifest))
    return report


def test_native_stdout_stderr_unicode_and_failure_code_are_captured(tmp_path, capsys):
    log = tmp_path / 'console.log'
    command = [sys.executable, '-u', '-c',
               "import sys; print('방송 커버댄스'); print('검증 실패', file=sys.stderr); sys.exit(2)"]
    assert capture(command, log, cwd=tmp_path) == 2
    text = log.read_text(encoding='utf-8')
    assert '방송 커버댄스' in text and '검증 실패' in text
    assert 'Child process exit: 2' in text
    terminal = capsys.readouterr().out
    assert '방송 커버댄스' in terminal and '검증 실패' in terminal


def test_feedback_keeps_original_verdict_bytes_and_records_console_failure(tmp_path):
    archive = tmp_path / 'feedback.zip'
    report = make_feedback(archive)
    log = tmp_path / 'console.log'
    log.write_text('모달리티 검증 실패\n', encoding='utf-8')
    source_bytes = archive.read_bytes()
    combined = attach_console(archive, log, child_exit=1)
    assert combined != archive and '_with_log_' in combined.name
    assert archive.read_bytes() == source_bytes
    with ZipFile(combined) as result:
        assert result.read('fresh/validation_report.json') == report
        assert result.read('console_run.log') == log.read_bytes()
        manifest = json.loads(result.read('export_manifest.json'))
        for name, info in manifest['files'].items():
            data = result.read(name)
            assert len(data) == info['bytes']
            assert hashlib.sha256(data).hexdigest() == info['sha256']
        captured = json.loads(result.read('capture_info.json'))
        assert captured['child_exit'] == 1
        assert captured['reports_modified'] is False
        assert captured['report_verdict_is_authoritative'] is True
        assert captured['source_archive_modified'] is False
        assert captured['source_archive_sha256'] == hashlib.sha256(source_bytes).hexdigest()
        assert captured['packaging_reran_evaluation'] is False
    with ZipFile(archive) as source:
        assert 'console_run.log' not in source.namelist()


def test_repeated_packaging_preserves_both_combined_files_and_source(tmp_path):
    archive, log = tmp_path / 'feedback.zip', tmp_path / 'console.log'
    original = make_feedback(archive)
    original_zip = archive.read_bytes()
    outputs = []
    for message in ('first failure', 'second attempt'):
        log.write_text(message, encoding='utf-8')
        outputs.append(attach_console(archive, log, child_exit=1))
    assert len(set(outputs)) == 2
    assert archive.read_bytes() == original_zip
    for combined, message in zip(outputs, ('first failure', 'second attempt')):
        with ZipFile(combined) as result:
            assert len(result.namelist()) == len(set(result.namelist()))
            assert result.read('fresh/validation_report.json') == original
            assert result.read('console_run.log').decode() == message


def test_read_only_open_source_is_never_replaced_or_renamed(tmp_path, monkeypatch):
    archive, log = tmp_path / 'feedback.zip', tmp_path / 'console.log'
    make_feedback(archive)
    original_zip = archive.read_bytes()
    log.write_text('fresh022: analyzer fallback\n', encoding='utf-8')

    def forbidden_replace(*args, **kwargs):
        raise PermissionError('Windows-style replacement lock')

    monkeypatch.setattr(capture_module.os, 'replace', forbidden_replace)
    monkeypatch.setattr(Path, 'replace', forbidden_replace)
    monkeypatch.setattr(Path, 'rename', forbidden_replace)
    archive.chmod(0o444)
    try:
        with archive.open('rb'):
            combined = attach_console(archive, log, child_exit=1)
        assert archive.read_bytes() == original_zip
        with ZipFile(combined) as result:
            assert result.read('console_run.log') == log.read_bytes()
    finally:
        archive.chmod(0o644)


def test_failed_combined_write_cleans_only_its_new_file(tmp_path, monkeypatch):
    archive, log = tmp_path / 'feedback.zip', tmp_path / 'console.log'
    make_feedback(archive)
    original_zip = archive.read_bytes()
    log.write_text('failure', encoding='utf-8')
    original_log = log.read_bytes()

    def failed_write(*args, **kwargs):
        raise OSError('disk write failed')

    monkeypatch.setattr(ZipFile, 'writestr', failed_write)
    with pytest.raises(OSError, match='disk write failed'):
        attach_console(archive, log, child_exit=1)
    assert archive.read_bytes() == original_zip
    assert log.read_bytes() == original_log
    assert not list(tmp_path.glob('*with_log*.zip'))


def test_resumed_capture_appends_without_erasing_previous_failure(tmp_path):
    log = tmp_path / 'console.log'
    log.write_text('previous fresh022 validator failure\nChild process exit: 1\n', encoding='utf-8')
    command = [sys.executable, '-u', '-c', "print('second run')"]
    assert capture(command, log, cwd=tmp_path) == 0
    text = log.read_text(encoding='utf-8')
    assert 'previous fresh022 validator failure' in text and 'second run' in text
    assert recorded_child_exit(log) == 0


def test_corrupted_feedback_is_preserved_and_rejected(tmp_path):
    archive = tmp_path / 'feedback.zip'
    make_feedback(archive, tamper=True)
    before = archive.read_bytes()
    log = tmp_path / 'console.log'
    log.write_text('original error')
    with pytest.raises(ValueError, match='checksum'):
        attach_console(archive, log, child_exit=1)
    assert archive.read_bytes() == before


def test_missing_feedback_does_not_fabricate_a_successful_report(tmp_path):
    log = tmp_path / 'console.log'
    log.write_text('build failed')
    with pytest.raises(FileNotFoundError, match='did not export'):
        attach_console(tmp_path / 'missing.zip', log, child_exit=1)
    assert not (tmp_path / 'missing.zip').exists()


def test_missing_log_does_not_change_feedback_or_create_output(tmp_path):
    archive = tmp_path / 'feedback.zip'
    make_feedback(archive)
    before = archive.read_bytes()
    with pytest.raises(FileNotFoundError):
        attach_console(archive, tmp_path / 'missing.log', child_exit=None)
    assert archive.read_bytes() == before
    assert not list(tmp_path.glob('*with_log*.zip'))


def test_unknown_process_exit_stays_unknown_and_report_stays_failed(tmp_path):
    archive, log = tmp_path / 'feedback.zip', tmp_path / 'console.log'
    report = make_feedback(archive)
    log.write_text('Saved partial console output without an exit footer\n', encoding='utf-8')
    assert recorded_child_exit(log) is None
    combined = attach_console(archive, log, child_exit=None)
    with ZipFile(combined) as result:
        assert json.loads(result.read('capture_info.json'))['child_exit'] is None
        assert result.read('fresh/validation_report.json') == report


@pytest.mark.parametrize('footer,expected', [
    ('Child process exit: 2\n', 2),
    ('Child process exit: 1\r\n', 1),
    ('Child process exit: 1\nretry\nChild process exit: 0\n', 0),
    ('status=passed\n', None),
    ('prefix Child process exit: 0\n', None),
])
def test_only_actual_capture_footer_supplies_exit_code(tmp_path, footer, expected):
    log = tmp_path / 'console.log'
    log.write_bytes(footer.encode('utf-8'))
    assert recorded_child_exit(log) == expected


def test_package_only_uses_existing_logs_without_starting_any_process(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'project'
    output = 'artifacts/current'
    archive = root / (output + '_feedback.zip')
    log = root / output / 'console_run.log'
    log.parent.mkdir(parents=True)
    archive_bytes = make_feedback(archive)
    source_zip = archive.read_bytes()
    log.write_text('fresh022: explicit auditory clue requires audio_english_query\nChild process exit: 1\n',
                   encoding='utf-8')
    source_log = log.read_bytes()
    monkeypatch.setattr(capture_module, '__file__', str(root / 'experiments/namuwiki/capture.py'))

    def process_forbidden(*args, **kwargs):
        raise AssertionError('PackageOnly must not run Docker, PowerShell or analysis')

    monkeypatch.setattr(capture_module.subprocess, 'Popen', process_forbidden)
    assert main(['--package-only', '--output-dir', output]) == 0
    assert archive.read_bytes() == source_zip and log.read_bytes() == source_log
    combined, = archive.parent.glob('current_feedback_with_log_*.zip')
    with ZipFile(combined) as result:
        assert result.read('fresh/validation_report.json') == archive_bytes
        assert result.read('console_run.log') == source_log
        assert json.loads(result.read('capture_info.json'))['child_exit'] == 1
    text = capsys.readouterr().out
    assert str(combined) in text and 'No evaluation was run' in text


def test_package_only_failure_retains_original_and_reports_saved_log_path(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'project'
    output = 'artifacts/current'
    archive = root / (output + '_feedback.zip')
    archive.parent.mkdir(parents=True)
    make_feedback(archive)
    before = archive.read_bytes()
    monkeypatch.setattr(capture_module, '__file__', str(root / 'experiments/namuwiki/capture.py'))
    assert main(['--package-only', '--output-dir', output]) == 2
    assert archive.read_bytes() == before
    assert str(root / output / 'console_run.log') in capsys.readouterr().err


@pytest.mark.parametrize('other', ['--check-only', '--inspect'])
def test_package_only_cannot_be_mixed_with_an_execution_mode(other):
    with pytest.raises(SystemExit) as exc:
        main(['--package-only', other])
    assert exc.value.code == 2


@pytest.mark.parametrize('path', [
    'artifacts/context_fixed_followup_20261005', '../artifacts/new',
    'artifacts/../old', 'C:/Users/example', 'artifacts/name;command',
])
def test_invalid_or_previous_output_roots_are_rejected(path):
    with pytest.raises(ValueError):
        output_path(path)


def test_windows_slashes_and_new_root_are_accepted():
    assert output_path(r'artifacts\context_fixed_followup_20261005_cover_dance_fix') == (
        'artifacts/context_fixed_followup_20261005_cover_dance_fix'
    )


def test_offline_parser_inspection_shows_separate_real_modality_boundaries(capsys):
    rows = inspect_boundary()
    assert len(rows) == 4
    radio = next(row for row in rows if row['case'] == 'radio_and_bass')
    assert (radio['image'], radio['audio'], radio['context']) == (False, True, True)
    assert 'no Gemini call and no Qdrant access' in capsys.readouterr().out
