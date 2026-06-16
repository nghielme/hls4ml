import json
import subprocess

import pytest

from hls4ml.backends import get_backend


def test_nanoxplore_backend_registered():
    b = get_backend('NanoXploreAccelerator')
    assert b is not None


def test_nanoxplore_backend_default_device():
    b = get_backend('NanoXploreAccelerator')
    assert b._default_device == 'nx2h540tsc'



def test_generate_bitstream_missing_command_raises():
    b = get_backend('NanoXploreAccelerator')
    b._bitstream_command = '/nonexistent/hls4ml-nanoxplore-bitstream'
    with pytest.raises(RuntimeError, match='not installed|not found'):
        b._generate_bitstream(object(), '/tmp', {'manifest_version': 1})


def test_generate_bitstream_calls_command(tmp_path, monkeypatch):
    report = {'returncode': 0, 'lut4': 1234, 'dff': 567}
    (tmp_path / 'report.json').write_text(json.dumps(report))

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(subprocess, 'run', fake_run)

    b = get_backend('NanoXploreAccelerator')
    b._bitstream_command = 'hls4ml-nanoxplore-bitstream'
    metrics = b._generate_bitstream(object(), str(tmp_path), {'manifest_version': 1})

    assert len(calls) == 1
    assert calls[0][0] == 'hls4ml-nanoxplore-bitstream'
    assert str(tmp_path) in calls[0]
    assert metrics['lut4'] == 1234
