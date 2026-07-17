import json
import subprocess

import pytest

from hls4ml.backends import get_backend


def test_nanoxplore_backend_registered():
    b = get_backend('NanoXploreAccelerator')
    assert b is not None


def test_nanoxplore_backend_name_roundtrips():
    """hls4ml stores backend.name in the model config and calls get_backend()
    on it again (HLSConfig.__init__); the name must be the registered alias."""
    b = get_backend('NanoXploreAccelerator')
    assert b.name == 'NanoXploreAccelerator'
    assert get_backend(b.name) is b


def test_nanoxplore_backend_default_device():
    b = get_backend('NanoXploreAccelerator')
    assert b._default_device == 'nx2h540tsc'


def test_nanoxplore_create_initial_config_defaults():
    """NG-ULTRA defaults, not the inherited Bambu ones (Xilinx part, 5 ns)."""
    b = get_backend('NanoXploreAccelerator')
    cfg = b.create_initial_config()
    assert cfg['Part'] == 'nx2h540tsc'
    assert cfg['ClockPeriod'] == 20
    assert cfg['FPGAFamily'] == 'NanoXplore'


def test_nanoxplore_create_initial_config_overridable():
    b = get_backend('NanoXploreAccelerator')
    cfg = b.create_initial_config(clock_period=40)
    assert cfg['ClockPeriod'] == 40



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
