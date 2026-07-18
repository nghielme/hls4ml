from tensorflow import keras

import hls4ml


def _build_bambu_sh(tmp_path, io_type, suffix=''):
    model = keras.Sequential([keras.layers.Dense(4, input_shape=(4,), name='fc1')])
    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Bambu')
    out_dir = tmp_path / f'prj_{io_type}{suffix}'
    hmodel = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=str(out_dir), backend='Bambu', io_type=io_type
    )
    # csim=False, synth=False: reaches REQ_ARGS assembly and writes build_bambu.sh,
    # then takes the "dry run" return before invoking the real bambu binary.
    hmodel.build(csim=False, synth=False)
    return (out_dir / 'build_bambu.sh').read_text()


def test_default_uses_shipped_headers_no_m64(tmp_path):
    """Default: Bambu's shipped ac/ap headers (usr/include/panda), 32-bit
    triple. No local ac_types include, no -m64 — for BOTH io types. The
    absence of -m64 is what keeps io_stream clear of Bambu's InterfaceInfer
    64-bit pointer bug on ac_channel FIFOs."""
    for io_type in ('io_parallel', 'io_stream'):
        content = _build_bambu_sh(tmp_path, io_type)
        assert '-m64' not in content, io_type
        assert '-Ifirmware/ac_types' not in content, io_type


def test_legacy_ac_types_parallel_keeps_m64(tmp_path, monkeypatch):
    """USE_HLS4ML_AC_TYPES=1 restores the local-checkout headers + -m64."""
    monkeypatch.setenv('USE_HLS4ML_AC_TYPES', '1')
    content = _build_bambu_sh(tmp_path, 'io_parallel', suffix='_legacy')
    assert '-m64' in content
    assert '-Ifirmware/ac_types' in content


def test_legacy_ac_types_stream_still_omits_m64(tmp_path, monkeypatch):
    """Even on the legacy path, io_stream must not get -m64 (InterfaceInfer bug)."""
    monkeypatch.setenv('USE_HLS4ML_AC_TYPES', '1')
    content = _build_bambu_sh(tmp_path, 'io_stream', suffix='_legacy')
    assert '-m64' not in content
    assert '-Ifirmware/ac_types' in content
