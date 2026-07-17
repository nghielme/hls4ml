import pathlib

import pytest
from tensorflow import keras

import hls4ml


def _build_bambu_sh(tmp_path, io_type):
    model = keras.Sequential([keras.layers.Dense(4, input_shape=(4,), name='fc1')])
    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Bambu')
    out_dir = tmp_path / f'prj_{io_type}'
    hmodel = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=str(out_dir), backend='Bambu', io_type=io_type
    )
    # csim=False, synth=False: reaches REQ_ARGS assembly and writes build_bambu.sh,
    # then takes the "dry run" return before invoking the real bambu binary.
    hmodel.build(csim=False, synth=False)
    return (out_dir / 'build_bambu.sh').read_text()


def test_io_parallel_keeps_m64(tmp_path):
    content = _build_bambu_sh(tmp_path, 'io_parallel')
    assert '-m64' in content


def test_io_stream_omits_m64(tmp_path):
    content = _build_bambu_sh(tmp_path, 'io_stream')
    assert '-m64' not in content
