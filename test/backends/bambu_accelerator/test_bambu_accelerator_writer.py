
from tensorflow import keras

import hls4ml


def _write_float_test_cpp(tmp_path, io_type):
    model = keras.Sequential([
        keras.layers.Dense(4, activation='relu', input_shape=(4,), name='fc1'),
        keras.layers.Dense(2, name='out'),
    ])
    config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', backend='NanoXploreAccelerator', default_precision='ap_fixed<16,6>'
    )
    out_dir = tmp_path / f'prj_{io_type}'
    hmodel = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=str(out_dir),
        backend='NanoXploreAccelerator', io_type=io_type,
    )
    # write() alone runs the writer flow (incl. write_float_test_bench) without
    # needing Bambu, a C++ compiler, or NxMap.
    hmodel.write()
    proj = hmodel.config.get_project_name()
    return (out_dir / f'{proj}_float_test.cpp').read_text()


def test_io_stream_float_test_writes_back_after_first_read(tmp_path):
    content = _write_float_test_cpp(tmp_path, 'io_stream')
    # The fallback ("no tb_data files") code path has exactly one block that
    # both reads AND writes output_stream (the stdout block, fixed by this
    # task); the second block (file output) reads only.
    assert content.count('output_stream.write(raw_i)') >= 1
    assert content.count('output_stream.read()') >= 2


def test_io_parallel_float_test_unaffected(tmp_path):
    content = _write_float_test_cpp(tmp_path, 'io_parallel')
    assert 'output_stream' not in content
