from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense

import hls4ml

test_root_path = Path(__file__).parent

@pytest.mark.parametrize(
    'backend, strategy',
    [
        ('Vivado', 'Latency'),
        ('Vivado', 'Resource'),
        ('Vitis', 'Latency'),
        ('Vitis', 'Resource'),
        ('Quartus', 'Resource'),
        ('oneAPI', 'Resource'),
        ('Bambu', 'Resource'),
        ('Catapult', 'Latency'),
        ('Catapult', 'Resource'),
    ],
)
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('shape', [(4, 3), (4, 1), (2, 3, 2), (1, 3, 1)])
def test_multi_dense(test_case_id, backend, strategy, io_type, shape):
    model = tf.keras.models.Sequential()
    model.add(Dense(7, input_shape=shape, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(100, *shape)
    X_input = np.round(X_input * 2**10) * 2**-10  # make it an exact ap_fixed<16,6>

    shapestr = '_'.join(str(x) for x in shape)
    keras_prediction = model.predict(X_input)

    input_data_tb = None
    output_data_tb = None
    if backend == 'Bambu':
        input_data_tb = test_root_path / f'tb_input_multi_dense_{io_type}_{shapestr}.npy'
        output_data_tb = test_root_path / f'tb_output_multi_dense_{io_type}_{shapestr}.npy'
        np.save(input_data_tb, X_input)
        np.save(output_data_tb, keras_prediction)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    config['Model']['Strategy'] = strategy
    output_dir = str(test_root_path / test_case_id)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
        input_data_tb=str(input_data_tb) if input_data_tb is not None else None,
        output_data_tb=str(output_data_tb) if output_data_tb is not None else None,
    )

    hls_model.compile()
    if backend == 'Bambu':
        tb_file = f'{hls_model.config.get_project_name()}_test.cpp'
        hls_model.build(check=True, args=[f'--generate-tb={tb_file}', '--simulate', '--generate-interface=INFER', '--compiler=I386_CLANG16'])

    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=1e-2, atol=0.01)
