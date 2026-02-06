from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import ELU, Activation, Input, LeakyReLU, PReLU, ReLU, ThresholdedReLU
from tensorflow.keras.models import Model

import hls4ml

test_root_path = Path(__file__).parent


def _pytest_case_id(request):
    callspec = getattr(request.node, 'callspec', None)
    if callspec is not None:
        return callspec.id

    node_name = request.node.name
    if '[' in node_name and node_name.endswith(']'):
        return node_name.split('[', 1)[1][:-1]

    return node_name

# Variable 'name' is simply used as an identifier for the activation


@pytest.mark.parametrize('backend', ['Bambu', 'Vivado', 'Vitis', 'Catapult', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('shape, io_type', [((8,), 'io_parallel'), ((8,), 'io_stream'), ((8, 8, 3), 'io_stream')])
@pytest.mark.parametrize(
    'activation, name',
    [
        (ReLU(), 'relu'),
        (LeakyReLU(alpha=1.5), 'leaky_relu'),
        (Activation('leaky_relu'), 'leaky_relu_act'),
        (ThresholdedReLU(theta=0.75), 'threshold_relu'),
        (ELU(alpha=1.25), 'elu'),
        (Activation('selu'), 'selu'),
        # Tensorflow exception of multi-dimensional PReLU (8, 8, 3)
        (PReLU(alpha_initializer=tf.initializers.constant(0.25)), 'prelu'),
        (Activation('softplus'), 'softplus'),
        (Activation('softsign'), 'softsign'),
        (Activation(activation='tanh'), 'tanh'),
        (Activation('sigmoid'), 'sigmoid'),
        # Theano and Tensorflow might have different definitions for hard sigmoid
        # Result is likely to be different when |x| > 1 (see TF/Theano docs)
        (Activation('hard_sigmoid'), 'hard_sigmoid'),
    ],
)
def test_activations(backend, activation, name, shape, io_type, request):
    if name == 'prelu' and shape == (8, 8, 3):
        return
    # Subtract 0.5 to include negative values
    X = np.random.rand(1000, *shape) - 0.5

    shape_tag = 'x'.join(str(dim) for dim in shape)
    input = Input(shape=shape)
    activation = activation(input)
    keras_model = Model(inputs=input, outputs=activation)
    keras_prediction = keras_model.predict(X)

    input_data_tb = None
    output_data_tb = None
    if backend == 'Bambu':
        input_data_tb = test_root_path / f'tb_input_{shape_tag}_{io_type}_{name}.npy'
        output_data_tb = test_root_path / f'tb_output_{shape_tag}_{io_type}_{name}.npy'
        np.save(input_data_tb, X)
        np.save(output_data_tb, keras_prediction)

    hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name', backend=backend)
    output_dir = str(test_root_path / _pytest_case_id(request))

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        hls_config=hls_config,
        io_type=io_type,
        output_dir=output_dir,
        backend=backend,
        input_data_tb=str(input_data_tb) if input_data_tb is not None else None,
        output_data_tb=str(output_data_tb) if output_data_tb is not None else None,
    )
    hls_model.compile()
    if backend == 'Bambu':
        tb_file = f'{hls_model.config.get_project_name()}_test.cpp'
        hls_model.build(check=True, args=[f'--generate-tb={tb_file}', '--simulate', '--generate-interface=INFER', '--compiler=I386_CLANG16'])

    hls_prediction = hls_model.predict(X).reshape(keras_prediction.shape)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=2e-2, atol=2e-2)
