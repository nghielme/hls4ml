import os
import urllib
from pathlib import Path

import numpy as np
import pytest
import qonnx.core.onnx_exec as oxe
import qonnx.util.cleanup
import qonnx.util.to_channels_last

# To conveniently run QONNX inference
from qonnx.core.modelwrapper import ModelWrapper

import hls4ml
from qonnx.core.onnx_exec import execute_onnx

import hls4ml.converters


test_root_path = Path(__file__).parent

@pytest.fixture(scope='module')
def sep_conv_model():
    """
    Load separabale conv model
    """
    dl_dir = test_root_path
    dl_file = str(dl_dir / "qonnx-sep-conv-ch-last.onnx")
    sep_conv_qonnx_url = (
        "https://raw.githubusercontent.com/fastmachinelearning"
        "/example-models/qonnx-new-models/onnx/branched_model_ch_last.onnx"
    )
    urllib.request.urlretrieve(sep_conv_qonnx_url, dl_file)
    assert os.path.isfile(dl_file)
    out_file = str(dl_dir / "qonnx-sep-conv-ch-last-clean.onnx")
    
    # cleanup
    qonnx.util.cleanup.cleanup(dl_file, out_file=out_file)
    model = ModelWrapper(out_file)

    return model

@pytest.fixture(scope='module')
def tfc_2w2a_model():
    '''
    Load the tiny fully-connected model
    '''
    dl_dir = test_root_path
    dl_file = str(dl_dir / "qonnx-tfc-2w2a.onnx")
    tfc_w2a2_qonnx_url = (
        "https://raw.githubusercontent.com/fastmachinelearning/"
        "QONNX_model_zoo/main/models/MNIST/Brevitas_FINN_TFC/TFC/TFC_2W2A.onnx"
    )
    urllib.request.urlretrieve(tfc_w2a2_qonnx_url, dl_file)
    assert os.path.isfile(dl_file)
    out_file = str(dl_dir / "qonnx-tfc-2w2a-clean.onnx")

    # cleanup
    qonnx.util.cleanup.cleanup(dl_file, out_file=out_file)
    model = ModelWrapper(out_file)
    return model


@pytest.fixture(scope='module')
def cnv_2w2a_model():
    '''
    Load the small convolution model
    '''
    dl_dir = test_root_path
    dl_file = str(dl_dir / "qonnx-cnv-2w2a.onnx")
    cnv_w2a2_qonnx_url = (
        "https://raw.githubusercontent.com/fastmachinelearning/"
        "QONNX_model_zoo/main/models/CIFAR10/Brevitas_FINN_CNV/CNV_2W2A.onnx"
    )
    urllib.request.urlretrieve(cnv_w2a2_qonnx_url, dl_file)
    assert os.path.isfile(dl_file)
    out_clean = str(dl_dir / "qonnx-cnv-2w2a-clean.onnx")
    out_chanlast = str(dl_dir / "qonnx-cnv-2w2a-clean-channels-last.onnx")
    out_file = str(dl_dir / "qonnx-cnv-2w2a-clean-channels-last-clean.onnx")

    # cleanup
    qonnx.util.cleanup.cleanup(dl_file, out_file=out_clean)
    qonnx.util.to_channels_last.to_channels_last(out_clean, make_input_channels_last=True, out_file=out_chanlast)
    qonnx.util.cleanup.cleanup(out_chanlast, out_file=out_file)
    model = ModelWrapper(out_file)
    return model


@pytest.fixture(scope='module')
def jettagging_model():
    '''
    Load the 3 hidden layer QKeras example model trained on the jet tagging dataset
    '''
    dl_dir = test_root_path
    dl_file = str(dl_dir / "qkeras_jettagging.onnx")
    jet_tagging_qonnx_url = (
        "https://raw.githubusercontent.com/fastmachinelearning/"
        "QONNX_model_zoo/main/models/JetTagging/QKeras_hls4ml_3layer/qkeras_jettagging.onnx"
    )
    urllib.request.urlretrieve(jet_tagging_qonnx_url, dl_file)
    assert os.path.isfile(dl_file)
    out_file = str(dl_dir / "qkeras_jettagging-clean.onnx")

    # cleanup
    qonnx.util.cleanup.cleanup(dl_file, out_file=out_file)
    model = ModelWrapper(out_file)
    return model

@pytest.mark.parametrize('backend', ['Vitis'])
def test_sep_conv(sep_conv_model, backend):
    model = sep_conv_model
    ishape = tuple(model.get_tensor_shape(model.graph.input[0].name))
    X = np.random.uniform(low=0, high=1, size=np.prod(ishape)).reshape(ishape)
    # X = (np.round(X * 2**16) * 2**-16).astype(np.float32)
    pred = execute_onnx(model, {model.graph.input[0].name:X})
    y_qonnx = pred[model.graph.output[0].name]

    config = hls4ml.utils.config.config_from_onnx_model(model, granularity='name', backend='Vitis', default_precision='ap_fixed<16,6>')
    
    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir='sep_conv_test_hls4ml',
        io_type='io_stream',
        backend='Vitis',
        hls_config=config
    )
    hls_model.compile()
    y_hls4ml = hls_model.predict(np.ascontiguousarray(X))

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)
    print('test')

@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
def test_tfc_2w2a(tfc_2w2a_model, backend):
    model = tfc_2w2a_model

    ishape = (1, 1, 28, 28)
    X = np.random.uniform(low=-1, high=+1, size=np.prod(ishape)).reshape(ishape)
    X = (np.round(X * 2**16) * 2**-16).astype(np.float32)

    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    # Convert QONNX model, compile, and run inference
    config = hls4ml.utils.config.config_from_onnx_model(model, backend=backend, default_precision='fixed<32,16>')
    hls_model = hls4ml.converters.convert_from_onnx_model(
        model, output_dir=str(test_root_path / f'hls4mlprj_qonnx_tfc-2w2a_{backend}'), backend=backend, hls_config=config
    )
    hls_model.compile()
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)


@pytest.mark.parametrize('backend', ['Vitis'])
def test_cnv_2w2a(cnv_2w2a_model, backend):
    """
    This tests a convolution model. Note:  the batch normalizations weights not quantized, so it is
    difficult to make this match perfectly. It is also a slow test, which is why only Vitis is tested.
    """
    model = cnv_2w2a_model

    ishape = (1, 32, 32, 3)
    X = np.random.uniform(low=-1, high=+1, size=np.prod(ishape)).reshape(ishape)
    X = (np.round(X * 2**6) * 2**-6).astype(np.float32)
    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    # Convert QONNX model, compile, and run inference
    config = hls4ml.utils.config.config_from_onnx_model(model, backend=backend, default_precision='fixed<32,6>')
    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir=str(test_root_path / f'hls4mlprj_qonnx_cnv-2w2a_{backend}'),
        io_type='io_stream',
        backend=backend,
        hls_config=config,
    )
    hls_model.compile()
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
def test_jet_tagging(jettagging_model, backend):
    model = jettagging_model

    # Execute QONNX model inference
    # TODO make the test bigger
    ishape = (1, 16)
    X = np.random.uniform(low=-1, high=+1, size=np.prod(ishape)).reshape(ishape)
    X = (np.round(X * 2**16) * 2**-16).astype(np.float32)
    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    # Convert QONNX model, compile, and run inference
    config = hls4ml.utils.config_from_onnx_model(model, backend=backend, default_precision='fixed<32,16>')

    hls_model = hls4ml.converters.convert_from_onnx_model(
        model, output_dir=str(test_root_path / f'hls4mlprj_qonnx_jettag_{backend}'), backend=backend, hls_config=config
    )
    hls_model.compile()
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)
