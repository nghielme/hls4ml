import pathlib
import pytest


def test_bambuaccelerator_not_registered():
    from hls4ml.backends import get_backend
    with pytest.raises(Exception):
        get_backend('BambuAccelerator')


def test_bambuaccelerator_cannot_be_instantiated_directly():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import BambuAcceleratorBackend
    with pytest.raises(TypeError):
        BambuAcceleratorBackend()


def test_bambuaccelerator_has_abstract_generate_bitstream():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import BambuAcceleratorBackend
    assert '_generate_bitstream' in BambuAcceleratorBackend.__abstractmethods__


def test_concrete_subclass_must_implement_generate_bitstream():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import BambuAcceleratorBackend

    class IncompleteBackend(BambuAcceleratorBackend):
        pass

    with pytest.raises(TypeError):
        IncompleteBackend()


def test_concrete_subclass_with_implementation_instantiates():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import BambuAcceleratorBackend

    class ConcreteBackend(BambuAcceleratorBackend):
        def __init__(self):
            pass

        def _generate_bitstream(self, model, project_dir, manifest):
            return {}

    b = ConcreteBackend()
    assert callable(b._generate_bitstream)


def test_rtl_templates_exist():
    import hls4ml
    rtl_dir = (pathlib.Path(hls4ml.__file__).parent
               / 'templates' / 'bambu_accelerator' / 'rtl')
    for fname in [
        'AXISlaveParallel.v', 'AXISlaveStream.v',
        'top_parallel.v', 'top_stream.v',
        'sfifo.v', 'axi_addr.v', 'skidbuffer.v',
    ]:
        assert (rtl_dir / fname).exists(), f"Missing: {fname}"
