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


import json
import tempfile


BRAM_V = """
module myproject_float (
  clock, reset, start_port, done_port,
  input_q0, input_q1, input_ce0, input_ce1, input_address0, input_address1,
  output_d0, output_d1, output_we0, output_we1, output_address0, output_address1,
  output_ce0, output_ce1
);
  input  clock; input  reset; input  start_port; output done_port;
  output [7:0] input_q0; output [7:0] input_q1;
  output input_ce0; output input_ce1;
  output [2:0] input_address0; output [2:0] input_address1;
  input  [7:0] output_d0; input  [7:0] output_d1;
  input  output_we0; input  output_we1;
  input  [3:0] output_address0; input  [3:0] output_address1;
  input  output_ce0; input  output_ce1;
endmodule
"""


def _make_project_dir(tmp_path, verilog, n_in=8, n_out=1):
    fw = tmp_path / 'firmware'
    fw.mkdir()
    (fw / 'myproject_float.h').write_text(
        f'static const unsigned N_IN = {n_in};\n'
        f'static const unsigned N_OUT = {n_out};\n'
    )
    (tmp_path / 'myproject_float.v').write_text(verilog)
    return tmp_path


def test_build_manifest_clock_period():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _build_manifest
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_project_dir(pathlib.Path(tmp), BRAM_V)
        m = _build_manifest(str(d), 'myproject', clock_period_ns=50.0, flow='parallel')
    assert m['clock_period_ns'] == 50.0
    assert abs(m['clock_mhz'] - 20.0) < 0.001


def test_build_manifest_flow():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _build_manifest
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_project_dir(pathlib.Path(tmp), BRAM_V)
        m = _build_manifest(str(d), 'myproject', clock_period_ns=50.0, flow='parallel')
    assert m['flow'] == 'parallel'


def test_build_manifest_rtl_files_parallel():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _build_manifest
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_project_dir(pathlib.Path(tmp), BRAM_V)
        m = _build_manifest(str(d), 'myproject', clock_period_ns=50.0, flow='parallel')
    assert 'top_parallel.v' in m['rtl_files']
    assert 'AXISlaveParallel.v' in m['rtl_files']
    assert 'top_stream.v' not in m['rtl_files']
    # complete list: HLS output + Bambu cell library included, so the private
    # side adds exactly these
    assert 'myproject_float.v' in m['rtl_files']
    assert 'panda_libtech.v' in m['rtl_files']


def test_build_manifest_mem_files_empty_for_parallel():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _build_manifest
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_project_dir(pathlib.Path(tmp), BRAM_V)
        m = _build_manifest(str(d), 'myproject', clock_period_ns=50.0, flow='parallel')
    assert m['mem_files'] == []


def test_build_manifest_n_words():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _build_manifest
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_project_dir(pathlib.Path(tmp), BRAM_V, n_in=8, n_out=1)
        m = _build_manifest(str(d), 'myproject', clock_period_ns=50.0, flow='parallel')
    assert m['n_words']['in'] == 8
    assert m['n_words']['out'] == 1


def test_build_manifest_version_1():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _build_manifest
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_project_dir(pathlib.Path(tmp), BRAM_V)
        m = _build_manifest(str(d), 'myproject', clock_period_ns=50.0, flow='parallel')
    assert m['manifest_version'] == 1


def test_build_manifest_device_none_default():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _build_manifest
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_project_dir(pathlib.Path(tmp), BRAM_V)
        m = _build_manifest(str(d), 'myproject', clock_period_ns=50.0, flow='parallel')
    assert m['device'] is None


def test_build_manifest_device_passthrough():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _build_manifest
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_project_dir(pathlib.Path(tmp), BRAM_V)
        m = _build_manifest(str(d), 'myproject', clock_period_ns=50.0,
                            flow='parallel', device='nx2h540tsc')
    assert m['device'] == 'nx2h540tsc'


def test_build_manifest_written_to_disk():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _build_manifest
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_project_dir(pathlib.Path(tmp), BRAM_V)
        _build_manifest(str(d), 'myproject', clock_period_ns=50.0, flow='parallel')
        data = json.loads((d / 'manifest.json').read_text())
    assert data['clock_period_ns'] == 50.0
    assert data['manifest_version'] == 1


def test_write_verilog_wrapper_appends_module():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _write_verilog_wrapper
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        (d / 'myproject_float.v').write_text(BRAM_V)
        _write_verilog_wrapper(str(d), 'myproject', 'parallel')
        content = (d / 'myproject_float.v').read_text()
    assert 'module myproject' in content
    assert 'myproject_float u0' in content


def test_write_verilog_wrapper_idempotent():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _write_verilog_wrapper
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        (d / 'myproject_float.v').write_text(BRAM_V)
        _write_verilog_wrapper(str(d), 'myproject', 'parallel')
        _write_verilog_wrapper(str(d), 'myproject', 'parallel')
        content = (d / 'myproject_float.v').read_text()
    assert content.count('module myproject (') == 1


def test_copy_rtl_templates_parallel():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _copy_rtl_templates
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        _copy_rtl_templates(str(d), 'parallel')
        assert (d / 'top_parallel.v').exists()
        assert (d / 'AXISlaveParallel.v').exists()
        assert not (d / 'top_stream.v').exists()


def test_copy_rtl_templates_stream():
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _copy_rtl_templates
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        _copy_rtl_templates(str(d), 'stream')
        assert (d / 'top_stream.v').exists()
        assert (d / 'AXISlaveStream.v').exists()
        assert not (d / 'top_parallel.v').exists()


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


def test_pll_markers_in_top_templates():
    """Both top templates carry exactly one HLS4ML PLL BEGIN/END region
    enclosing the default 50 MHz NX_PLL_U block (Task 3 splices there)."""
    rtl_dir = (pathlib.Path(__file__).parents[3]
               / 'hls4ml' / 'templates' / 'bambu_accelerator' / 'rtl')
    for fname in ('top_parallel.v', 'top_stream.v'):
        content = (rtl_dir / fname).read_text()
        assert content.count('// HLS4ML PLL BEGIN') == 1, fname
        assert content.count('// HLS4ML PLL END') == 1, fname
        region = content.split('// HLS4ML PLL BEGIN')[1].split('// HLS4ML PLL END')[0]
        assert 'NX_PLL_U' in region, fname
        assert 'clk_50_0mhz' in region, fname


def test_render_pll_block_25mhz():
    pytest.importorskip('ortools')
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import _render_pll_block
    block = _render_pll_block(40.0)  # 1000/40 = 25 MHz
    assert 'NX_PLL_U' in block
    assert 'clk_50_0mhz' in block          # fixed interface wire
    assert '(~rstn_i)' in block            # template reset polarity
    assert 'clk_25mhz' not in block        # generator's own name post-processed away
    assert 'locked_1' in block


def test_patch_pll_splices_region(tmp_path):
    pytest.importorskip('ortools')
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import (
        _RTL_TEMPLATES_DIR, _patch_pll,
    )
    import shutil as _sh
    _sh.copy2(_RTL_TEMPLATES_DIR / 'top_parallel.v', tmp_path / 'top_parallel.v')
    _patch_pll(str(tmp_path), 'parallel', 40.0)
    content = (tmp_path / 'top_parallel.v').read_text()
    assert content.count('// HLS4ML PLL BEGIN') == 1
    assert content.count('// HLS4ML PLL END') == 1
    region = content.split('// HLS4ML PLL BEGIN')[1].split('// HLS4ML PLL END')[0]
    assert '25' in region                  # 25 MHz config in comments/values
    assert 'clk_50_0mhz' in region
    assert 'generates 50.0 MHz' not in region  # default block replaced


def test_patch_pll_noop_at_50mhz(tmp_path):
    from hls4ml.backends.bambu_accelerator.bambu_accelerator_backend import (
        _RTL_TEMPLATES_DIR, _patch_pll,
    )
    import shutil as _sh
    _sh.copy2(_RTL_TEMPLATES_DIR / 'top_parallel.v', tmp_path / 'top_parallel.v')
    before = (tmp_path / 'top_parallel.v').read_text()
    _patch_pll(str(tmp_path), 'parallel', 20.0)   # 50 MHz -> keep default, no ortools needed
    assert (tmp_path / 'top_parallel.v').read_text() == before


def test_patch_pll_missing_ortools_fails_loud(tmp_path, monkeypatch):
    from hls4ml.backends.bambu_accelerator import bambu_accelerator_backend as bab
    import shutil as _sh
    _sh.copy2(bab._RTL_TEMPLATES_DIR / 'top_parallel.v', tmp_path / 'top_parallel.v')

    def _no_ortools(*a, **k):
        raise ImportError('No module named ortools')

    monkeypatch.setattr('hls4ml.backends.bambu_accelerator.pll_solver.solve_pll', _no_ortools)
    with pytest.raises(RuntimeError, match=r'hls4ml\[nanoxplore\]'):
        bab._patch_pll(str(tmp_path), 'parallel', 40.0)
