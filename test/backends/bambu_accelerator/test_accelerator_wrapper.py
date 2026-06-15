import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[4]))

from hls4ml.backends.bambu_accelerator.wrapper import (
    parse_module, detect_flow, build_rename_map,
    generate_wrapper_verilog, extract_data_widths, generate_top_localparams,
)

BRAM_VERILOG = """
module myproject_float (
  clock, reset, start_port, done_port,
  input_q0, input_q1, input_ce0, input_ce1,
  input_address0, input_address1,
  output_d0, output_d1, output_we0, output_we1,
  output_address0, output_address1,
  output_ce0, output_ce1
);
  input  clock;
  input  reset;
  input  start_port;
  output done_port;
  output [7:0] input_q0;
  output [7:0] input_q1;
  output input_ce0;
  output input_ce1;
  output [2:0] input_address0;
  output [2:0] input_address1;
  input  [7:0] output_d0;
  input  [7:0] output_d1;
  input  output_we0;
  input  output_we1;
  input  [3:0] output_address0;
  input  [3:0] output_address1;
  input  output_ce0;
  input  output_ce1;
endmodule
"""

AXIS_VERILOG = """
module myproject_float (
  clock, reset, start_port, done_port,
  input_stream_TDATA, input_stream_TVALID, input_stream_TREADY,
  output_stream_TDATA, output_stream_TVALID, output_stream_TREADY
);
  input  clock;
  input  reset;
  input  start_port;
  output done_port;
  input  [15:0] input_stream_TDATA;
  input  input_stream_TVALID;
  output input_stream_TREADY;
  output [63:0] output_stream_TDATA;
  output output_stream_TVALID;
  input  output_stream_TREADY;
endmodule
"""


def test_parse_module_bram_name():
    name, _, _ = parse_module(BRAM_VERILOG)
    assert name == 'myproject_float'

def test_parse_module_bram_has_clock_port():
    _, ports, _ = parse_module(BRAM_VERILOG)
    assert 'clock' in ports

def test_detect_flow_bram_returns_parallel():
    _, ports, _ = parse_module(BRAM_VERILOG)
    assert detect_flow(ports) == 'parallel'

def test_detect_flow_axis_returns_stream():
    _, ports, _ = parse_module(AXIS_VERILOG)
    assert detect_flow(ports) == 'stream'

def test_rename_map_axis_tdata():
    _, ports, decls = parse_module(AXIS_VERILOG)
    rmap = build_rename_map(ports, decls, 'stream')
    assert rmap['input_stream_TDATA'] == 'hls_in_tdata'
    assert rmap['output_stream_TDATA'] == 'hls_out_tdata'

def test_rename_map_axis_tvalid_tready():
    _, ports, decls = parse_module(AXIS_VERILOG)
    rmap = build_rename_map(ports, decls, 'stream')
    assert rmap['input_stream_TVALID'] == 'hls_in_tvalid'
    assert rmap['output_stream_TREADY'] == 'hls_out_tready'

def test_rename_map_standard_ports_absent():
    _, ports, decls = parse_module(BRAM_VERILOG)
    rmap = build_rename_map(ports, decls, 'parallel')
    assert 'clock' not in rmap

def test_extract_data_widths_bram():
    _, ports, decls = parse_module(BRAM_VERILOG)
    in_dw, out_dw = extract_data_widths(ports, decls, 'parallel')
    assert in_dw == 8
    assert out_dw == 8

def test_extract_data_widths_axis():
    _, ports, decls = parse_module(AXIS_VERILOG)
    in_dw, out_dw = extract_data_widths(ports, decls, 'stream')
    assert in_dw == 16
    assert out_dw == 64

def test_generate_wrapper_verilog_module_name():
    _, ports, decls = parse_module(BRAM_VERILOG)
    verilog = generate_wrapper_verilog('myproject_float', ports, decls, 'parallel')
    assert 'module myproject' in verilog
    assert 'endmodule' in verilog

def test_generate_wrapper_verilog_instantiation():
    _, ports, decls = parse_module(BRAM_VERILOG)
    verilog = generate_wrapper_verilog('myproject_float', ports, decls, 'parallel')
    assert 'myproject_float u0' in verilog

def test_generate_wrapper_verilog_axis_renamed_ports():
    _, ports, decls = parse_module(AXIS_VERILOG)
    verilog = generate_wrapper_verilog('myproject_float', ports, decls, 'stream')
    assert 'hls_in_tdata' in verilog
    assert 'hls_out_tdata' in verilog

def test_generate_top_localparams_parallel_has_addr_w():
    lp = generate_top_localparams(8, 4, 8, 1, 'parallel')
    assert 'HLS_IN_ADDR_W' in lp
    assert 'HLS_OUT_ADDR_W' in lp

def test_generate_top_localparams_stream_no_addr_w():
    lp = generate_top_localparams(16, 8, 64, 1, 'stream')
    assert 'HLS_IN_ADDR_W' not in lp
