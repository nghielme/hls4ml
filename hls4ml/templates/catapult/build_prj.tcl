#################
#    HLS4ML
#################
array set opt {
  reset      0
  csim       1
  synth      1
  cosim      1
  validation 1
  export     0
  vsynth     1
}

foreach arg $::argv {
  foreach {optname optval} [split $arg '='] {}
  if { [info exists opt($optname)] } {
    set opt($optname) [string is true -strict $optval]
  }
}

puts "***** INVOKE OPTIONS *****"
foreach x [lsort [array names opt]] {
  puts "[format {   %-20s %s} $x $opt($x)]"
}
puts ""

proc report_time { op_name time_start time_end } {
  set time_taken [expr $time_end - $time_start]
  set time_s [expr ($time_taken / 1000) % 60]
  set time_m [expr ($time_taken / (1000*60)) % 60]
  set time_h [expr ($time_taken / (1000*60*60)) % 24]
  puts "***** ${op_name} COMPLETED IN ${time_h}h${time_m}m${time_s}s *****"
}

options set Input/CppStandard {c++11}

if {$opt(reset)} {
  project load myproject_prj.ccs
  go new
} else {
  project new -name myproject_prj
}

solution file add firmware/myproject.cpp
solution file add myproject_test.cpp -exclude true

flow package require /SCVerify
# Ideally, the path to the weights/testbench data should be runtime configurable
# instead of compile-time WEIGHTS_DIR macro. If the nnet_helpers.h load_ functions
# are ever enhanced to take a path option then this setting can be used.
# flow package option set /SCVerify/INVOKE_ARGS {firmware/weights tb_data}

directive set -DESIGN_HIERARCHY myproject

go compile

if {$opt(csim)} {
  puts "***** C SIMULATION *****"
  set time_start [clock clicks -milliseconds]
  flow run /SCVerify/launch_make ./scverify/Verify_orig_cxx_osci.mk {} SIMTOOL=osci sim
  set time_end [clock clicks -milliseconds]
  report_time "C SIMULATION" $time_start $time_end
}

if {$opt(synth)} {
  puts "***** C/RTL SYNTHESIS *****"
  set time_start [clock clicks -milliseconds]

  solution library add mgc_Xilinx-KINTEX-u-2_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family KINTEX-u -speed -2 -part xcku115-flvb2104-2-i
  solution library add Xilinx_RAMS
  solution library add Xilinx_ROMS
  solution library add Xilinx_FIFO

  directive set -CLOCKS {clk {-CLOCK_PERIOD 5 -CLOCK_EDGE rising -CLOCK_HIGH_TIME 2.5 -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND sync -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high}}

  go extract
  set time_end [clock clicks -milliseconds]
  report_time "C/RTL SYNTHESIS" $time_start $time_end
}

project save

if {$opt(cosim) || $opt(validation)} {
  flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {} SIMTOOL=msim sim
}

if {$opt(export)} {
  puts "***** EXPORT IP *****"
  set time_start [clock clicks -milliseconds]
# Not yet implemented
#  flow package option set /Vivado/BoardPart xilinx.com:zcu102:part0:3.1
#  flow package option set /Vivado/IP_Taxonomy {/Catapult}
#  flow run /Vivado/launch_package_ip -shell ./vivado_concat_v/concat_v_package_ip.tcl
  set time_end [clock clicks -milliseconds]
  report_time "EXPORT IP" $time_start $time_end
}

if {$opt(vsynth)} {
  puts "***** VIVADO SYNTHESIS *****"
  set time_start [clock clicks -milliseconds]
  flow run /Vivado/synthesize -shell vivado_concat_v/concat_rtl.v.xv
  set time_end [clock clicks -milliseconds]
  report_time "VIVADO SYNTHESIS" $time_start $time_end
}

