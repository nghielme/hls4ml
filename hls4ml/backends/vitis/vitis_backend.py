import os
import sys

from hls4ml.backends import XilinxBackend
from hls4ml.model.flow import register_flow
from hls4ml.report import parse_vivado_report
from hls4ml.model.optimizer import get_backend_passes


class VitisBackend(XilinxBackend):
    def __init__(self, name='Vitis'):
        super().__init__(name)


    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        validation_passes = [
            'vitis:validate_conv_implementation',
            'vitis:validate_strategy',
        ]
        validation_flow = register_flow('validation', validation_passes, requires=['vitis:init_layers'], backend=self.name)

        streaming_passes = [
            'vitis:reshape_stream',
            'vitis:clone_output',
            'vitis:insert_zero_padding_before_conv1d',
            'vitis:insert_zero_padding_before_conv2d',
            'vitis:broadcast_stream',
        ]
        streaming_flow = register_flow('streaming', streaming_passes, requires=[init_flow], backend=self.name)

        quantization_passes = [
            'vitis:merge_batch_norm_quantized_tanh',
            'vitis:quantize_dense_output',
            'fuse_consecutive_batch_normalization',
        ]
        quantization_flow = register_flow('quantization', quantization_passes, requires=[init_flow], backend=self.name)

        optimization_passes = ['vitis:remove_final_reshape', 'vitis:optimize_pointwise_conv', 'vitis:skip_softmax']
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        vivado_types = [
            'vitis:transform_types',
            'vitis:register_bram_weights',
            'vitis:generate_conv_streaming_instructions',
            'vitis:apply_resource_strategy',
            'vitis:generate_conv_im2col',
        ]
        vivado_types_flow = register_flow('specific_types', vivado_types, requires=[init_flow], backend=self.name)

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

        writer_passes = ['make_stamp', 'vitis:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['vitis:ip'], backend=self.name)

        fifo_depth_opt_passes = [
            'vitis:fifo_depth_optimization'
        ] + writer_passes  # After optimization, a new project will be written

        register_flow('fifo_depth_optimization', fifo_depth_opt_passes, requires=[self._writer_flow], backend=self.name)

        all_passes = get_backend_passes(self.name)

        extras = [
            # Ideally this should be empty
            opt_pass
            for opt_pass in all_passes
            if opt_pass
            not in initializers
            + streaming_passes
            + quantization_passes
            + optimization_passes
            + vivado_types
            + templates
            + writer_passes
            + fifo_depth_opt_passes
        ]

        if len(extras) > 0:
            extras_flow = register_flow('extras', extras, requires=[init_flow], backend=self.name)
        else:
            extras_flow = None

        ip_flow_requirements = [
            'optimize',
            init_flow,
            validation_flow,
            streaming_flow,
            quantization_flow,
            optimization_flow,
            vivado_types_flow,
            extras_flow,
            template_flow,
        ]
        ip_flow_requirements = list(filter(None, ip_flow_requirements))

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def build(self, model, reset=False, csim=True, synth=True, cosim=False, validation=False, export=False, vsynth=False):
        if 'linux' in sys.platform:
            found = os.system('command -v vitis_hls > /dev/null')
            if found != 0:
                raise Exception('Vitis HLS installation not found. Make sure "vitis_hls" is on PATH.')
        
        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())
        os.system('vitis_hls -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} validation={validation} export={export} vsynth={vsynth}"'
            .format(reset=reset, csim=csim, synth=synth, cosim=cosim, validation=validation, export=export, vsynth=vsynth))
        os.chdir(curr_dir)

        return parse_vivado_report(model.config.get_output_dir())