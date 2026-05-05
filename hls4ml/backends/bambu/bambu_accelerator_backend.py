import inspect
import os
import subprocess
from warnings import warn

from hls4ml.backends.bambu.bambu_backend import BambuBackend
from hls4ml.model.flow import register_flow
from hls4ml.model.optimizer import get_backend_passes
from hls4ml.model.optimizer.optimizer import extract_optimizers_from_path


class BambuAcceleratorBackend(BambuBackend):
    """Extends BambuBackend with a float wrapper around the ap_fixed HLS core.

    Generates additional files:
      - firmware/<proj>_float.h / .cpp  : flat float interface
      - <proj>_float_test.cpp           : float testbench
      - build_tb_float_exe.sh           : builds float testbench executable
      - build_lib.sh (overwritten)      : includes float wrapper in shared lib
    """

    def __init__(self):
        # Skip BambuBackend.__init__ to set our own name, then call FPGABackend.__init__
        super(BambuBackend, self).__init__(name='BambuAccelerator')
        self._register_layer_attributes()
        self._register_flows()

    def _init_file_optimizers(self):
        """Override to walk the full MRO and deduplicate passes directories.

        The default implementation only looks at direct bases + self:
          [*self.__class__.__bases__, self.__class__]
        For BambuBackend that's [FPGABackend, BambuBackend], so fpga/passes/ is
        correctly included. For BambuAcceleratorBackend the direct base is
        BambuBackend (not FPGABackend), so fpga/passes/ would be skipped and
        passes like clone_output/reshape_stream would be missing.

        We walk the full MRO (excluding object) and deduplicate by path so
        bambu/passes/ is only loaded once even though both BambuBackend and
        BambuAcceleratorBackend share the same directory.
        """
        file_optimizers = {}
        seen_paths = set()
        mro_classes = [c for c in type(self).__mro__ if c is not object]
        for cls in mro_classes:
            try:
                opt_path = os.path.dirname(inspect.getfile(cls)) + '/passes'
            except (TypeError, OSError):
                continue
            if opt_path in seen_paths:
                continue
            seen_paths.add(opt_path)
            module_path = cls.__module__[: cls.__module__.rfind('.')] + '.passes'
            cls_optimizers = extract_optimizers_from_path(opt_path, module_path, self)
            file_optimizers.update(cls_optimizers)
        return file_optimizers

    def _register_flows(self):
        bk = self.name.lower()  # 'bambuaccelerator'

        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        streaming_passes = [
            f'{bk}:inplace_stream_flatten',
            f'{bk}:reshape_stream',
            f'{bk}:clone_output',
            f'{bk}:insert_zero_padding_before_conv1d',
            f'{bk}:insert_zero_padding_before_conv2d',
            f'{bk}:broadcast_stream',
        ]
        streaming_flow = register_flow('streaming', streaming_passes, requires=[init_flow], backend=self.name)

        quantization_passes = [
            f'{bk}:merge_batch_norm_quantized_tanh',
            f'{bk}:quantize_dense_output',
            'fuse_consecutive_batch_normalization',
            f'{bk}:xnor_pooling',
        ]
        quantization_flow = register_flow('quantization', quantization_passes, requires=[init_flow], backend=self.name)

        optimization_passes = [
            f'{bk}:remove_final_reshape',
            f'{bk}:optimize_pointwise_conv',
            f'{bk}:inplace_parallel_reshape',
            f'{bk}:inplace_stream_flatten',
            f'{bk}:skip_softmax',
            f'{bk}:fix_softmax_table_size',
            'infer_precision_types',
            f'{bk}:distributed_arithmetic_codegen',
            f'{bk}:distributed_arithmetic_einsum_codegen',
            f'{bk}:fuse_quantizer_into_d_a_layers',
            f'{bk}:process_fixed_point_quantizer_layer',
        ]
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        bambu_types = [
            f'{bk}:transform_types',
            f'{bk}:register_bram_weights',
            f'{bk}:generate_conv_streaming_instructions',
            f'{bk}:apply_resource_strategy',
            f'{bk}:generate_conv_im2col',
            f'{bk}:generate_unrolled_dense_resource',
            f'{bk}:set_pipeline_style',
            f'{bk}:d_a_latency_dense_template',
            f'{bk}:d_a_latency_conv_template',
        ]
        bambu_types_flow = register_flow('specific_types', bambu_types, requires=[init_flow], backend=self.name)

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

        writer_passes = ['make_stamp', f'{bk}:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=[f'{bk}:ip'], backend=self.name)

        fifo_depth_opt_passes = [f'{bk}:fifo_depth_optimization'] + writer_passes
        register_flow('fifo_depth_optimization', fifo_depth_opt_passes, requires=[f'{bk}:ip'], backend=self.name)

        all_passes = get_backend_passes(self.name)

        extras = [
            opt_pass
            for opt_pass in all_passes
            if opt_pass
            not in initializers
            + streaming_passes
            + quantization_passes
            + optimization_passes
            + bambu_types
            + templates
            + writer_passes
            + fifo_depth_opt_passes
        ]

        if len(extras) > 0:
            for opt in extras:
                warn(f'WARNING: Optimizer "{opt}" is not part of any flow and will not be executed.')

        ip_flow_requirements = [
            'optimize',
            init_flow,
            streaming_flow,
            quantization_flow,
            optimization_flow,
            bambu_types_flow,
            template_flow,
        ]

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def _get_hls_sources(self, project_name):
        # myproject.cpp is still emitted for the CPU testbench (myproject_test.cpp)
        # but it must NOT be handed to Bambu: the accelerator writer inlines the full
        # layer pipeline directly into myproject_float.cpp, so Bambu only synthesises
        # a single top-level function with all DATAFLOW streams visible at the outer
        # scope.
        return [os.path.join('firmware', f'{project_name}_float.cpp')]

    def _get_top_fname(self, project_name):
        return f'{project_name}_float'

    def _get_cosim_testbench(self, project_name):
        return f'{project_name}_float_test.cpp'

    def build(
        self,
        model,
        *,
        reset=False,
        csim=True,
        synth=True,
        cosim=False,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=False,
        log_to_stdout=True,
        args=None,
        env=None,
        run_kwargs=None,
    ):
        """Run build, using the float testbench for C-simulation.

        Mirrors the BambuBackend.build() signature exactly.  csim=True compiles
        and runs the float testbench (*_float_tb.exe) instead of the ap_fixed
        one.  All other arguments (synth, cosim, vsynth, …) behave identically
        to BambuBackend.build().
        """
        # Replicate the parent's validation pre-check here because we suppress
        # csim=True when calling super() and the parent would otherwise raise.
        if validation and not (csim and cosim):
            raise ValueError("To validate C simulation & RTL simulation equality, csim and cosim must both be run.")

        # Pass csim=False so the parent never builds/runs the ap_fixed testbench.
        # Pass validation=False so the parent's own pre-check doesn't fire.
        result = super().build(
            model,
            reset=reset,
            csim=False,
            synth=synth,
            cosim=cosim,
            validation=False,
            export=export,
            vsynth=vsynth,
            fifo_opt=fifo_opt,
            log_to_stdout=log_to_stdout,
            args=args,
            env=env,
            run_kwargs=run_kwargs,
        )

        if csim:
            self._build_float_testbench_exe(model)

            project_name = model.config.get_project_name()
            stamp = model.config.get_config_value('Stamp')
            project_dir = model.config.get_output_dir()

            ret = subprocess.run(
                [f'./{project_name}-{stamp}_float_tb.exe'],
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if ret.returncode != 0:
                raise RuntimeError(
                    f'Float testbench execution failed:\nSTDOUT:\n{ret.stdout}\nSTDERR:\n{ret.stderr}'
                )

        return result

    def _build_float_testbench_exe(self, model):
        ret = subprocess.run(
            ['bash', 'build_tb_float_exe.sh'],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=model.config.get_output_dir(),
        )
        if ret.returncode != 0:
            raise RuntimeError(
                f'Failed to build float testbench executable for "{model.config.get_project_name()}":\n'
                f'STDOUT:\n{ret.stdout}\nSTDERR:\n{ret.stderr}'
            )
