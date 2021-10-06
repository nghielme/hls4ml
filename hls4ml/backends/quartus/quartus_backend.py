import numpy as np
import math
import os
import copy
import webbrowser
from calmjs.parse import es5
from calmjs.parse import asttypes
from tabulate import tabulate
from ast import literal_eval
from contextlib import contextmanager

from hls4ml.model.hls_types import NamedType, IntegerPrecisionType, FixedPrecisionType
from hls4ml.model.hls_layers import Layer, Dense, BatchNormalization, Activation, ParametrizedActivation, PReLU, Softmax
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer
from hls4ml.model.flow import register_flow
from hls4ml.backends import FPGABackend
from hls4ml.report import parse_quartus_report

@contextmanager
def chdir(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

class QuartusBackend(FPGABackend):
    def __init__(self):
        super(QuartusBackend, self).__init__('Quartus')
        self._register_flows()
    
    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        quartus_types = [
            'quartus:transform_types',
        ]
        quartus_types_flow = register_flow('specific_types', quartus_types, requires=[init_flow], backend=self.name)

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', templates, requires=[init_flow], backend=self.name)

        all_passes = get_backend_passes(self.name)

        extras = [
            # Ideally this should be empty
            opt_pass for opt_pass in all_passes if opt_pass not in initializers + quartus_types + templates
        ]

        if len(extras) > 0:
            extras_flow = register_flow('extras', extras, requires=[init_flow], backend=self.name)
        else:
            extras_flow = None

        ip_flow_requirements = ['optimize', init_flow, quartus_types_flow, extras_flow, template_flow]
        ip_flow_requirements = list(filter(None, ip_flow_requirements))

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self):
        return self._default_flow
    
    def create_initial_config(self, part='Arria10', clock_period=5, io_type='io_parallel'):
        config = {}

        config['Part'] = part if part is not None else 'Arria10'
        config['ClockPeriod'] = clock_period
        config['IOType'] = io_type
        config['HLSConfig'] = {}

        return config

    def get_precision_string_backend(self, precision):
        if isinstance(precision, IntegerPrecisionType):
            typestring = 'ac_int<{width}, {signed}>'.format(width=precision.width, signed='false' if not precision.signed else 'true')
        elif isinstance(precision, FixedPrecisionType):
            args = [precision.width, precision.integer, 'false' if not precision.signed else 'true', precision.rounding_mode, precision.saturation_mode]
            args = ','.join([str(arg) for arg in args if arg is not None])
            typestring = 'ac_fixed<{args}>'.format(args=args)
        else:
            typestring = precision
        return typestring

    def gen_quartus_weight_array(self, layer):
        rf = layer.get_attr('reuse_factor')
        block_factor = int((layer.attributes['n_in']*layer.attributes['n_out'])/rf)
        bf_rounded = int(pow(2, np.ceil(np.log2(block_factor))))
        rf_rounded = int(pow(2, np.ceil(np.log2(rf))))

        layer.weights['weight'].data = np.transpose(layer.weights['weight'].data).flatten()

        if(layer.attributes['n_in']*layer.attributes['n_out'] > 2048 and rf_rounded != rf):
            layer.set_attr('rfpad', rf_rounded-rf)
            layer.set_attr('bfpad', bf_rounded-block_factor)

            temp = np.empty([bf_rounded, rf_rounded])
            for i in range(rf_rounded):
                for j in range (bf_rounded):
                    if (i < rf and j < block_factor):
                        w_index = i + rf * j
                        temp[j][i] = layer.weights['weight'].data[w_index]
                    else:
                        temp[j][i] = 0
            layer.weights['weight'].data = temp.flatten()

        layer.weights['weight'].data_length = layer.weights['weight'].data.size
        return

    def validate_hls(self, config):
        if config.model_strategy.lower() == 'latency' and config.model_compression:
            print('WARNING: Compression enabled while model strategy set to "Latency".')
            print('WARNING: Changing model strategy to "Resource"')
            config.model_strategy = 'Resource'

    def build(self, model, synth=True, fpgasynth=False):
        """
        Builds the project using Intel HLS compiler.
        
        Users should generally not call this function directly but instead use `HLSModel.build()`.
        This function assumes the model was written with a call to `HLSModel.write()`

        Args:
            model (HLSModel): The model to build
            synth, optional: Whether to run synthesis
            fpgasynth, optional:  Whether to run fpga synthesis

        Errors raise exceptions
        """
        found = os.system('command -v i++ > /dev/null')
        if found != 0:
            raise Exception('Intel HLS installation not found. Make sure "i++" is on PATH.')

        with chdir(model.config.get_output_dir()):
            if synth:
                os.system('make {}-fpga'.format(model.config.get_project_name()))
                os.system('./{}-fpga'.format(model.config.get_project_name()))

            if fpgasynth:
                found = os.system('command -v quartus_sh > /dev/null')
                if found != 0:
                    raise Exception('Quartus installation not found. Make sure "quartus_sh" is on PATH.')
                os.chdir(model.config.get_project_name() + '-fpga.prj/quartus')
                os.system('quartus_sh --flow compile quartus_compile')
        
        return parse_quartus_report(model.config.get_output_dir())

    def get_supportedlayers(self):
        #Define supported laers
        core_layers = ['InputLayer', 'Dropout', 'Flatten', 'Reshape']
        dense_layers = ['Dense', 'BinaryDense', 'TernaryDense']
        conv_layers = ['Conv1D', 'Conv2D', 'BinaryConv2D']
        pooling_layers = ['MaxPooling1D', 'MaxPooling2D', 'AveragePooling1D', 'AveragePooling2D']
        norm_layers = ['BatchNormalization']
        activation_layers = ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU']
        merge_layers = ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate']
        qkeras_layers = ['QDense', 'QActivation', 'QConv1D', 'QConv2D']
        qkeras_dense = ['QDense', 'QActivation']
        #Define layers to skip for conversion to HLS
        skip_layers = ['Dropout', 'Flatten']
        #All supported layers
        return core_layers + dense_layers + norm_layers + activation_layers + qkeras_dense + skip_layers

    def get_pstring (self, width, intbits, signed=True, rounding_mode=None, saturation_mode=None, saturation_bits=None):
        decimal = width - intbits
        if decimal > 0:
            args = [width, intbits, 'false' if not signed else 'true', rounding_mode, saturation_mode]
            args = ', '.join([str(arg) for arg in args if arg is not None])
            return 'ac_fixed<{args}>'.format(args=args)
        else:
            return 'ac_int<{width}, {signed}>'.format(width=width, signed='false' if not signed else 'true')

    @layer_optimizer(Layer)
    def init_base_layer(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('reuse_factor', reuse_factor)

        target_cycles = layer.model.config.get_target_cycles(layer)
        layer.set_attr('target_cycles', target_cycles)

    @layer_optimizer(Dense)
    def init_dense(self, layer):
        index_t = IntegerPrecisionType(width=1, signed=False)

        layer.set_attr('rfpad', 0)
        layer.set_attr('bfpad', 0)

        if layer.model.config.get_compression(layer):
            layer.set_attr('strategy', 'compressed')
        else:
            self.set_closest_reuse_factor(layer)
            self.gen_quartus_weight_array(layer)
            layer.set_attr('strategy', 'resource')

        if layer.model.config.is_resource_strategy(layer):
            if layer.model.config.get_compression(layer):
                index_t = layer.get_weights('weight').type.index_precision

        layer.set_attr('index_t', NamedType('layer{}_index'.format(layer.index), index_t))

    @layer_optimizer(Activation)
    def init_activation(self, layer):
        if 'table_t' not in layer.attributes:
            layer.set_attr('table_t', NamedType(name=layer.name + '_table_t', precision=FixedPrecisionType(width=18, integer=8)))
        if 'table_size' not in layer.attributes:
            layer.set_attr('table_size', 1024)

    @layer_optimizer(Softmax)
    def init_softmax(self, layer):
        if 'exp_table_t' not in layer.attributes:
            layer.set_attr('exp_table_t', layer.get_attr('table_t'))
        if 'inv_table_t' not in layer.attributes:
            layer.set_attr('inv_table_t', layer.get_attr('table_t'))
