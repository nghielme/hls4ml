import numpy as np
import math
import os
import ctypes
import platform
from bisect import bisect_left

from hls4ml.templates.templates import Backend
from hls4ml.model.hls_layers import IntegerPrecisionType, FixedPrecisionType

dense_config_template = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const bool store_weights_in_bram = false;
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
    typedef {index_t} index_t;
}};\n"""

batchnorm_config_template = """struct config{index} : nnet::batchnorm_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
    typedef {bias_t} bias_t;
    typedef {scale_t} scale_t;
}};\n"""

conv1d_config_template = """struct config{index} : nnet::conv1d_config {{
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned n_in = {n_in};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_width = {filt_width};
    static const unsigned n_filt = {n_filt};
    static const unsigned stride = {stride};
    static const unsigned dilation = {dilation};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const bool store_weights_in_bram = false;
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
    typedef {config_t} mult_config;
}};\n"""

conv_mult_config_template = """struct config{index}_mult : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
}};\n"""

conv2d_config_template = """struct config{index} : nnet::conv2d_config {{
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_height = {filt_height};
    static const unsigned filt_width = {filt_width};
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const bool store_weights_in_bram = false;
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
    typedef {config_t} mult_config;
}};\n"""

activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t} table_t;
}};\n"""

softmax_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {exp_table_t} exp_table_t;
    typedef {inv_table_t} inv_table_t;
}};\n"""

pooling1d_config_template = """struct config{index} : nnet::pooling1d_config {{
    static const unsigned n_in = {n_in};
    static const unsigned pool_size = {pool_size};
    static const unsigned n_out = {n_out};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned stride = {stride};
    static const nnet::Pool_Op pool_op = nnet::{pool_op};
}};\n"""

pooling2d_config_template = """struct config{index} : nnet::pooling2d_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned pool_height = {pool_height};
    static const unsigned pool_width = {pool_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const nnet::Pool_Op pool_op = nnet::{pool_op};
    static const unsigned reuse = {reuse};
}};\n"""

merge_config_template = """struct config{index} : nnet::merge_config {{
    static const unsigned n_elem = {n_elem};
}};\n"""

concat_config_template = """struct config{index} : nnet::concat_config {{
    static const unsigned n_elem1_0 = {n_elem1_0};
    static const unsigned n_elem1_1 = {n_elem1_1};
    static const unsigned n_elem1_2 = {n_elem1_2};
    static const unsigned n_elem2_0 = {n_elem2_0};
    static const unsigned n_elem2_1 = {n_elem2_1};
    static const unsigned n_elem2_2 = {n_elem2_2};

    static const unsigned axis = {axis};
}};\n"""

resize_config_template = """struct config{index} : nnet::resize_config {{
    static const unsigned height = {height};
    static const unsigned width = {width};
    static const unsigned n_chan = {n_chan};
    static const unsigned new_height = {new_height};
    static const unsigned new_width = {new_width};
}};\n"""

transpose_config_template = """struct config{index} : nnet::transpose_config {{
    static const unsigned depth = {depth};
    static const unsigned height = {height};
    static const unsigned width = {width};
    static const unsigned perm[3] = {{{perm_str}}};
}};\n"""

dense_function_template = 'nnet::dense_{strategy}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
batchnorm_function_template = 'nnet::normalize<{input_t}, {output_t}, {config}>({input}, {output}, {scale}, {bias});'
conv1d_function_template = 'nnet::conv_1d_{strategy}_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
conv2d_function_template = 'nnet::conv_2d_{strategy}_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {output});'
param_activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {param}, {output});'
pooling1d_function_template = 'nnet::pooling1d<{input_t}, {config}>({input}, {output});'
pooling2d_function_template = 'nnet::pooling2d_{data_format}<{input_t}, {config}>({input}, {output});'
merge_function_template = 'nnet::{merge}<{input1_t}, {input2_t}, {output_t}, {config}>({input1}, {input2}, {output});'
resize_function_template = 'nnet::resize_{algorithm}<{input_t}, {config}>({input}, {output});'
transpose_function_template = 'nnet::transpose{dim}<{input_t}, {config}>({input}, {output});'

dense_include_list = ['nnet_utils/nnet_dense.h', 'nnet_utils/nnet_dense_compressed.h', 'nnet_utils/nnet_dense_large.h']
batchnorm_include_list = ['nnet_utils/nnet_batchnorm.h']
conv1d_include_list = ['nnet_utils/nnet_conv.h', 'nnet_utils/nnet_conv_large.h']
conv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_conv2d_large.h']
activ_include_list = ['nnet_utils/nnet_activation.h']
pooling_include_list = ['nnet_utils/nnet_pooling.h']
merge_include_list = ['nnet_utils/nnet_merge.h']
resize_include_list = ['nnet_utils/nnet_image.h']
transpose_include_list = ['nnet_utils/nnet_array.h']

class VivadoBackend(Backend):
    def __init__(self):
        super(VivadoBackend, self).__init__('Vivado')
        self.register_templates('Dense', dense_function_template, dense_config_template, dense_include_list)
        self.register_templates('BinaryDense'            , dense_function_template,       dense_config_template, dense_include_list)
        self.register_templates('BatchNormalization'     , batchnorm_function_template,   batchnorm_config_template, batchnorm_include_list)
        self.register_templates('Conv1D'                 , conv1d_function_template,      [conv1d_config_template, conv_mult_config_template], conv1d_include_list)
        self.register_templates('Conv2D'                 , conv2d_function_template,      [conv2d_config_template, conv_mult_config_template], conv2d_include_list)
        self.register_templates('Activation'             , activ_function_template,       activ_config_template, activ_include_list)
        self.register_templates('ParametrizedActivation' , param_activ_function_template, activ_config_template, activ_include_list)
        self.register_templates('PReLU'                  , param_activ_function_template, activ_config_template, activ_include_list)
        self.register_templates('Softmax'                , activ_function_template,       softmax_config_template, activ_include_list)
        self.register_templates('Pooling1D'              , pooling1d_function_template,   pooling1d_config_template, pooling_include_list)
        self.register_templates('Pooling2D'              , pooling2d_function_template,   pooling2d_config_template, pooling_include_list)
        self.register_templates('Merge'                  , merge_function_template,       merge_config_template, merge_include_list)
        self.register_templates('Concatenate'            , merge_function_template,       concat_config_template, merge_include_list)
        self.register_templates('Resize'                 , resize_function_template,      resize_config_template, resize_include_list)
        self.register_templates('Transpose'              , transpose_function_template,   transpose_config_template, transpose_include_list)

    def get_valid_reuse_factors(self, layer):
        n_in = 0
        n_out = 0
        if layer.__class__.__name__ == 'Dense':
            n_in = layer.get_attr('n_in')
            n_out = layer.get_attr('n_out')
        elif layer.__class__.__name__ == 'Conv1D':
            n_in = layer.get_attr('n_chan') * layer.get_attr('filt_width')
            n_out = layer.get_attr('n_filt')
        elif layer.__class__.__name__ == 'Conv2D':
            n_in = layer.get_attr('n_chan') * layer.get_attr('filt_height') * layer.get_attr('filt_width')
            n_out = layer.get_attr('n_filt')

        max_rf = n_in * n_out
        valid_reuse_factors = []
        for rf in range(1, max_rf):
            _assert = self._check_conditions(n_in, n_out, rf)
            if _assert:
                valid_reuse_factors.append(rf)
        # Avoid using RF=1
        if valid_reuse_factors[0] == 1:
            valid_reuse_factors.pop(0)
        return valid_reuse_factors

    def _check_conditions(self, n_in, n_out, rf):
        multfactor = min(n_in, rf)
        multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
        #
        # THIS ASSERTION IS FOR THE FUNCTIONAL CORRECTNESS OF THE DENSE LAYER
        #
        _assert = (((multiplier_limit % n_out) == 0) or (rf >= n_in))
        _assert = _assert and (((rf % n_in) == 0) or (rf < n_in))
        #
        # THIS ASSERTION IS FOR QoR AND EXECUTION TIME OF VIVADO HLS
        #
        _assert = _assert and (((n_in * n_out) % rf) == 0)

        return _assert

    def get_closest_reuse_factor(self, valid_rf, chosen_rf):
        """
        Returns closest value to chosen_rf. valid_rf is sorted (obtained from get_valid_reuse_factors())
        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(valid_rf, chosen_rf)
        if pos == 0:
            return valid_rf[0]
        if pos == len(valid_rf):
            return valid_rf[-1]
        before = valid_rf[pos - 1]
        after = valid_rf[pos]
        if after - chosen_rf < chosen_rf - before:
            return after
        else:
            return before

    def set_closest_reuse_factor(self, layer):
        valid_rf = self.get_valid_reuse_factors(layer)
        chosen_rf = layer.reuse_factor
        if chosen_rf not in valid_rf:
            closest_rf = self.get_closest_reuse_factor(valid_rf, chosen_rf)
            print('WARNING: Invalid ReuseFactor={} with "Resource" strategy in layer "{}". Using ReuseFactor={} instead. Valid ReuseFactor(s): {}.'
                .format(chosen_rf, layer.name, closest_rf, ','.join(map(str, valid_rf))))
            layer.reuse_factor = closest_rf

    def get_precision_string_backend(self, precision):
        if isinstance(precision, IntegerPrecisionType):
            typestring = 'ap_{signed}int<{width}>'.format(signed='u' if not precision.signed else '', width=precision.width)
        elif isinstance(precision, FixedPrecisionType):
            args = [precision.width, precision.integer, precision.rounding_mode, precision.saturation_mode, precision.saturation_bits]
            args = ','.join([str(arg) for arg in args if arg is not None])
            typestring = 'ap_{signed}fixed<{args}>'.format(signed='u' if not precision.signed else '', args=args)
        else:
            typestring = precision
        return typestring

    def set_strategy(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.model.config.backend.set_closest_reuse_factor(layer)
            if layer.model.config.get_compression(layer):
                layer.set_attr('strategy', 'compressed')
            else:
                layer.set_attr('strategy', 'large')
        else:
            layer.set_attr('strategy', 'latency')

    def configure_weights(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            if not layer.model.config.get_compression(layer):
                layer.weights['weight'].data = np.transpose(layer.weights['weight'].data)

    def bn_weight_fuse(self, model, node):
        dense_node = node.get_input_node()

        dense_weight = dense_node.weights['weight']
        dense_bias = dense_node.weights['bias']

        bn_scale = node.weights['scale']
        bn_bias = node.weights['bias']

        if dense_node.get_attr('strategy') != 'large':
            fused_weight = bn_scale.data * dense_weight.data
        else:
            fused_weight = (bn_scale.data * dense_weight.data.T).T

        fused_bias = bn_scale.data * dense_bias.data + bn_bias.data

        model.remove_node(node, rewire=True)
        dense_node.weights['weight'].data = fused_weight
        dense_node.weights['bias'].data = fused_bias

    def validate_hls(self, config):
        use_resource = False
        if config.model_strategy.lower() == 'latency' and config.model_compression:
            print('WARNING: Compression enabled while model strategy set to "Latency".')
            use_resource = True
        for layer_type, strategy in config.layer_type_strategy.items():
            if strategy.lower() == 'resource' and config.model_strategy.lower() == 'latency':
                print('WARNING: Strategy for layer type {} set to "Resource", while model strategy set to "Latency".'.format(layer_type))
                use_resource = True

        for layer_name, strategy in config.layer_name_strategy.items():
            if strategy.lower() == 'resource' and config.model_strategy.lower() == 'latency':
                print('WARNING: Strategy for layer {} set to "Resource", while model strategy set to "Latency".'.format(layer_name))
                use_resource = True

        for layer_type, compression in config.layer_type_compression.items():
            if compression and config.model_strategy.lower() == 'latency':
                print('WARNING: Compression enabled for layer type {}, while model strategy set to "Latency".'.format(layer_type))
                use_resource = True

        for layer_name, compression in config.layer_name_compression.items():
            if compression and config.model_strategy.lower() == 'latency':
                print('WARNING: Compression enabled for layer {}, while model strategy set to "Latency".'.format(layer_name))
                use_resource = True

        if use_resource:
            print('WARNING: Changing model strategy to "Resource"')
            config.model_strategy = 'Resource'

    def compile(self, model):
        ret_val = os.system('bash build_lib.sh')
        if ret_val != 0:
            raise Exception('Failed to compile project "{}"'.format(model.config.get_project_name()))
        lib_name = 'firmware/{}.so'.format(model.config.get_project_name())
        if model._top_function_lib is not None:

            if platform.system() == "Linux":
                dlclose_func = ctypes.CDLL('libdl.so').dlclose
            elif platform.system() == "Darwin":
                dlclose_func = ctypes.CDLL('libc.dylib').dlclose

            dlclose_func.argtypes = [ctypes.c_void_p]
            dlclose_func.restype = ctypes.c_int
            dlclose_func(model._top_function_lib._handle)
        model._top_function_lib = ctypes.cdll.LoadLibrary(lib_name)

    def build(self, dir, reset=False, csim=True, synth=True, cosim=False, validation=False, export=False, vsynth=False):
        found = os.system('command -v vivado_hls > /dev/null')
        if found != 0:
            raise Exception('Vivado HLS installation not found. Make sure "vivado_hls" is on PATH.')
        curr_dir = os.getcwd()
        os.chdir(dir)
        os.system('vivado_hls -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} validation={validation} export={export} vsynth={vsynth}"'
            .format(reset=reset, csim=csim, synth=synth, cosim=cosim, validation=validation, export=export, vsynth=vsynth))
        os.chdir(curr_dir)

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
        return core_layers + dense_layers + conv_layers + pooling_layers + norm_layers + activation_layers + merge_layers + qkeras_layers + skip_layers

    def get_pstring (self, bits, integer, signed=True):
        decimal = bits - integer
        if decimal > 0:
            if signed:
                return 'ap_fixed<{},{}>'.format(bits, integer)
            else:
                return 'ap_ufixed<{},{}>'.format(bits, integer)
        else:
            if signed:
                return 'ap_int<{}>'.format(bits)
            else:
                return 'ap_uint<{}>'.format(bits)
