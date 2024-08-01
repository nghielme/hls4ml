import struct

from hls4ml.converters.onnx_to_hls import get_onnx_attribute, onnx_handler


@onnx_handler('Transpose')
def parse_transpose_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Transpose'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    perm = [list(i.ints) for i in node.attribute][0]  # This will get something like [[a,b,c]][0] = [a,b,c]
    layer['perm'] = [x - 1 for x in perm[1:]]  # Ignore the batch dimension in ONNX, and adjust the perm indexing

    return layer


@onnx_handler('Reshape')
def parse_reshape_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    return layer


@onnx_handler('Flatten')
def parse_flatten_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)
    layer['target_shape'] = [-1]  # does not contain batch dimension

    return layer


@onnx_handler('Resize')
def parse_resize_layer(node, input_names, input_shapes, graph):
    # def get_scales(graph, name, scale_shapes):
    #     for i in graph.initializer:
    #         if i.name == name:
    #             fmt = "<"
    #             for _ in range(scale_shapes):
    #                 fmt += "f"
    #             return struct.unpack(fmt, i.raw_data)

    # scales = get_scales(graph, node.input[-1], input_shapes[-1][0])
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Resize'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)
    layer['in_height'] = input_shapes[0][2]
    layer['in_width'] = input_shapes[0][1]
    # layer['out_width'] = int(input_shapes[0][1] * scales[1])
    # layer['out_height'] = int(input_shapes[0][2] * scales[2])
    # layer['n_chan'] = int(input_shapes[0][3] * scales[3])
    layer['out_width'] = input_shapes[0][1]
    layer['out_height'] = input_shapes[0][2]
    layer['n_chan'] = input_shapes[0][3]
    layer['algorithm'] = get_onnx_attribute(node, 'mode')

    return layer
