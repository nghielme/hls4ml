from hls4ml.model.layers import Resize
from hls4ml.model.optimizer import OptimizerPass

def register_match_quantizer_resize(backend):
    backend.register_pass('match_quantizer_resize', MatchQuantizerResize)

class MatchQuantizerResize(OptimizerPass):
    def match(self, node):
        if isinstance(node, Resize) and node.get_input_variable().type.precision != node.get_output_variable().type.precision:
            return True
        else:
            return False

    def transform(self, model, node):
        node.get_input_variable().type.precision = node.get_output_variable().type.precision
        return True