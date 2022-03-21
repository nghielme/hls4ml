//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_BATCHNORM_RESOURCE_H_
#define NNET_BATCHNORM_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void normalize_resource(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_in],
    typename CONFIG_T::scale_t  scale[CONFIG_T::n_scale_bias],
    typename CONFIG_T::bias_t   bias[CONFIG_T::n_scale_bias]
)
{
    data_T cache;

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=scale,bias

    // The configuration might limit reuse_factor, but just in case
    const int rufactor = CONFIG_T::reuse_factor < CONFIG_T::n_in ? CONFIG_T::reuse_factor : CONFIG_T::n_in;
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in, CONFIG_T::reuse_factor);

    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

    // Calcuate result
    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        int ires = ir;

        MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL

            if (ires < CONFIG_T::n_in) {
                if (CONFIG_T::n_filt==-1) {
                    res[ires] = CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::product(data[ires], scale[ires]) + bias[ires];
	            } else {
                    int norm_index = ires%CONFIG_T::n_filt;
                    res[ires] = CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::product(data[ires], scale[norm_index]) + bias[norm_index];
                }
            }
            // Increment in_index
            ires += rufactor;
        }
	}
}

}

#endif
