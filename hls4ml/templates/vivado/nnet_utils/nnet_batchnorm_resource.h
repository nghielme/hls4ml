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
#include <iostream>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void normalize_resource_reg (
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_in],
    typename CONFIG_T::scale_t  scale[CONFIG_T::n_scale_bias],
    typename CONFIG_T::bias_t   bias[CONFIG_T::n_scale_bias]
)
{
    data_T cache;

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=scale,bias

    // The configuration limits reuse_factor to divide n_in evenly (and be no larger than n_in), so these can be simplified:
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = CONFIG_T::n_in / CONFIG_T::reuse_factor;

    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

    // Calcuate result
    ReuseLoop:
    for (int ib = 0; ib < CONFIG_T::n_in; ib += block_factor) {
        #pragma HLS PIPELINE II=1 rewind

        MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            const int ires = ib + im;
            res[ires] = CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::product(data[ires], scale[ires]) + bias[ires];
        }
	}
}

template<class data_T, class res_T, typename CONFIG_T>
void normalize_resource_conv(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_in],
    typename CONFIG_T::scale_t  scale[CONFIG_T::n_scale_bias],
    typename CONFIG_T::bias_t   bias[CONFIG_T::n_scale_bias]
)
{
    data_T cache;

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=scale,bias

    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = CONFIG_T::n_in / CONFIG_T::reuse_factor;

    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

    // the configuration makes it so that either block_factor/n_scale_bias or n_scale_bias/block_factor divide evenly
    // note, n_scale_bias == n_filt
    if (block_factor <= CONFIG_T::n_scale_bias) {

        // Calcuate result
        ReuseLoop_lt:
        for (int ir = 0; ir < rufactor; ir++) {
            #pragma HLS PIPELINE II=1 rewind
            int filt = 0;
            int ires = ir;

            MultLoop_lt:
            for (int im = 0; im < block_factor; im++) {
                #pragma HLS UNROLL
                std::cout << "Lt branch, ires = " << ires << ", filt = " << filt << std::endl;
                res[ires] = CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::product(data[ires], scale[filt]) + bias[filt];
                // Increment in_index
                ires += rufactor;
            }
            filt += 1;
            if (filt == CONFIG_T::n_scale_bias) {
                filt = 0;
            }
        }

    } else {

        const int subblock_factor = block_factor / CONFIG_T::n_scale_bias;

        // Calcuate result
        ReuseLoop_gt:
        for (int ir = 0; ir < rufactor; ir++) {
            #pragma HLS PIPELINE II=1 rewind
            int ires = ir;

            filt_loop:
            for (int filt = 0; filt < CONFIG_T::n_scale_bias; filt++) {
                #pragma HLS UNROLL
                subloop:
                for (int sub = 0; sub < subblock_factor; sub++) {
                    #pragma HLS UNROLL
                    std::cout << "gt branch, ires = " << ires << ", filt = " << filt << std::endl;
                    res[ires] = CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::product(data[ires], scale[filt]) + bias[filt];
                    ires += rufactor;
                }
            }
        }
	}
}

}

#endif
