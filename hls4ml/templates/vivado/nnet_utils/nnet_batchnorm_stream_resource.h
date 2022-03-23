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

#ifndef NNET_BATCHNORM_STREAM_RESOURCE_H_
#define NNET_BATCHNORM_STREAM_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_types.h"
#include "hls_stream.h"

namespace nnet {

// ****************************************************
//       Streaming Batch Normalization
// ****************************************************


template<class data_T, class res_T, typename CONFIG_T>
void normalize_resource_reg(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::scale_t scale[CONFIG_T::n_scale_bias],
    typename CONFIG_T::bias_t  bias[CONFIG_T::n_scale_bias]
) {
    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

    typename CONFIG_T::scale_t scale_part[data_T::size];
    typename CONFIG_T::bias_t  bias_part[data_T::size];
    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

    // maybe this can be done a bit better in the python
    const int rufactor_initial = (CONFIG_T::reuse_factor < data_T::size) ? CONFIG_T::reuse_factor : data_T::size;
    const int rufactor = (data_T::size % rufactor_initial == 0) ? rufactor_initial : data_T::size;
    const int block_factor = data_T::size / rufactor;

    BatchNormLoop: for (int i = 0; i < CONFIG_T::n_in / data_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        PrepWeights: for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            const int idx = i * data_T::size + j;
            scale_part[j] = scale[idx];
            bias_part[j] = bias[idx];
        }

        // Calcuate result
        ReuseLoop:
        for (int ib = 0; ib < data_T::size; ib += block_factor) {
            #pragma HLS PIPELINE II=1 rewind

            MultLoop:
            for (int im = 0; im < block_factor; im++) {
                #pragma HLS UNROLL
                const int ires = ib + im;
                out_data[ires] =
                    CONFIG_T::template product<typename data_T::value_type,
                                               typename CONFIG_T::scale_t>::product(in_data[ires], scale_part[ires])
                    + bias_part[ires];
            }
	    }

        res.write(out_data);
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void normalize_resource_conv(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::scale_t scale[CONFIG_T::n_scale_bias],
    typename CONFIG_T::bias_t  bias[CONFIG_T::n_scale_bias]
) {
    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

    // maybe this can be done a bit better in the python
    const int rufactor_initial = (CONFIG_T::reuse_factor < data_T::size) ? CONFIG_T::reuse_factor : data_T::size;
    const int rufactor = (data_T::size % rufactor_initial == 0) ? rufactor_initial : data_T::size;
    const int block_factor = data_T::size / rufactor;

    BatchNormLoop: for (int i = 0; i < CONFIG_T::n_in / data_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        // Calcuate result
        ReuseLoop:
        for (int ib = 0; ib < data_T::size; ib += block_factor) {
            #pragma HLS PIPELINE II=1 rewind

            MultLoop:
            for (int im = 0; im < block_factor; im++) {
                #pragma HLS UNROLL
                const int ires = ib + im;
                out_data[ires] =
                    CONFIG_T::template product<typename data_T::value_type,
                                               typename CONFIG_T::scale_t>::product(in_data[ires], scale[ires])
                    + bias[ires];
            }
	    }

        res.write(out_data);
    }
}

}

#endif
