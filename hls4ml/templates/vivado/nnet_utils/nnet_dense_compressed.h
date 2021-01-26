//
//    hls4ml: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2018 Giuseppe Di Guglielmo
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

#ifndef NNET_COMPRESSED_LAYER_H_
#define NNET_COMPRESSED_LAYER_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void dense_compressed(
        data_T    data[CONFIG_T::n_in],
        res_T     res[CONFIG_T::n_out],
        typename CONFIG_T::weight_t  weights[CONFIG_T::n_nonzeros],
        typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    //#pragma HLS function_instantiate variable=weights,biases

    typename CONFIG_T::accum_t acc [CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc    complete
    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_RESHAPE   variable=weights cyclic factor=CONFIG_T::compressed_block_factor
    //if (CONFIG_T::store_weights_in_bram){
    //#pragma HLS RESOURCE variable=weights core=ROM_1P_BRAM
    #pragma HLS data_pack variable=weights struct_level
    //}

    InitAccum:
    for(unsigned i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS UNROLL
        acc[i] = (typename CONFIG_T::accum_t) (biases[i]);
    }

    // Do the compressed matrix-multiply
    ReuseLoop:
    for(unsigned ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
        #pragma HLS PIPELINE  II=1 rewind

        typename CONFIG_T::accum_t mult[CONFIG_T::compressed_block_factor];
        #pragma HLS ARRAY_PARTITION variable=mult complete

	decltype(CONFIG_T::weight_t::col_index) col[CONFIG_T::compressed_block_factor];
        #pragma HLS ARRAY_PARTITION variable=col complete

        CompressedMultLoop:
        for(unsigned im = 0; im < CONFIG_T::compressed_block_factor; im++) {
            #pragma HLS UNROLL
            unsigned w = ir + CONFIG_T::reuse_factor * im;
	    mult[im] =
	      CONFIG_T::template product<data_T,
					 decltype(weights[w].weight),
					 typename CONFIG_T::accum_t>::product(data[weights[w].row_index], weights[w].weight);

	    col[im] = weights[w].col_index;
        }

        typename CONFIG_T::accum_t tmp_acc[CONFIG_T::n_out];
        #pragma HLS ARRAY_PARTITION variable=tmp_acc complete
        ResetMult:
        for (int tacc = 0; tacc < CONFIG_T::n_out; tacc++) {
            #pragma HLS UNROLL
            tmp_acc[tacc] = 0;
        }

        AccumLoop1:
        for(int im = 0; im < CONFIG_T::compressed_block_factor; im++) {
	    #pragma HLS_UNROLL
            tmp_acc[col[im]] += mult[im];
        }

        AccumLoop2:
        for (int im = 0; im < CONFIG_T::n_out; im++) {
            #pragma HLS UNROLL
            acc[im] += tmp_acc[im];
        }
    }

    // Cast to "res_t" type
    ResultLoop:
    for(unsigned i = 0; i < CONFIG_T::n_out; i++){
        #pragma HLS UNROLL
        //res[i] = (res_T) (acc[i]);
        res[i] = cast<data_T, res_T, CONFIG_T>(acc[i]);
    }
}

}

#endif
