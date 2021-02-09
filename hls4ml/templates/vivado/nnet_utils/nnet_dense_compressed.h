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
#include "nnet_types.h"
#include "nnet_dense.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

template<typename CONFIG_T>
void fill_mult(typename CONFIG_T::index_t index,
        typename CONFIG_T::accum_t mult[CONFIG_T::n_out],
        typename CONFIG_T::accum_t weight) {
    for(unsigned  k = 0; k < CONFIG_T::n_out; k++) {
        #pragma HLS UNROLL
        if (k == index) mult[k] += weight;
    }
}

// This version only works when reuse_factor < n_in, and it doesn't merge rows for now

template<class data_T, class res_T, typename CONFIG_T>
void dense_compressed(
        data_T    data[CONFIG_T::n_in],
        res_T     res[CONFIG_T::n_out],
        typename CONFIG_T::weight_t  weights[CONFIG_T::n_weights],
        typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    // currently only implementing reuse-factor < n_in
    static_assert(CONFIG_T::reuse_factor < CONFIG_T::n_in, "Currently only implementing reuse-factor < n_in");

    #pragma HLS dataflow

    constexpr unsigned block_size = CONFIG_T::n_in / CONFIG_T::reuse_factor;

    typename CONFIG_T::accum_t acc [CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc    complete
    #pragma HLS ARRAY_PARTITION variable=biases complete
    //#pragma HLS ARRAY_RESHAPE variable=weaghts block factor=CONFIG_T::n_in
    //if (CONFIG_T::store_weights_in_bram){
    //#pragma HLS RESOURCE variable=weights core=ROM_1P_BRAM
    #pragma HLS data_pack variable=weights struct_level
    //}


 InitAccum:
    for(unsigned i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS UNROLL
        acc[i] = (typename CONFIG_T::accum_t) (biases[i]);
    }

    typename CONFIG_T::accum_t prod[CONFIG_T::n_in][CONFIG_T::max_columns];
    typename CONFIG_T::index_t index[CONFIG_T::n_in][CONFIG_T::max_columns];
    typename CONFIG_T::accum_t mult[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=mult complete

 ResetMult:
    for(unsigned imult = 0; imult < CONFIG_T::n_out; imult++) {
        #pragma HLS UNROLL
	mult[imult] = 0;
    }

 ReuseLoop:
    for(unsigned row = 0; row < CONFIG_T::n_in; row++) {
        #pragma HLS UNROLL factor=block_size
	data_T  data_cache = data[row];
	for (unsigned i = 0; i < CONFIG_T::max_columns; i++) {
            #pragma HLS UNROLL
	    auto weight = weights[row * CONFIG_T::max_columns + i];
	    index[row][i] = weight.col_index;
	    auto weight_cache = weight.weight;
	    prod[row][i] = CONFIG_T::template product<data_T,
						      decltype(weight_cache),
						      typename CONFIG_T::accum_t>::product(data_cache, weight_cache);
	}
    }

 MergeColumns:
    for (unsigned row = 0; row < CONFIG_T::n_in; row++) {
	for (unsigned i = 0; i < CONFIG_T::max_columns; i++) {
	    # pragma UNROLL
	    fill_mult<CONFIG_T>(index[row][i], mult, prod[row][i]);
	}
    }

 Accumulate:
    for (int im = 0; im < CONFIG_T::n_out; im++) {
	# pragma UNROLL
	acc[im] += mult[im];
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
