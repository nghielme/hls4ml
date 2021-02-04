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

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_rows, CONFIG_T::reuse_factor);
    constexpr unsigned reduced_in = CONFIG_T::n_in - CONFIG_T::n_zero_rows;

    typename CONFIG_T::accum_t acc [CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc    complete
    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_PARTITION variable=weights block factor=CONFIG_T::n_rows
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

        typename CONFIG_T::accum_t mult[CONFIG_T::n_out];
        #pragma HLS ARRAY_PARTITION variable=mult complete

	ResetMult:
        for(unsigned imult = 0; imult < CONFIG_T::n_out; imult++) {
            #pragma HLS UNROLL
            mult[imult] = 0;
        }

        CompressedMultLoop:
        for(unsigned im = 0; im < multiplier_limit; im++) {
            #pragma HLS UNROLL
            unsigned w = ir * CONFIG_T::reuse_factor + im;
	    if (w < reduced_in) {
		auto row = CONFIG_T::zero_remapping[w];
		auto col = weights[w].col_index;
		auto weight_cache = weights[w].weight;
		auto data_cache = data[row];
		auto prod =
		    CONFIG_T::template product<data_T,
					       decltype(weights[w].weight),
					       typename CONFIG_T::accum_t>::product(data_cache, weight_cache);
		fill_mult<CONFIG_T>(col, mult, prod);
	    } else if (w < CONFIG_T::n_rows) {
		auto row = CONFIG_T::extra_rows[w - CONFIG_T::n_rows];
		auto col = weights[w].col_index;
		auto weight_cache = weights[w].weight;
		auto data_cache = data[row];
		auto prod =
		    CONFIG_T::template product<data_T,
					       decltype(weights[w].weight),
					       typename CONFIG_T::accum_t>::product(data_cache, weight_cache);
		fill_mult<CONFIG_T>(col, mult, prod);
	    }		
        }

        AccumLoop2:
        for (int im = 0; im < CONFIG_T::n_out; im++) {
            #pragma HLS UNROLL
            acc[im] += mult[im];
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
