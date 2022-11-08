#ifndef NNET_PADDING_STREAM_H_
#define NNET_PADDING_STREAM_H_

namespace nnet {

template<class res_T, typename CONFIG_T>
inline void fill_zero(stream<res_T> &res) {
    hls_register res_T res_part;
	#pragma unroll
    for (int i = 0; i < CONFIG_T::n_chan; i++) {
	    res_part[i] = 0;
    }
    res.write(res_part);
}

template<class data_T, class res_T, typename CONFIG_T>
inline void fill_data(stream<data_T> &data, stream<res_T> &res) {
    hls_register data_T data_part = data.read();
    hls_register res_T res_part;
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_chan; i++) {
        res_part[i] = data_part[i];
    }
    res.write(res_part);
}

// template<class data_T, class res_T, typename CONFIG_T>
// void zeropad1d_cl(stream<data_T> &data, stream<res_T> &res) {
//     PadLeft:
//     for (int i = 0; i < CONFIG_T::pad_left; i++) {
//         fill_zero<res_T, CONFIG_T>(res);
//     }

//     CopyMain:
//     for (int i = 0; i < CONFIG_T::in_width; i++) {
//         fill_data<data_T, res_T, CONFIG_T>(data, res);
//     }

//     PadRight:
//     for (int i = 0; i < CONFIG_T::pad_right; i++) {
//         fill_zero<res_T, CONFIG_T>(res);
//     }
// }

// This is an unoptimal version that doesn't run into the error above

template<class data_T, class res_T, typename CONFIG_T>
void zeropad1d_cl(stream<data_T> &data, stream<res_T> &res) {

    res_T res_array[CONFIG_T::out_width];

    ZeroOutputArray:
    for (int i = 0; i < CONFIG_T::out_width; i++) {
        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            res_array[i][j] = 0;
        }
    }

    CopyMain:
    for (int i = 0; i < CONFIG_T::in_width; i++) {
        auto dataval = data.read();
        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            res_array[i+CONFIG_T::pad_left][j] = dataval[j];
        }
    }

    StreamOut:
    for (int i = 0; i < CONFIG_T::out_width; i++) {
        res.write(res_array[i]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cl(stream<data_T> &data, stream<res_T> &res) {
    #pragma loop_coalesce 2
    PadTop:
    for (int i = 0; i < CONFIG_T::pad_top; i++) {
        PadTopWidth:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero<res_T, CONFIG_T>(res);
        }
    }

    #pragma loop_coalesce 2
    PadMain:
    for (int i = 0; i < CONFIG_T::in_height; i++) {

        PadLeft:
        for (int j = 0; j < CONFIG_T::pad_left; j++) {
            fill_zero<res_T, CONFIG_T>(res);
        }

        CopyMain:
        for (int j = 0; j < CONFIG_T::in_width; j++) {
            fill_data<data_T, res_T, CONFIG_T>(data, res);
        }

        PadRight:
        for (int j = 0; j < CONFIG_T::pad_right; j++) {
            fill_zero<res_T, CONFIG_T>(res);
        }
    }

    PadBottom:
    for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
        PadBottomWidth:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero<res_T, CONFIG_T>(res);
        }
    }
}

}

#endif