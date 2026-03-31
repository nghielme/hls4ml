#ifndef NNET_IMAGE_H_
#define NNET_IMAGE_H_

#include "nnet_common.h"

namespace nnet {

// ============================================================================
// Configuration structures for image resize operations
// ============================================================================

struct resize_config {
    static const unsigned height = 10;
    static const unsigned width = 10;
    static const unsigned n_chan = 10;
    static const unsigned new_height = 10;
    static const unsigned new_width = 10;
};

// ============================================================================
// BAMBU-OPTIMIZED: Nearest-neighbor upsampling (io_parallel)
// ============================================================================
// Key optimizations for Bambu/HLS pragma experiment:
// 1. Compile-time ratio computation using constexpr
// 2. Hoisted index calculations (y2, x2) outside inner loops
// 3. Pre-computed base indices to reduce arithmetic in inner loop
// 4. Explicit unroll pragma for loops using HLS directives
// ============================================================================
template <class data_T, typename CONFIG_T>
void resize_nearest(data_T image[CONFIG_T::height * CONFIG_T::width * CONFIG_T::n_chan],
                    data_T resized[CONFIG_T::new_height * CONFIG_T::new_width * CONFIG_T::n_chan]) {
    // Compute scaling ratios at compile time for Bambu optimization
    // These are fixed-point ratios (16-bit fractional) for integer-only arithmetic
    constexpr int y_ratio = static_cast<int>((CONFIG_T::height << 16) / CONFIG_T::new_height) + 1;
    constexpr int x_ratio = static_cast<int>((CONFIG_T::width << 16) / CONFIG_T::new_width) + 1;

    // Outer loops: height and width iteration
    #pragma HLS UNROLL
    for (int i = 0; i < static_cast<int>(CONFIG_T::new_height); i++) {
        // Compute source y-coordinate once per row (hoisted from inner loop)
        const int y2 = ((i * y_ratio) >> 16);
        
        #pragma HLS UNROLL
        for (int j = 0; j < static_cast<int>(CONFIG_T::new_width); j++) {
            // Compute source x-coordinate once per pixel
            const int x2 = ((j * x_ratio) >> 16);
            
            // Pre-compute base indices to minimize arithmetic in channel loop
            const int src_base = (y2 * static_cast<int>(CONFIG_T::width) + x2) * static_cast<int>(CONFIG_T::n_chan);
            const int dst_base = (i * static_cast<int>(CONFIG_T::new_width) + j) * static_cast<int>(CONFIG_T::n_chan);
            
            // Unroll channel loop completely (typically n_chan is small: 1, 3, or 4)
            #pragma HLS UNROLL
            for (int k = 0; k < static_cast<int>(CONFIG_T::n_chan); k++) {
                resized[dst_base + k] = image[src_base + k];
            }
        }
    }
}

} // namespace nnet

#endif
