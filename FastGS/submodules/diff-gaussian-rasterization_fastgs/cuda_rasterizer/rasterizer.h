/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <functional>
#include <vector>

// --- DICTIONARY --- //
// FORWARD PASS
// scale modifier -> used to scale the gaussians
// viewmatrix -> used to transform the gaussians from world space to view space -> View space is the space of the camera 
// projmatrix -> used to transform the gaussians from view space to clip space -> Clip space is the space of the screen
// metric_map -> used to store the metrics for each tile_count -> Metrics are used to determine the order in which the gaussians are rendered
// tan_fovx, tan_fovy -> used to transform the gaussians from clip space to screen space -> They are calculated from the camera intrinsics and image size
// multipler -> used to scale the gaussians
// prefiltered -> used to filter the gaussians to remove the gaussians that are not visible in the current frame
// debug -> used to debug the gaussians to see the gaussians that are visible in the current frame
// metricCount -> used to count the metrics for each tile_count
// out_color -> output color (per pixel) -> It is the final image (color + depth + alpha)
// radii -> radii of the gaussians (per pixel)

namespace CudaRasterizer {
class Rasterizer {
public:
  static void markVisible(int P,             // number of gaussians
                          float *means3D,    // 3D positions of gaussians
                          float *viewmatrix, // view matrix
                          float *projmatrix, // projection matrix
                          bool *present);    // present flags

  static std::tuple<int, int> // return (tile_count, pixel_count)
  forward(
      std::function<char *(size_t)>geometryBuffer, // geometry buffer (per gaussian) -> store gaussians that are visible in the current frame for each tile
      std::function<char *(size_t)>binningBuffer, // binning buffer (per tile) -> store gaussians that are visible in the current frame for each tile
      std::function<char *(size_t)>imageBuffer, // image buffer (per pixel) -> store the final image (color + depth + alpha)
      std::function<char *(size_t)>sampleBuffer, // sample buffer (per pixel) -> store the samples (color + depth + alpha)
      const int P, int D, int M, // P: number of gaussians, D: number of SH coefficients, M: number of tiles
      const float *background, const int width, int height, // background color, image width, image height
      const float *means3D, const float *dc, const float *shs, // 3D positions of gaussians, mean color, SH coefficients
      const float *colors_precomp, const float *opacities, // precomputed colors, opacities
      const float *scales, const float scale_modifier, // scales, scale modifier
      const float *rotations, const float *cov3D_precomp, // rotations, covariance matrices
      const int *metric_map, const float *viewmatrix, // metric map, view matrix
      const float *projmatrix, const float *cam_pos,  // projection matrix, camera position
      const float mult, // multiplier
      const float tan_fovx, float tan_fovy, // tan_fovx, tan_fovy
      const bool prefiltered, // prefiltered
      float *out_color, int *radii = nullptr, // output color, radii
      bool debug = false, // debug
      bool get_flag = false, int *metricCount = nullptr); // get flag, metric count

  static void backward(
      const int P, int D, int M, int R, int B, const float *background, // P: number of gaussians, D: number of SH coefficients, M: number of tiles, R: number of tiles in x direction, B: number of tiles in y direction
      const int width, int height, const float *means3D, const float *dc, // width, height, 3D positions of gaussians, mean color
      const float *shs, const float *colors_precomp, const float *scales, // SH coefficients, precomputed colors, scales
      const float scale_modifier, const float *rotations, const float *cov3D_precomp, // scale modifier, rotations, covariance matrices
      const float *viewmatrix, // view matrix
      const float *projmatrix, const float *campos, const float tan_fovx, // projection matrix, camera position, tan_fovx
      float tan_fovy, const int *radii, char *geom_buffer, char *binning_buffer, // tan_fovy, radii, geometry buffer, binning buffer
      char *image_buffer, char *sample_buffer, const float *dL_dpix, // image buffer, sample buffer, gradient of pixel color
      float *dL_dmean2D, float *dL_dconic, float *dL_dopacity, float *dL_dcolor, // gradient of 2D mean, conics, opacity, color
      float *dL_dmean3D, float *dL_dcov3D, float *dL_ddc, float *dL_dsh, // gradient of 3D mean, covariance, mean color, SH coefficients
      float *dL_dscale, float *dL_drot, bool debug); // gradient of scale, rotation, debug
};
}; // namespace CudaRasterizer

#endif