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

#pragma once

#include <iostream>
#include <vector>
#include "auxiliary.h"
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	// T: type of the data, chunk: pointer to the chunk, count: number of elements, alignment: alignment of the data
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment) 
	{
		// align the chunk to the alignment by adding alignment - 1 and then ANDing with the bitwise NOT of alignment - 1
		// used to ensure that the data is aligned to the alignment and doesn't overlap with other data
		// For example, if chunk is at address 10 and alignment is 16, then chunk + alignment - 1 is 25 and ~(alignment - 1) is -16 (in 2's complement), so the result is 16
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1); 
		// set the pointer to the aligned chunk
		ptr = reinterpret_cast<T*>(offset); 
		// move the chunk to the next position
		chunk = reinterpret_cast<char*>(ptr + count); 
	}

	struct GeometryState // for forward pass | used to store intermediate values for forward pass. Particularly, it stores the depths, scanning space, clamped flags, internal radii, means2D, cov3D, conic_opacity, rgb, point_offsets, and tiles_touched for each gaussian
	{
		size_t scan_size; // size of the scanning space
		float* depths; // depth of each gaussian for sorting
		char* scanning_space; // scanning space. For example, if there are 100 gaussians, then the scanning space will be of size 100 * sizeof(float) to store the depths of the gaussians
		bool* clamped; // clamped flags for each gaussian to clamp the gaussians to the image boundaries
		int* internal_radii; // internal radii for each gaussian to determine the size of the gaussians
		float2* means2D; // 2D means
		float* cov3D; // covariance matrices
		float4* conic_opacity; // conic opacities
		float* rgb; // rgb colors_precomp
		uint32_t* point_offsets; // point offsets used for binning. Binning means to group gaussians that are visible in the current frame for each tile
		uint32_t* tiles_touched; // tiles touched by each gaussian to determine the order in which the gaussians are rendered

		static GeometryState fromChunk(char*& chunk, size_t P); // create geometry state from chunk
	};

	struct ImageState // for forward pass | used to store intermediate values for forward pass. Particularly, it stores the bucket_count and bucket_offsets for each tile_count
	{
		uint32_t *bucket_count; // number of gaussians in each tile
		uint32_t *bucket_offsets; // offsets of each tile in the bucket_count_scan_size for each tile
		size_t bucket_count_scan_size; // size of the scanning space
		char * bucket_count_scanning_space; // scanning space for bucket_count
		float* pixel_colors; // pixel colors
		uint32_t* max_contrib; // max contribution of each pixel to limit the number of gaussians in each tile

		size_t scan_size; // size of the scanning space
		uint2* ranges; // ranges of each tile in the bucket_count
		uint32_t* n_contrib; // number of contributions of each pixel
		float* accum_alpha; // accumulated alpha of each pixel
		char* contrib_scan; // scanning space

		static ImageState fromChunk(char*& chunk, size_t N); // create image state from chunk
	};

	struct BinningState // for backward pass | used to store intermediate values for backward pass. Particularly, it stores the sorted list of gaussians for each tile_count
	{
		size_t scan_size; // size of the scanning space 
		size_t sorting_size; // size of the sorting space
		uint64_t* point_list_keys_unsorted; // point list keys unsorted
		uint64_t* point_list_keys; // point list keys
		uint32_t* point_list_unsorted; // point list unsorted
		uint32_t* point_list; // point list
		int* scan_src; // source of the scan
		int* scan_dst; // destination of the scan
		char* scan_space; // scanning space
		char* list_sorting_space; // sorting space

		static BinningState fromChunk(char*& chunk, size_t P); // create binning state from chunk
	};

	struct SampleState // for backward pass | used to store intermediate values for backward pass. Particularly, it stores the transmittance and accumulated alpha for each pixel
	{
		uint32_t *bucket_to_tile; // bucket to tile mapping
		float *T; // transmittance -> used for backward pass to calculate the gradient of the color. It is calculated as exp(-alpha)
		float *ar; // accumulated alpha -> used for backward pass to calculate the gradient of the color. It is calculated as 1 - T
		static SampleState fromChunk(char*& chunk, size_t C); // create sample state from chunk
	};

	template<typename T> 
	size_t required(size_t P) // get the required size for the chunk	
	{
		char* size = nullptr;
		T::fromChunk(size, P); // get the size of the chunk
		return ((size_t)size) + 128; // return the size of the chunk + 128 bytes for alignment
	}
};