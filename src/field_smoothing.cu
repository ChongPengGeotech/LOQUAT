/*********************************************************************************
 * Copyright (c) 2018, Chong Peng. All rights reserved.

 * <LOQUAT> Three-dimensional GPU-accelerated SPH solver for geotechnical modeling

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    1. Redistributions of source code must retain the copyright
 *       notice, this list of conditions and the following disclaimer.
 *    2. The origin of this software must not be misrepresented; you must
 *       not claim that you wrote the original software.
 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
 * EVENT SHALL THE HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE.
 *********************************************************************************/

#include <helper_cuda.h>
#include "field_smoothing.h"

static __constant__ Parameters par;


__inline__ __device__ void StressRegularizationShepardCalculation_cuk(
	const int part_beg,
	const int part_end,
	const float4 pospres_i,
	float4* pospres,
	float4* velrhop,
	tensor2d* str,
	tensor2d& str_shepard,
	float& sumker)
{
	for(int j = part_beg; j < part_end; j++)
	{
		float4 pospres_j = pospres[j];

		float3 dr = make_float3(pospres_i.x - pospres_j.x, pospres_i.y - pospres_j.y, pospres_i.z - pospres_j.z);
		float r = norm3df(dr.x, dr.y, dr.z);
		float q = r * par.i_h;

		if(q < 2.0f)
		{
			float ker = par.kernel_normalization_par * (2.0f - q) * (2.0f - q) * (2.0f - q) * (2.0f - q) * (q + 0.5f);

			float vol = par.m / velrhop[j].w;

			str_shepard.xx += str[j].xx * ker * vol;
			str_shepard.yy += str[j].yy * ker * vol;
			str_shepard.zz += str[j].zz * ker * vol;
			str_shepard.xy += str[j].xy * ker * vol;
			str_shepard.xz += str[j].xz * ker * vol;
			str_shepard.yz += str[j].yz * ker * vol;

			sumker += ker * vol;
		}
	}
}



__global__ void StressRegularizationShepard_cuk(
	float4* pospres,
	float4* velrhop,
	tensor2d* str,
	tensor2d* tensor2d_cache,
	const int* cell_beg,
	const int* cell_end)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < par.npm)
	{
		int i = index + par.npb;
		float4 pospres_i = pospres[i];

		int3 celli = make_int3(floorf(0.5f * pospres_i.x * par.i_h),
							   floorf(0.5f * pospres_i.y * par.i_h),
							   floorf(0.5f * pospres_i.z * par.i_h));
		
		tensor2d str_shepard = make_tensor2d(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		float sumker = 0.0f;

		for(int cz = -1; cz <= 1; cz++)
		{
			for(int cy = -1; cy <= 1; cy++)
			{
				for(int cx = -1; cx <= 1; cx++)
				{
					int3 cellj = make_int3(celli.x + cx, celli.y + cy, celli.z + cz);
					
					if(cellj.x < 0 || cellj.x > par.cell_num_x - 1 ||
					   cellj.y < 0 || cellj.y > par.cell_num_y - 1 ||
					   cellj.z < 0 || cellj.z > par.cell_num_z - 1)
					{
						continue;
					}

					int cell_idx = cellj.x * par.cell_num_y * par.cell_num_z + cellj.y * par.cell_num_z + cellj.z;
					cell_idx += par.ncell_one_layer;
					int part_beg = cell_beg[cell_idx];
					if(part_beg != -1)
					{
						StressRegularizationShepardCalculation_cuk(part_beg, cell_end[cell_idx], pospres_i, pospres, velrhop, str, str_shepard, sumker);
					}
				}
			}
		}

		str_shepard.xx /= sumker;
		str_shepard.yy /= sumker;
		str_shepard.zz /= sumker;
		str_shepard.xy /= sumker;
		str_shepard.xz /= sumker;
		str_shepard.yz /= sumker;

		tensor2d_cache[i] = str_shepard;
	}
}


void StressRegularization(
	Parameters* par_host,
	Parameters* par_device,
	float4* pospres,
	float4* velrhop,
	tensor2d* str,
	tensor2d* tensor2d_cache,
	const int* cell_beg,
	const int* cell_end)
{
	if(par_host->npm == 0) return;
	
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->npm + block.x - 1) / block.x, 1);

	StressRegularizationShepard_cuk <<<cuda_grid_size, block, 0, 0>>> (pospres, velrhop, str, tensor2d_cache, cell_beg, cell_end);
	getLastCudaError("StressRegularizationShepard_cuk failed.....     \n");

	checkCudaErrors(cudaMemcpy(str + par_host->npb, tensor2d_cache + par_host->npb, par_host->npm * sizeof(tensor2d), cudaMemcpyDeviceToDevice));
}