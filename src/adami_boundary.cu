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
#include "adami_boundary.h"

static __constant__ Parameters par;


static __inline__ __device__ void AdamiBoundaryCalculation_cuk(
	int part_beg,
	int part_end,
	float4 pospres_i,
	float4* pospres,
	float4* velrhop,
	tensor2d* str,
	tensor2d& results_stress,
	float4& results_pos_sumker)
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

			results_stress.xx += str[j].xx * ker * vol;
			results_stress.yy += str[j].yy * ker * vol;
			results_stress.zz += str[j].zz * ker * vol;
			results_stress.xy += str[j].xy * ker * vol;
			results_stress.xz += str[j].xz * ker * vol;
			results_stress.yz += str[j].yz * ker * vol;

			results_pos_sumker.x += dr.x * ker * vol;
			results_pos_sumker.y += dr.y * ker * vol;
			results_pos_sumker.z += dr.z * ker * vol;
			results_pos_sumker.w += ker * vol;
		}
	}
}


__global__ void AdamiBoundary_cuk(
	float4* pospres,
	float4* velrhop,
	tensor2d* str,
	const int* cell_beg,
	const int* cell_end)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < par.npb)
	{
		float4 pospres_i = pospres[i];
		
		tensor2d results_stress = make_tensor2d(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		float4 results_pos_sumker = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		int3 celli = make_int3(floorf(0.5f * pospres_i.x * par.i_h),
							   floorf(0.5f * pospres_i.y * par.i_h),
							   floorf(0.5f * pospres_i.z * par.i_h));
		
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
						AdamiBoundaryCalculation_cuk(part_beg, cell_end[cell_idx], pospres_i, pospres, velrhop, str, results_stress, results_pos_sumker);
					}

				}
			}
		}

		if(results_pos_sumker.w > 0.0f)
		{
			str[i].xx = (results_stress.xx - par.rho0 * par.acc_x * results_pos_sumker.x) / results_pos_sumker.w;
			str[i].yy = (results_stress.yy - par.rho0 * par.acc_y * results_pos_sumker.y) / results_pos_sumker.w;
			str[i].zz = (results_stress.zz - par.rho0 * par.acc_z * results_pos_sumker.z) / results_pos_sumker.w;

			str[i].xy = results_stress.xy / results_pos_sumker.w;
			str[i].xz = results_stress.xz / results_pos_sumker.w;
			str[i].yz = results_stress.yz / results_pos_sumker.w;
		}
	}
}


void AdamiBoundary(
	Parameters* par_host,
	Parameters* par_device,
	float4* pospres,
	float4* velrhop,
	tensor2d* str,
	const int* cell_beg,
	const int* cell_end)
{
	if(par_host->npb == 0) return;
	
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->npb + block.x - 1) / block.x, 1);

	AdamiBoundary_cuk <<<cuda_grid_size, block, 0, 0>>> (pospres, velrhop, str, cell_beg, cell_end);
	getLastCudaError("AdamiBoundary_cuk failed.....     \n");
}