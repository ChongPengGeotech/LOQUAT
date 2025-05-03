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
#include "particle_interaction.h"

static __constant__ Parameters par;

__inline__ __device__ void ParticleInteractionCalculation_cuk(
		const int part_beg,
		const int part_end,
		const float4 pospres_i,
		const float4 velrhop_i,
		const tensor2d str_over_rho2_i,
		float4* pospres,
		float4* velrhop,
		float4* idep,
		tensor2d* str,
		float4& accad_i,
		float4& grad_u,
		float4& grad_v,
		float4& grad_w,
		const bool bound,
		int& number_of_interacting_particles)
{
	for(int j = part_beg; j < part_end; j++)
	{
		float4 pospres_j = pospres[j];

		float3 dr = make_float3(pospres_i.x - pospres_j.x, pospres_i.y - pospres_j.y, pospres_i.z - pospres_j.z);
		float r = norm3df(dr.x, dr.y, dr.z);
		float q = r * par.i_h;

		if(q < 2.0f)
		{
			number_of_interacting_particles += 1;

			float dwdr = par.kernel_gradient_normalization_par * (2.0f - q) * (2.0f - q) * (2.0f - q);
			float4 gker = make_float4(dwdr * dr.x, dwdr * dr.y, dwdr * dr.z, 0.0f);

			float4 velrhop_j = velrhop[j];
			float4 dvel = make_float4(velrhop_i.x - velrhop_j.x, velrhop_i.y - velrhop_j.y, velrhop_i.z - velrhop_j.z, 0.0f);

			// Density rate
			accad_i.w += par.m * (dvel.x * gker.x + dvel.y * gker.y + dvel.z * gker.z);

			float vol_j = par.m / velrhop_j.w;

			// Velocity gradient
			grad_u.x -= dvel.x * gker.x * vol_j;
			grad_u.z -= dvel.x * gker.z * vol_j;

			grad_w.x -= dvel.z * gker.x * vol_j;
			grad_w.z -= dvel.z * gker.z * vol_j;

			#if SIMULATION2D != 1
			grad_u.y -= dvel.x * gker.y * vol_j;

			grad_v.x -= dvel.y * gker.x * vol_j;
			grad_v.y -= dvel.y * gker.y * vol_j;
			grad_v.z -= dvel.y * gker.z * vol_j;

			grad_w.y -= dvel.z * gker.y * vol_j;
			#endif

			
			if(par.viscosity_type == 1)
			{
				float artv = 0.0f;
				float visc = dvel.x * dr.x + dvel.y * dr.y + dvel.z * dr.z;
				if(visc < 0.0f)
				{
					float phi = par.h * visc / (r * r + par.eta);
					artv = par.artificial_viscosity_alpha * par.cs * phi / (0.5f * (velrhop_i.w + velrhop_j.w));
				}

				accad_i.x += par.m * artv * gker.x;
				#if SIMULATION2D != 1
				accad_i.y += par.m * artv * gker.y;
				#endif
				accad_i.z += par.m * artv * gker.z;
			}

			if(par.artificial_pressure == 1)
			{
				float artp = -(str_over_rho2_i.xx + str_over_rho2_i.yy + str_over_rho2_i.zz);
				if(artp < 0.0f) artp *= -par.artificial_pressure_coefficient;
				else			artp = 0.0f;

				float rj = -(str[j].xx + str[j].yy + str[j].zz);
				if(rj < 0.0f) rj *= - par.artificial_pressure_coefficient / (velrhop_j.w * velrhop_j.w);
				else		  rj = 0.0f;

				artp += rj;
				if(artp > 0.0f)
				{
					float ker = par.kernel_normalization_par * (2.0f - q) * (2.0f - q) * (2.0f - q) * (2.0f - q) * (q + 0.5f);
					artp *= powf(ker / par.kernel_dr, par.kernel_zero / par.kernel_dr);

					accad_i.x -= par.m * artp * gker.x;
					#if SIMULATION2D != 1
					accad_i.y -= par.m * artp * gker.y;
					#endif
					accad_i.z -= par.m * artp * gker.z;
				}
			}

			// Acceleration from stress
			tensor2d str_term = str_over_rho2_i + str[j] / (velrhop_j.w * velrhop_j.w);

			accad_i.x += par.m * (str_term.xx * gker.x + str_term.xy * gker.y + str_term.xz * gker.z);
			#if SIMULATION2D != 1
			accad_i.y += par.m * (str_term.xy * gker.x + str_term.yy * gker.y + str_term.yz * gker.z);
			#endif
			accad_i.z += par.m * (str_term.xz * gker.x + str_term.yz * gker.y + str_term.zz * gker.z);
		}
	}
}


__global__ void ParticleInteraction_cuk(
	float4* pospres,
	float4* velrhop,
	float4* idep,
	tensor2d* str,
	tensor2d* stn,
	float4* accad,
	float4* spin,
	tensor2d* acestn,
	const int* cell_beg,
	const int* cell_end)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < par.npm)
	{
		int i = par.npb + index;
		float4 pospres_i = pospres[i];
		float4 velrhop_i = velrhop[i];

		tensor2d str_over_rho2_i = str[i] / (velrhop_i.w * velrhop_i.w);

		float4 accad_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		float4 grad_u = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		float4 grad_v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		float4 grad_w = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		int3 celli = make_int3(floorf(0.5f * pospres_i.x * par.i_h),
							   floorf(0.5f * pospres_i.y * par.i_h),
							   floorf(0.5f * pospres_i.z * par.i_h));

		int number_of_interacting_particles = 0;

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
					int part_beg = cell_beg[cell_idx];
					if(part_beg != -1)
					{
						ParticleInteractionCalculation_cuk(part_beg, cell_end[cell_idx], pospres_i, velrhop_i, str_over_rho2_i, pospres, velrhop, idep, str, accad_i, grad_u, grad_v, grad_w, true, number_of_interacting_particles); 
					}

					cell_idx += par.ncell_one_layer;
					part_beg = cell_beg[cell_idx];
					if(part_beg != -1)
					{
						ParticleInteractionCalculation_cuk(part_beg, cell_end[cell_idx], pospres_i, velrhop_i, str_over_rho2_i, pospres, velrhop, idep, str, accad_i, grad_u, grad_v, grad_w, false, number_of_interacting_particles);
					}
				}
			}
		}

		accad_i.x += par.acc_x;
		accad_i.y += par.acc_y;
		accad_i.z += par.acc_z;

		#if COMPUTE_DISPLACEMENT
		// Damping
		if(par.time < 0.5f)
		{
			float cd = 0.02f * sqrtf(100.0e6f / (par.rho0 * par.h * par.h));
			accad_i.x -= cd * velrhop_i.x;
			accad_i.y -= cd * velrhop_i.y;
			accad_i.z -= cd * velrhop_i.z;
		}
		#endif

		accad[i] = accad_i;

		acestn[i] = make_tensor2d(grad_u.x, grad_v.y, grad_w.z, 0.5f * (grad_u.y + grad_v.x),
																0.5f * (grad_u.z + grad_w.x),
																0.5f * (grad_v.z + grad_w.y));

		spin[i] = make_float4(0.5f * (grad_v.x - grad_u.y),
							  0.5f * (grad_w.x - grad_u.z),
							  0.5f * (grad_w.y - grad_v.z),
							  0.0f);
	}
}


void ParticleInteraction(
	Parameters* par_host,
	Parameters* par_device,
	float4* pospres,
	float4* velrhop,
	float4* idep,
	tensor2d* str,
	tensor2d* stn,
	float4* accad,
	float4* spin,
	tensor2d* acestn,
	const int* cell_beg,
	const int* cell_end)
{
	if(par_host->npm == 0) return;

	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->npm + block.x - 1) / block.x, 1);

	ParticleInteraction_cuk <<<cuda_grid_size, block, 0>>> (pospres, velrhop, idep, str, stn, accad, spin, acestn, cell_beg, cell_end);
	getLastCudaError("ParticleInteraction_cuk failed...     \n");
}
