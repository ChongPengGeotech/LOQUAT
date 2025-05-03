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

#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cub/cub.cuh>
#include "loquat_grid.h"

static __constant__ Parameters par;


LoquatGrid::LoquatGrid()
{
	part_cel = NULL;
    part_idx = NULL;
    cell_beg = NULL;
    cell_end = NULL;

	part_cel_cache = NULL;
	part_idx_cache = NULL;
	buffer = NULL;

	buffer_storage_bytes = 0;
}


LoquatGrid::~LoquatGrid()
{
    if(part_cel != NULL) cudaFree(part_cel);
    if(part_idx != NULL) cudaFree(part_idx);
    if(cell_beg != NULL) cudaFree(cell_beg);
    if(cell_end != NULL) cudaFree(cell_end);

	if(part_cel_cache != NULL) cudaFree(part_cel_cache);
	if(part_idx_cache != NULL) cudaFree(part_idx_cache);
	if(buffer != NULL) cudaFree(buffer);

	buffer_storage_bytes = 0;
}


void LoquatGrid::GridConfig(Parameters* par_host)
{
	checkCudaErrors(cudaMalloc((void **)&part_cel, par_host->np * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&part_idx, par_host->np * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&cell_beg, par_host->ncell * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&cell_end, par_host->ncell * sizeof(int)));

	checkCudaErrors(cudaMalloc((void **)&part_cel_cache, par_host->np * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&part_idx_cache, par_host->np * sizeof(int)));
}


void LoquatGrid::ResizeArrayLength(int np_new, int ncell_new)
{
	if(part_cel != NULL) cudaFree(part_cel);
    if(part_idx != NULL) cudaFree(part_idx);
    if(cell_beg != NULL) cudaFree(cell_beg);
	if(cell_end != NULL) cudaFree(cell_end);
	
	checkCudaErrors(cudaMalloc((void **)&part_cel, np_new * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&part_idx, np_new * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&cell_beg, ncell_new * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&cell_end, ncell_new * sizeof(int)));
}



__global__ void GetParticleCellIndex_cuk(
		int* part_cel,
		int* part_idx,
		float4* pospres,
		float4* idep)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < par.np)
	{
		float4 pospres_i = pospres[i];

		if(pospres_i.x < 0.0f || pospres_i.x > par.domain_size_x ||
		   pospres_i.y < 0.0f || pospres_i.y > par.domain_size_y ||
		   pospres_i.z < 0.0f || pospres_i.z > par.domain_size_z ||
		   idep[i].x == -1.0f)
		{
			part_cel[i] = par.ncell_delete;
			part_idx[i] = i;

			idep[i].x = -1.0f; // Particles will not be used in computation

			return;
		}

		int3 celli;
		celli.x = floorf(0.5f * pospres_i.x * par.i_h);
		celli.y = floorf(0.5f * pospres_i.y * par.i_h);
		celli.z = floorf(0.5f * pospres_i.z * par.i_h);
		int cell_idx = celli.x * par.cell_num_y * par.cell_num_z + celli.y * par.cell_num_z + celli.z;
		
		if(idep[i].x < 100.0f) // Boundary
		{
			part_cel[i] = cell_idx;
		}
		else if(idep[i].x >= 100.0f) // Material
		{
			part_cel[i] = cell_idx + par.ncell_one_layer;
		}

		part_idx[i] = i;
	}
}


void LoquatGrid::SortParticleByCellIndex(Parameters* par_host)
{
	size_t temp_storage_bytes = 0;

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, part_cel, part_cel_cache, part_idx, part_idx_cache, par_host->np));

	if(temp_storage_bytes > buffer_storage_bytes)
	{
		buffer_storage_bytes = temp_storage_bytes;
		if(buffer != NULL) cudaFree(buffer);
		checkCudaErrors(cudaMalloc(&buffer, buffer_storage_bytes));
	}

	checkCudaErrors(cub::DeviceRadixSort::SortPairs(buffer, temp_storage_bytes, part_cel, part_cel_cache, part_idx, part_idx_cache, par_host->np));

	checkCudaErrors(cudaMemcpyAsync(part_cel, part_cel_cache, par_host->np * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(part_idx, part_idx_cache, par_host->np * sizeof(int), cudaMemcpyDeviceToDevice));
}


__global__ void FindBegEndInCell_cuk(int* cell_beg, int* cell_end, int* part_cel)
{
	extern __shared__ int shared_cells[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int celli;

	if(i < par.np)
	{
		celli = part_cel[i];
		shared_cells[threadIdx.x + 1] = celli;

		// First thread in this block
		if(i > 0 && threadIdx.x == 0) shared_cells[0] = part_cel[i - 1];
	}

	__syncthreads();

	if(i < par.np)
	{
		if(i == 0 || celli != shared_cells[threadIdx.x])
		{
			cell_beg[celli] = i;
			if(i > 0) cell_end[shared_cells[threadIdx.x]] = i;
		}

		if(i == par.np - 1) cell_end[celli] = i + 1;
	}

}


void LoquatGrid::NeighborSearch(Parameters* par_host, Parameters* par_device, float4* pospres, float4* idep)
{
	checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((par_host->np + block.x - 1) / block.x, 1);

	GetParticleCellIndex_cuk <<<cuda_grid_size, block, 0>>> (part_cel, part_idx, pospres, idep);
	getLastCudaError("GetParticleCellIndex_cuk failed...     \n");

	SortParticleByCellIndex(par_host);

	checkCudaErrors(cudaMemsetAsync(cell_beg, -1, par_host->ncell * sizeof(int)));
	checkCudaErrors(cudaMemsetAsync(cell_end, -1, par_host->ncell * sizeof(int)));

	FindBegEndInCell_cuk <<<cuda_grid_size, block, (block.x + 1) * sizeof(int)>>> (cell_beg, cell_end, part_cel);
	getLastCudaError("FindBegEndInCell_cuk failed...     \n");

	int beg;
	checkCudaErrors(cudaMemcpy(&beg, cell_beg + par_host->ncell_delete, sizeof(int), cudaMemcpyDeviceToHost));
	if(beg != -1)
	{
		int end;
		checkCudaErrors(cudaMemcpy(&end, cell_end + par_host->ncell_delete, sizeof(int), cudaMemcpyDeviceToHost));
		par_host->np_delete = end - beg;
		par_host->npm = par_host->np - par_host->npb - par_host->np_delete;
	}
}


static __global__ void SortArrayFloat4_cuk(int np, float4* data, float4* data_cache, int* part_idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < np)
	{
		data[i] = data_cache[part_idx[i]];
	}
}


void LoquatGrid::SortArrayFloat4(int np, float4* data, float4* data_cache)
{
	checkCudaErrors(cudaMemcpyAsync(data_cache, data, np * sizeof(float4), cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((np + block.x - 1) / block.x, 1);

	SortArrayFloat4_cuk <<<cuda_grid_size, block, 0>>> (np, data, data_cache, part_idx);
	getLastCudaError("SortArrayFloat4_cuk failed...     \n");
}


static __global__ void SortArrayTensor2d_cuk(int np, tensor2d* data, tensor2d* data_cache, int* part_idx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < np)
	{
		data[i] = data_cache[part_idx[i]];
	}
}

void LoquatGrid::SortArrayTensor2d(int np, tensor2d* data, tensor2d* data_cache)
{
	checkCudaErrors(cudaMemcpyAsync(data_cache, data, np * sizeof(tensor2d), cudaMemcpyDeviceToDevice));

	dim3 block(128, 1);
	dim3 cuda_grid_size = dim3((np + block.x - 1) / block.x, 1);

	SortArrayTensor2d_cuk <<<cuda_grid_size, block, 0>>> (np, data, data_cache, part_idx);
	getLastCudaError("SortArrayTensor2d_cuk failed...     \n");
}
