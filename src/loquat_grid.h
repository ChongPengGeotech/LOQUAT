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

#pragma once
#ifndef __LOQUAT_GRID_H__
#define __LOQUAT_GRID_H__

#include <cuda_runtime.h>
#include <types.h>

class LoquatGrid
{
	private:

		int* part_cel;
		int* part_idx;
		int* cell_beg;
		int* cell_end;

		int* part_cel_cache;
		int* part_idx_cache;
		void* buffer;

		size_t buffer_storage_bytes;

		void SortParticleByCellIndex(Parameters* par_host);

	public:

		LoquatGrid();

		~LoquatGrid();

		const int* GetPartCel() const { return part_cel; }
		const int* GetPartIdx() const { return part_idx; }
		const int* GetCellBeg() const { return cell_beg; }
		const int* GetCellEnd() const { return cell_end; }

		void GridConfig(Parameters* par_host);

		void ResizeArrayLength(int np_new, int ncell_new);

		void NeighborSearch(Parameters* par_host, Parameters* par_device, float4* pospres, float4* idep);

		void SortArrayFloat4(int np, float4* data, float4* data_cache);

		void SortArrayTensor2d(int np, tensor2d* data, tensor2d* data_cache);

};


#endif
