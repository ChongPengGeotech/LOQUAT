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
#ifndef __LOQUAT_H__
#define __LOQUAT_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <boost/variant.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <types.h>
#include "tinyxml2.h"
#include "loquat_io.h"
#include "loquat_grid.h"


class LoquatSph
{
	private:
	
		std::string project_name;
		int selected_device;
		
		Parameters* par_host;
		Parameters* par_device;
		
		Timer* timer;
		
		LoquatGrid* grid;
		
		LoquatIo* io;
		
		// Primary arrays
		float4* pospres;		// Position + pression
		float4* velrhop;		// Velocity + density
		float4* idep;			// Object id + equivalent plastic strain
		tensor2d* str;			// Shear stress tensor
		tensor2d* stn;			// Total strain tensor
		
		// Arrays for second order integration
		float4* pospres_ini;
		float4* velrhop_ini;
		float4* idep_ini;
		tensor2d* str_ini;
		tensor2d* stn_ini;
		
		// Arrays for computation, no need for sorting
		float4* accad;			// Acceleration + density rate
		float4* spin;			// Components of spin tensor, for Jaumann stress rate
		tensor2d* acestn;		// Strain rate
		
		// Cache arrays for sorting and temporary
		float* float_cache;
		float4* float4_cache;
		tensor2d* tensor2d_cache;

		void InitializeParameters();

		void AllocateGpuMemory();

		void CopyDataFromHostToDevice();

		void CopyDataToIo();

		void SortArraysPrimary();

		void SortArraysAll();

		std::string TimeToString(double time);

	public:

		LoquatSph();

		~LoquatSph();

		void RunSimulation(int argc, char* argv[]);
};


#endif
