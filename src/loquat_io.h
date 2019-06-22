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
#ifndef __LOQUAT_IO_H__
#define __LOQUAT_IO_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <map>
#include <vector>
#include <string>
#include <ctime>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <boost/variant.hpp>
#include <boost/filesystem.hpp>
#include <types.h>
#include "tinyxml2.h"

class LoquatIo
{
	private:
		std::string project_name;

		Parameters* par;

		// Primary arrays
		float4* pospres;
		float4* velrhop;
		float4* idep;
		tensor2d* str;
		tensor2d* stn;

		float4* float4_cache;
		tensor2d* tensor2d_cache;

		boost::variant<std::string, int, float> CheckParameter
		(
			std::string parameter_name,	std::map<std::string, 
			boost::variant<std::string, int, float> > parameters_from_xml
		);

		void SortFloat4(int np, std::vector<int> sort_idx, float4* data, float4* cache);

		void SortTensor2d(int np, std::vector<int> sort_idx, tensor2d* data, tensor2d* cache);

	public:
		
		LoquatIo(std::string name);

		~LoquatIo();

		Parameters* GetParameters() { return par; }

		float4* GetPosPres() {return pospres; }

		float4* GetVelRhop() {return velrhop; }

		float4* GetIdEp() {return idep; }

		tensor2d* GetStr() {return str; }

		tensor2d* GetStn() {return stn; }

		void LoadParameters();

		void LoadParticles();

		void SaveXml();

		void SaveParticles();

		void ResizeArraysLength(int new_np);
};


#endif
