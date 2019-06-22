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

/*
 Declare the data used in GPU computation, the main loop and related functions
*/

#pragma once
#ifndef __PREDICTOR_CORRECTOR_H__
#define __PREDICTOR_CORRECTOR_H__

#include <stdio.h>
#include <cuda_runtime.h>
#include <types.h>


void Predictor(
        Parameters* par_host,
        Parameters* par_device,
        float4* pospres,
        float4* velrhop,
        float4* idep,
        tensor2d* str,
        tensor2d* stn,
        float4* pospres_ini,
        float4* velrhop_ini,
        float4* idep_ini,
        tensor2d* str_ini,
        tensor2d* stn_ini,
        float4* accad,
        float4* spin,
        tensor2d* acestn);


void Corrector(
        Parameters* par_host,
        Parameters* par_device,
        float4* pospres,
        float4* velrhop,
        float4* idep,
        tensor2d* str,
        tensor2d* stn,
        float4* pospres_ini,
        float4* velrhop_ini,
        float4* idep_ini,
        tensor2d* str_ini,
        tensor2d* stn_ini,
        float4* accad,
        float4* spin,
        tensor2d* acestn);


#endif
