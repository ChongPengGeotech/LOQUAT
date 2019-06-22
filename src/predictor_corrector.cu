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
#include "predictor_corrector.h"

static __constant__ Parameters par;

/**
* Update stress tensor using Drucker-Prager constitutive model
* Please refer to the book Zhang et al. "Material Point Method" (张雄 等，《物质点法》)
*/
static __inline__ __device__ void MatModDP_cuk(
        const float deltat,
        const float4 omega,
        tensor2d dedt,
        float& p,
        float& ep,
        tensor2d& sigma)
{
    // Jaumann stress - rotation
    tensor2d acestr;
    acestr.xx = -2.0f * (omega.x * sigma.xy + omega.y * sigma.xz);
    acestr.yy =  2.0f * (omega.x * sigma.xy - omega.z * sigma.yz);
    acestr.zz =  2.0f * (omega.y * sigma.xz + omega.z * sigma.yz);
    acestr.xy =  omega.x * (sigma.xx - sigma.yy) - omega.z * sigma.xz - omega.y * sigma.yz;
    acestr.xz =  omega.y * (sigma.xx - sigma.zz) + omega.z * sigma.xy - omega.x * sigma.yz;
    acestr.yz =  omega.z * (sigma.yy - sigma.zz) + omega.x * sigma.xz + omega.y * sigma.xy;

    acestr = acestr + dedt * (2.0f * par.model_par_2);

    float deii =  dedt.xx + dedt.yy + dedt.zz;
    float ace = (par.model_par_1 - 2.0f / 3.0f * par.model_par_2) * deii;
    acestr.xx += ace;
    acestr.yy += ace;
    acestr.zz += ace;

    sigma = sigma + acestr * deltat;

    float sm = (sigma.xx + sigma.yy + sigma.zz) / 3.0f;
    sigma.xx -= sm;
    sigma.yy -= sm;
    sigma.zz -= sm;

    // Plastic correction
    float tenf = 0.0f;
    if(par.model_par_3 == 0.0f)
    {
        tenf = 0.0f;
    }
    else
    {
        tenf = par.model_par_5 / par.model_par_3;
    }

    float dpTi = sm - tenf; // Tensile yielding
    if(dpTi >= 0.0f)
    {
        float dlamda = dpTi / par.model_par_1;
        sm = tenf;
        ep += dlamda * (1.0f / 3.0f) * sqrtf(2.0f);
    }

	float tau = sqrtf(0.5f * J2(sigma));
	float dpFi = tau + par.model_par_3 * sm - par.model_par_5; // Shear yielding
	if(dpFi > 0.0f)
	{
		float dlamda = dpFi / (par.model_par_2 + par.model_par_1 * par.model_par_3 * par.model_par_4);

		sm -= par.model_par_1 * par.model_par_4 * dlamda;

        float ratio = tau < 1e-10f? 1.0f : (par.model_par_5 - par.model_par_3 * sm) / tau;
        sigma.xx *= ratio;
        sigma.yy *= ratio;
        sigma.zz *= ratio;
        sigma.xy *= ratio;
        sigma.xz *= ratio;
        sigma.yz *= ratio;

        ep += dlamda * sqrtf(1.0f / 3.0f + 2.0f / 9.0f * par.model_par_4 * par.model_par_4);
	}
    
    sigma.xx += sm;
    sigma.yy += sm;
    sigma.zz += sm;

    p = -sm;
}


/**
* Update stress tensor using hypoplastic model (Wang 2008)
*/
static __inline__ __device__ void MatModHypo_cuk(
    const float deltat,
    const tensor2d strain_rate,
    const float4 omega,
    tensor2d& sigma)
{
    // Jaumann
    tensor2d jau;
    jau.xx = -2.0f * (omega.x * sigma.xy + omega.y * sigma.xz);
    jau.yy =  2.0f * (omega.x * sigma.xy - omega.z * sigma.yz);
    jau.zz =  2.0f * (omega.y * sigma.xz + omega.z * sigma.yz);
    jau.xy =  omega.x * (sigma.xx - sigma.yy) - omega.z * sigma.xz - omega.y * sigma.yz;
    jau.xz =  omega.y * (sigma.xx - sigma.zz) + omega.z * sigma.xy - omega.x * sigma.yz;
    jau.yz =  omega.z * (sigma.yy - sigma.zz) + omega.x * sigma.xz + omega.y * sigma.xy;

    tensor2d str = sigma;
    if(par.model_par_5 > 0.0f)
    {
        str.xx -= par.model_par_5;
        str.yy -= par.model_par_5;
        str.zz -= par.model_par_5;
    }
    float t1 = str.xx + str.yy + str.zz;
    if(fabs(t1) < 4.0f) 
    {
        t1 = -4.0f;
        str.xx = -1.0f;
        str.yy = -1.0f;
        str.zz = -2.0f;
        str.xy = 0.0f;
        str.xy = 0.0f;
        str.yz = 0.0f;
    }

    tensor2d str_star = str;
    str_star.xx -= t1 / 3.0f;
    str_star.yy -= t1 / 3.0f;
    str_star.zz -= t1 / 3.0f;

    float d1 = strain_rate.xx + strain_rate.yy + strain_rate.zz;
    float t1d1 = J2(str, strain_rate);
    float eculidean = sqrtf(J2(strain_rate));

    tensor2d stress_rate = jau + strain_rate * (par.model_par_1 * t1) + str * (par.model_par_2 * d1 + par.model_par_3 * t1d1 / t1) + (str + str_star) * (par.model_par_4 * eculidean);

    sigma = sigma + stress_rate * deltat;

    float pres = (sigma.xx + sigma.yy + sigma.zz) / 3.0f;
    if(pres > par.model_par_5)
    {
        sigma.xx -= (pres - par.model_par_5);
        sigma.yy -= (pres - par.model_par_5);
        sigma.zz -= (pres - par.model_par_5);
    }
}



__global__ void Predictor_cuk(
        float4* pospres,
        float4* velrhop,
        float4* idep,
        tensor2d* str,
        tensor2d* stn,
        float4* accad,
        float4* spin,
        tensor2d* acestn,
        int npm)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < par.npm)
	{
        int i = par.npb + index;

        velrhop[i].x += 0.5f * par.dt * accad[i].x;
        velrhop[i].y += 0.5f * par.dt * accad[i].y;
        velrhop[i].z += 0.5f * par.dt * accad[i].z;
        velrhop[i].w += 0.5f * par.dt * accad[i].w;

        pospres[i].x += 0.5f * par.dt * velrhop[i].x;
        pospres[i].y += 0.5f * par.dt * velrhop[i].y;
        pospres[i].z += 0.5f * par.dt * velrhop[i].z; 

        tensor2d dstn = acestn[i] * (0.5f * par.dt);

        stn[i] = stn[i] + dstn;

        float p = pospres[i].w;
        float ep = idep[i].y;
        tensor2d sigma = str[i];
        float4 omega = spin[i];
        
        if(par.constitutive_model == 1)
        {
            MatModDP_cuk(0.5f * par.dt, omega, acestn[i], p, ep, sigma);
        }
        else
        {
            MatModHypo_cuk(0.5f * par.dt, acestn[i], omega, sigma);
            ep = sqrt(2.0f / 3.0f * J2(stn[i]));
        }

        pospres[i].w = p;
        str[i] = sigma;
        idep[i].y = ep;
    }
}


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
    tensor2d* acestn)
{
    if(par_host->npm == 0) return;

    checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpyAsync(pospres_ini, pospres, par_host->np * sizeof(float4), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyAsync(velrhop_ini, velrhop, par_host->np * sizeof(float4), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyAsync(idep_ini, idep, par_host->np * sizeof(float4), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyAsync(str_ini, str, par_host->np * sizeof(tensor2d), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyAsync(stn_ini, stn, par_host->np * sizeof(tensor2d), cudaMemcpyDeviceToDevice));

    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((par_host->npm + block.x - 1) / block.x, 1);

    Predictor_cuk <<<cuda_grid_size, block, 0>>> (pospres, velrhop, idep, str, stn, accad, spin, acestn, par_host->npm);
    getLastCudaError("Predictor_cuk failed...     \n");
}


__global__ void Corrector_cuk(
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
        tensor2d* acestn)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < par.npm)
	{
        int i = par.npb + index;

        velrhop[i].x = velrhop_ini[i].x + par.dt * accad[i].x;
        velrhop[i].y = velrhop_ini[i].y + par.dt * accad[i].y;
        velrhop[i].z = velrhop_ini[i].z + par.dt * accad[i].z;
        velrhop[i].w = velrhop_ini[i].w + par.dt * accad[i].w;

        pospres[i].x = 0.5f * (velrhop[i].x + velrhop_ini[i].x) * par.dt + pospres_ini[i].x;
        pospres[i].y = 0.5f * (velrhop[i].y + velrhop_ini[i].y) * par.dt + pospres_ini[i].y;
        pospres[i].z = 0.5f * (velrhop[i].z + velrhop_ini[i].z) * par.dt + pospres_ini[i].z;

        tensor2d dstn = acestn[i] * par.dt;
        stn[i] = stn_ini[i] + dstn;

        float p = pospres[i].w;
        float ep = idep_ini[i].y;
        tensor2d sigma = str_ini[i];
        float4 omega = spin[i];

        if(par.constitutive_model == 1)
        {
            MatModDP_cuk(par.dt, omega, acestn[i], p, ep, sigma);
        }
        else
        {
            MatModHypo_cuk(par.dt, acestn[i], omega, sigma);
            ep = sqrt(2.0f / 3.0f * J2(stn[i]));
        }

        pospres[i].w = p;
        str[i] = sigma;
        idep[i].y = ep;
    }
}


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
        tensor2d* acestn)
{
    if(par_host->npm == 0) return;

    checkCudaErrors(cudaMemcpyToSymbolAsync(par, par_device, sizeof(Parameters), 0, cudaMemcpyDeviceToDevice));

    dim3 block(128, 1);
    dim3 cuda_grid_size = dim3((par_host->npm + block.x - 1) / block.x, 1);

    Corrector_cuk <<<cuda_grid_size, block, 0, 0>>> (pospres, velrhop, idep, str, stn, pospres_ini, velrhop_ini, idep_ini, str_ini, stn_ini, accad, spin, acestn);
    getLastCudaError("Corrector_cuk failed...     \n");
}