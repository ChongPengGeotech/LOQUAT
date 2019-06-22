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
 This file declares all global parameters. Any additional parameters should be added here.
*/

#pragma once
#ifndef __TYPES_H__
#define __TYPES_H__


#define PI (3.14159265358979323846f)

#include <ctime>
#include <cuda_runtime.h>

// Second order tensor and related operations
struct {
	float xx, yy, zz, xy, xz, yz;
} typedef tensor2d;

static __inline__ __host__ __device__ tensor2d make_tensor2d(float v)
{
	tensor2d result = {v, v, v, v, v, v};
	return(result);
}

static __inline__ __host__ __device__ tensor2d make_tensor2d(float v1, float v2, float v3, float v4, float v5, float v6)
{
	tensor2d result = {v1, v2, v3, v4, v5, v6};
	return(result);
}

static __inline__ __host__ __device__ tensor2d operator + (const tensor2d& v1, const tensor2d& v2)
{
	return(make_tensor2d(v1.xx + v2.xx, v1.yy + v2.yy, v1.zz + v2.zz, v1.xy + v2.xy, v1.xz + v2.xz, v1.yz + v2.yz));
}

static __inline__ __host__ __device__ tensor2d operator - (const tensor2d& v1, const tensor2d& v2)
{
	return(make_tensor2d(v1.xx - v2.xx, v1.yy - v2.yy, v1.zz - v2.zz, v1.xy - v2.xy, v1.xz - v2.xz, v1.yz - v2.yz));
}

static __inline__ __host__ __device__ tensor2d operator * (const tensor2d& v1, const float& v2)
{
	return(make_tensor2d(v1.xx * v2, v1.yy * v2, v1.zz * v2, v1.xy * v2, v1.xz * v2, v1.yz * v2));
}

static __inline__ __host__ __device__ tensor2d operator / (const tensor2d& v1, const float& v2)
{
	return(make_tensor2d(v1.xx / v2, v1.yy / v2, v1.zz / v2, v1.xy / v2, v1.xz / v2, v1.yz / v2));
}

static __inline__ __host__ __device__ float trace(const tensor2d& v)
{
	return(v.xx + v.yy + v.zz);
}

static __inline__ __host__ __device__ float J2(const tensor2d& v)
{
	return(v.xx * v.xx + v.yy * v.yy + v.zz * v.zz + 2.0 * v.xy * v.xy + 2.0 * v.xz * v.xz + 2.0 * v.yz * v.yz);
}

static __inline__ __host__ __device__ float J2(const tensor2d& v1, const tensor2d& v2)
{
	return(v1.xx * v2.xx + v1.yy * v2.yy + v1.zz * v2.zz + 2.0 * v1.xy * v2.xy + 2.0 * v1.xz * v2.xz + 2.0 * v1.yz * v2.yz);
}


// Parameters for the SPH simulation
struct {
	int np;					// Total number of particle
	int npb;				// Number of boundary particle
	int npm;				// Number of material particle
	int np_delete;			// Number of deleted particle

	float domain_size_x;	// Size of domain in x-direction
	float domain_size_y;	// Size of domain in y-direction
	float domain_size_z;	// Size of domain in z-direction
	
	float domain_min_x;		// Starting point of the domain - x
	float domain_min_y;		// Starting point of the domain - y
	float domain_min_z;		// Starting point of the domain - z
	
	int cell_num_x;			// Number of cells in x direction
	int cell_num_y;			// Number of cells in y direction
	int cell_num_z;			// Number of cells in z direction
	int ncell;				// Total number of cell
	int ncell_one_layer;	// Number of cells in one layer of grid
	int ncell_delete;		// The cell index for deleted particles
	
	float h;				// Smoothing length
	float i_h;				// Inverse of smoothing length
	float eta;				// 0.01 * h * h, to prevent singularity
	float dr;				// Particle size
	float hdr;				// Ratio between h and dr, hdr = h / dr

	float m;				// Mass of particle
	float cs;				// Speed of sound
	float cfl;				// CFL number
	float rho0;				// Reference density

	int constitutive_model;	// Type of constitutive model: 1 - Drucker_Prager, 2 - Hypoplastic model
	float model_par_1;		// Model parameter
	float model_par_2;		// Model parameter
	float model_par_3;		// Model parameter
	float model_par_4;		// Model parameter
	float model_par_5;		// Model parameter

	int density_shepard_filter;					// Interval of steps to apply shepard filter to density
	int stress_shepard_filter;					// Interval of steps to apply shepard filter to stress
	int dynamic_time_step;						// Switch for variable time step
	int integration_method;						// Method for time integration, 1- Predictor-Corrector is implemented
	int viscosity_type;							// Type of viscosity treatment, 1- artificial viscosity
	float artificial_viscosity_alpha;			// Parameter for the artificial viscosity

	int artificial_pressure;					// Switch for artificial pressure: 0 - deactivated; 1 - activiated
	float artificial_pressure_coefficient;		// Parameter for the artificial pressure
	
	float acc_x;								// Body force x
	float acc_y;								// Body force y
	float acc_z;								// Body force z
	
	float time_max;								// Total simulation time
	float time;									// Current time
	float dt;									// Time step
	float output_frequency;						// Time interval for saving results
	int output_number;							// Number of output files

	float kernel_normalization_par;				// Kernel renormalization parameter
	float kernel_gradient_normalization_par;	// Kernal gradient renormalization parameter

	float kernel_zero;							// W(0)
	float kernel_dr;							// W(dr)

	int output_format;							// 1 - dat; 2 - vtu;
} typedef Parameters;


struct {
	double total;
	double neighbor_search;
	double adami_boundary;
	double interaction;
	double integration;
	double saving;
	
	std::clock_t clock_initial;
	std::clock_t clock_neighbor_search;
	std::clock_t clock_adami_boundary;
	std::clock_t clock_interaction;
	std::clock_t clock_integration;
	std::clock_t clock_saving;
	std::clock_t clock_interval;
} typedef Timer;

#endif
