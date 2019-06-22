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
#include "loquat.h"
#include "adami_boundary.h"
#include "particle_interaction.h"
#include "predictor_corrector.h"
#include "field_smoothing.h"

LoquatSph::LoquatSph()
{
    io = NULL;
    grid = NULL;
    par_host = NULL;
    par_device = NULL;
    timer = NULL;
    selected_device = 0;

	pospres = NULL;
	velrhop = NULL;
	idep = NULL;
	str = NULL;
	stn = NULL;
		
	pospres_ini = NULL;
	velrhop_ini = NULL;
	idep_ini = NULL;
	str_ini = NULL;
	stn_ini = NULL;
		
	accad = NULL;
	spin = NULL;
	acestn = NULL;
	
	float_cache = NULL;
	float4_cache = NULL;
	tensor2d_cache = NULL;
}


LoquatSph::~LoquatSph()
{
	if(pospres != NULL) cudaFree(pospres);
	if(velrhop != NULL) cudaFree(velrhop);
	if(idep != NULL) cudaFree(idep);
	if(str != NULL) cudaFree(str);
	if(str != NULL) cudaFree(str);

	if(pospres_ini != NULL) cudaFree(pospres_ini);
	if(velrhop_ini != NULL) cudaFree(velrhop_ini);
	if(idep_ini != NULL) cudaFree(idep_ini);
	if(str_ini != NULL) cudaFree(str_ini);
	if(str_ini != NULL) cudaFree(stn_ini);

	if(accad != NULL) cudaFree(accad);
    if(spin != NULL) cudaFree(spin);
	if(acestn != NULL) cudaFree(acestn);

	if(float_cache != NULL) cudaFree(float_cache);
	if(float4_cache != NULL) cudaFree(float4_cache);
	if(tensor2d_cache != NULL) cudaFree(tensor2d_cache);
}


std::string LoquatSph::TimeToString(double time)
{
	int days = 0;
	int hours = 0;
	int minutes = 0;
	int seconds = static_cast<int>(time);

	hours = seconds / 3600;
	if(hours > 24) 
	{
		days = hours / 24;
		hours = (seconds - days * 24 * 3600) / 3600;
	}

	minutes = (seconds % 3600) / 60;
	seconds = (seconds % 3600) % 60;

	std::stringstream string;

	string << std::setfill('0') << std::setw(2) << days << "d ";
	string << std::setfill('0') << std::setw(2) << hours << "h ";
	string << std::setfill('0') << std::setw(2) << minutes << "m ";
	string << std::setfill('0') << std::setw(2) << seconds << "s ";

	return string.str();
}


void LoquatSph::InitializeParameters()
{
	par_host->h = par_host->dr * par_host->hdr;
	par_host->i_h = 1.0f / par_host->h;
	par_host->eta = 0.01f * par_host->h * par_host->h;
	
	par_host->m = par_host->dr * par_host->dr * par_host->dr * par_host->rho0;

	par_host->cell_num_x = static_cast<int>(0.5f * par_host->i_h * par_host->domain_size_x) + 1;
	par_host->cell_num_y = static_cast<int>(0.5f * par_host->i_h * par_host->domain_size_y) + 1;
	par_host->cell_num_z = static_cast<int>(0.5f * par_host->i_h * par_host->domain_size_z) + 1;
	par_host->ncell_one_layer = par_host->cell_num_x * par_host->cell_num_y * par_host->cell_num_z;
	par_host->ncell = 2 * par_host->ncell_one_layer + 1;
	par_host->ncell_delete = par_host->ncell - 1;

	printf("ncellx = %d, ncelly = %d, ncellz = %d, ncell = %d\n", par_host->cell_num_x, par_host->cell_num_y, par_host->cell_num_z, par_host->ncell);
	printf("\n");

	// Wendland C2 function
	par_host->kernel_normalization_par = 42.0f / 256.0f / (powf(par_host->h, 3) * PI);
	par_host->kernel_gradient_normalization_par = -5.0f * 42.0f / 256.0f / (powf(par_host->h, 5) * PI);

	if(par_host->artificial_pressure == 1)
	{
		par_host->kernel_zero = par_host->kernel_normalization_par * (2.0f) * (2.0f) * (2.0f) * (2.0f) * (0.5f);
		float q = par_host->dr * par_host->i_h;
		par_host->kernel_dr = par_host->kernel_normalization_par * (2.0f - q) * (2.0f - q) * (2.0f - q) * (2.0f - q) * (q + 0.5f);
	}
}


void LoquatSph::AllocateGpuMemory()
{
    checkCudaErrors(cudaMalloc((void **)&par_device, sizeof(Parameters)));

    checkCudaErrors(cudaMalloc((void **)&pospres, par_host->np * sizeof(float4)));
    checkCudaErrors(cudaMalloc((void **)&velrhop, par_host->np * sizeof(float4)));
    checkCudaErrors(cudaMalloc((void **)&idep, par_host->np * sizeof(float4)));
    checkCudaErrors(cudaMalloc((void **)&str, par_host->np * sizeof(tensor2d)));
    checkCudaErrors(cudaMalloc((void **)&stn, par_host->np * sizeof(tensor2d)));

    checkCudaErrors(cudaMalloc((void **)&pospres_ini, par_host->np * sizeof(float4)));
    checkCudaErrors(cudaMalloc((void **)&velrhop_ini, par_host->np * sizeof(float4)));
    checkCudaErrors(cudaMalloc((void **)&idep_ini, par_host->np * sizeof(float4)));
    checkCudaErrors(cudaMalloc((void **)&str_ini, par_host->np * sizeof(tensor2d)));
    checkCudaErrors(cudaMalloc((void **)&stn_ini, par_host->np * sizeof(tensor2d)));

    checkCudaErrors(cudaMalloc((void **)&accad, par_host->np * sizeof(float4)));
    checkCudaErrors(cudaMalloc((void **)&spin, par_host->np * sizeof(float4)));
    checkCudaErrors(cudaMalloc((void **)&acestn, par_host->np * sizeof(tensor2d)));

    checkCudaErrors(cudaMalloc((void **)&float_cache, par_host->np * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&float4_cache, par_host->np * sizeof(float4)));
    checkCudaErrors(cudaMalloc((void **)&tensor2d_cache, par_host->np * sizeof(tensor2d)));
}


void LoquatSph::CopyDataFromHostToDevice()
{
    checkCudaErrors(cudaMemcpy(pospres, io->GetPosPres(), par_host->np * sizeof(float4), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(velrhop, io->GetVelRhop(), par_host->np * sizeof(float4), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(idep, io->GetIdEp(), par_host->np * sizeof(float4), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(str, io->GetStr(), par_host->np * sizeof(tensor2d), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(stn, io->GetStn(), par_host->np * sizeof(tensor2d), cudaMemcpyHostToDevice));
}


void LoquatSph::CopyDataToIo()
{
    checkCudaErrors(cudaMemcpy(io->GetPosPres(), pospres, par_host->np * sizeof(float4), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(io->GetVelRhop(), velrhop, par_host->np * sizeof(float4), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(io->GetIdEp(), idep, par_host->np * sizeof(float4), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(io->GetStr(), str, par_host->np * sizeof(tensor2d), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(io->GetStn(), stn, par_host->np * sizeof(tensor2d), cudaMemcpyDeviceToHost));

	memcpy(io->GetParameters(), par_host, sizeof(Parameters));
}


void LoquatSph::SortArraysPrimary()
{
	grid->SortArrayFloat4(par_host->np, pospres, float4_cache);
	grid->SortArrayFloat4(par_host->np, velrhop, float4_cache);
	grid->SortArrayFloat4(par_host->np, idep, float4_cache);
	grid->SortArrayTensor2d(par_host->np, str, tensor2d_cache);
	grid->SortArrayTensor2d(par_host->np, stn, tensor2d_cache);
}


void LoquatSph::SortArraysAll()
{
	grid->SortArrayFloat4(par_host->np, pospres, float4_cache);
	grid->SortArrayFloat4(par_host->np, velrhop, float4_cache);
	grid->SortArrayFloat4(par_host->np, idep, float4_cache);
	grid->SortArrayTensor2d(par_host->np, str, tensor2d_cache);
	grid->SortArrayTensor2d(par_host->np, stn, tensor2d_cache);

	grid->SortArrayFloat4(par_host->np, pospres_ini, float4_cache);
	grid->SortArrayFloat4(par_host->np, velrhop_ini, float4_cache);
	grid->SortArrayFloat4(par_host->np, idep_ini, float4_cache);
	grid->SortArrayTensor2d(par_host->np, str_ini, tensor2d_cache);
	grid->SortArrayTensor2d(par_host->np, stn_ini, tensor2d_cache);
}


void LoquatSph::RunSimulation(int argc, char* argv[])
{
    std::cout << "**********************************" << std::endl;
	std::cout << "** LOQUAT Solver for single GPU **" << std::endl;
	std::cout << "**********************************" << std::endl;
	std::cout << " " << std::endl;

	switch(argc)
	{
		case 1:
		{
			std::cout << "Error! Please give the project name!" << std::endl;
			exit(EXIT_FAILURE);
			break;
		}
		
		case 2:
		{
			project_name = std::string(argv[1]);
			selected_device = 0;
			break;
		}
		
		case 3:
		{
			project_name = std::string(argv[1]);
			selected_device = atoi(argv[2]);
			break;
		}
		
		default:
			break;
	}

	int number_of_gpus;
	cudaGetDeviceCount(&number_of_gpus);
	std::cout << "Number of GPUs on this computer: " << number_of_gpus << std::endl;
	std::cout << "Information of the GPUs: " << std::endl;
	for(int i = 0; i < number_of_gpus; i++)
	{
		std::stringstream info;
		info << "  (" << i << ") ";
		cudaDeviceProp gpu_properties;
		cudaGetDeviceProperties(&gpu_properties, i);
		info << gpu_properties.name << " " << gpu_properties.totalGlobalMem / 1048576 << " MB"
			 << " (" << gpu_properties.major << "." << gpu_properties.minor << ")";
		std::cout << info.str() << std::endl;
	}
	std::cout << "Device check finished" << std::endl;

	cudaSetDevice(selected_device);
	std::cout << std::endl;
	std::cout << "Simulation running on GPU " << selected_device << std::endl;

    io = new LoquatIo(project_name);

    io->LoadParameters();

    io->LoadParticles();

    par_host = io->GetParameters();

    InitializeParameters();

    AllocateGpuMemory();

    CopyDataFromHostToDevice();

    grid = new LoquatGrid();

	grid->GridConfig(par_host);
	
	std::string path(project_name);
	if(!boost::filesystem::exists(path))
	{
		std::cout << "Create results folder: " << path << std::endl;
		boost::filesystem::create_directory(path);
	}

	cudaMemcpy(par_device, par_host, sizeof(Parameters), cudaMemcpyHostToDevice);

	grid->NeighborSearch(par_host, par_device, pospres, idep);
	SortArraysPrimary();

	CopyDataToIo();

	io->SaveXml();
	io->SaveParticles();

	double time = static_cast<double>(par_host->time);

	int step = 0;
	double output_time = 0.0;
	int step_interval = 0;

	timer = new Timer;
	timer->total = 0.0;
	timer->neighbor_search = 0.0;
	timer->adami_boundary = 0.0;
	timer->interaction = 0.0;
	timer->integration = 0.0;
	timer->saving = 0.0;

	timer->clock_initial = std::clock();
	timer->clock_interval = std::clock();

	while(time < static_cast<double>(par_host->time_max))
	{
		//printf("\nstep = %d\n", step);
		if(par_host->dynamic_time_step == 1)
		{
			// DynamicTimeStep(par_host, par_device, pospres, accad, idep, float_cache);
		}
		else
		{
			par_host->dt = par_host->cfl * par_host->h / par_host->cs;
		}
		par_host->dt = par_host->cfl * par_host->h / par_host->cs;

		par_host->time = static_cast<float>(time);

		cudaMemcpy(par_device, par_host, sizeof(Parameters), cudaMemcpyHostToDevice);

		printf(" SimTime = %0.5f, dt = %0.3e\r", par_host->time, par_host->dt);
		std::cout.flush();

		// Prediction
		{
			//========================Neighbor search========================
			timer->clock_neighbor_search = std::clock();
			grid->NeighborSearch(par_host, par_device, pospres, idep);
			SortArraysPrimary();
			cudaDeviceSynchronize();
			timer->neighbor_search += (std::clock() - timer->clock_neighbor_search) / static_cast<double>(CLOCKS_PER_SEC);

			//=======================Boundary treatment=======================
			timer->clock_adami_boundary = std::clock();
			AdamiBoundary(par_host, par_device, pospres, velrhop, str, grid->GetCellBeg(), grid->GetCellEnd());
			cudaDeviceSynchronize();
			timer->adami_boundary += (std::clock() - timer->clock_adami_boundary) / static_cast<double>(CLOCKS_PER_SEC);

			//==========================Interaction===========================
			timer->clock_interaction = std::clock();
			ParticleInteraction(par_host, par_device, pospres, velrhop, idep, str, stn, accad, spin, acestn, grid->GetCellBeg(), grid->GetCellEnd());
			cudaDeviceSynchronize();
			timer->interaction += (std::clock() - timer->clock_interaction) / static_cast<double>(CLOCKS_PER_SEC);

			//==========================Integration===========================
			timer->clock_integration = std::clock();
			Predictor(par_host, par_device, pospres, velrhop, idep, str, stn, pospres_ini, velrhop_ini, idep_ini, str_ini, stn_ini, accad, spin, acestn);
			cudaDeviceSynchronize();
			timer->integration += (std::clock() - timer->clock_integration) / static_cast<double>(CLOCKS_PER_SEC);
		}

		// Correction
		{
			//========================Neighbor search========================
			timer->clock_neighbor_search = std::clock();
			grid->NeighborSearch(par_host, par_device, pospres, idep);
			SortArraysAll();
			cudaDeviceSynchronize();
			timer->neighbor_search += (std::clock() - timer->clock_neighbor_search) / static_cast<double>(CLOCKS_PER_SEC);

			//=======================Boundary treatment=======================
			timer->clock_adami_boundary = std::clock();
			AdamiBoundary(par_host, par_device, pospres, velrhop, str, grid->GetCellBeg(), grid->GetCellEnd());
			cudaDeviceSynchronize();
			timer->adami_boundary += (std::clock() - timer->clock_adami_boundary) / static_cast<double>(CLOCKS_PER_SEC);

			//==========================Interaction===========================
			timer->clock_interaction = std::clock();
			ParticleInteraction(par_host, par_device, pospres, velrhop, idep, str, stn, accad, spin, acestn, grid->GetCellBeg(), grid->GetCellEnd());
			cudaDeviceSynchronize();
			timer->interaction += (std::clock() - timer->clock_interaction) / static_cast<double>(CLOCKS_PER_SEC);

			//==========================Integration===========================
			timer->clock_integration = std::clock();
			Corrector(par_host, par_device, pospres, velrhop, idep, str, stn, pospres_ini, velrhop_ini, idep_ini, str_ini, stn_ini, accad, spin, acestn);
			cudaDeviceSynchronize();
			timer->integration += (std::clock() - timer->clock_integration) / static_cast<double>(CLOCKS_PER_SEC);
		}


		// Stress regularization
		if(par_host->stress_shepard_filter != 0 && step % par_host->stress_shepard_filter == 0 && step != 0)
		{
			timer->clock_neighbor_search = std::clock();
			grid->NeighborSearch(par_host, par_device, pospres, idep);
			SortArraysPrimary();
			cudaDeviceSynchronize();
			timer->neighbor_search += (std::clock() - timer->clock_neighbor_search) / static_cast<double>(CLOCKS_PER_SEC);

			StressRegularization(par_host, par_device, pospres, velrhop, str, tensor2d_cache, grid->GetCellBeg(), grid->GetCellEnd());
		}

		// Saving
		if(output_time > static_cast<double>(par_host->output_frequency))
		{
			timer->clock_saving = std::clock();

			double time_interval = (std::clock() - timer->clock_interval) / static_cast<double>(CLOCKS_PER_SEC);
			double fps = static_cast<double>(step - step_interval + 1) / time_interval;

			double used_time = (std::clock() - timer->clock_initial) / static_cast<double>(CLOCKS_PER_SEC);
			double predicted_time = (par_host->time_max / par_host->time) * used_time;

			printf(" \n");
			printf("----------Save-----------\n");
			printf("Time = %0.5f | Step = %d | FPS = %0.5f\n", par_host->time, step, fps);
			std::cout << "Clock time total : " << TimeToString(predicted_time) << std::endl;
			std::cout << "Clock time used  : " << TimeToString(used_time) << std::endl;
			std::cout << "Clock time needed: " << TimeToString(predicted_time - used_time) << std::endl;
			printf("Number of particles: %d | Number of cells: %d\n", par_host->np, par_host->ncell);
			
			par_host->output_number++;
			
			CopyDataToIo();

			io->SaveXml();
			io->SaveParticles();

			std::cout << "Save file: " << project_name << "/";
			if(par_host->output_format == 1)
			{
				std::cout << par_host->output_number << ".dat + " << par_host->output_number << ".xml" << std::endl;
			}
			else if(par_host->output_format == 2)
			{
				std::cout << "*_" << par_host->output_number << ".vtu + " << par_host->output_number << ".xml" << std::endl;
			}
			std::cout << std::endl;

			output_time = 0.0;
			step_interval = step;
			timer->clock_interval = std::clock();

			timer->saving += (std::clock() - timer->clock_saving) / static_cast<double>(CLOCKS_PER_SEC);
		}

		time += static_cast<double>(par_host->dt);
		output_time += static_cast<double>(par_host->dt);
		step++;
	}

	timer->total = (std::clock() - timer->clock_initial) / static_cast<double>(CLOCKS_PER_SEC);
	double fps_average = static_cast<double>(step) / (timer->total - timer->saving);

	std::cout << "Simulation finished ..." << std::endl;
	std::cout << std::endl;
	std::cout << "----------SUMMARY----------" << std::endl;
	std::cout << "1. Total clock time = " << TimeToString(timer->total) << std::endl;
	std::cout << "2. Search and sorting = " << TimeToString(timer->neighbor_search) << std::endl;
	std::cout << "3. Boundary treatment = " << TimeToString(timer->adami_boundary) << std::endl;
	std::cout << "4. Interaction = " << TimeToString(timer->interaction) << std::endl;
	std::cout << "5. Integration = " << TimeToString(timer->integration) << std::endl;
	std::cout << "6. Saving = " << TimeToString(timer->saving) << std::endl;
	std::cout << "7. Total steps = " << step << std::endl;
	std::cout << "8. Average time step = " << par_host->time / static_cast<float>(step) << std::endl;
	std::cout << "9. Average FPS = " << fps_average << std::endl;

}
