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

#include "loquat_io.h"

LoquatIo::LoquatIo(std::string name)
{
    project_name = name;

    pospres = NULL;
    velrhop = NULL;
    idep = NULL;
    
    str = NULL;
    stn = NULL;

    float4_cache = NULL;
    tensor2d_cache = NULL;

	par = NULL;
}


LoquatIo::~LoquatIo()
{
    if(pospres != NULL) cudaFreeHost(pospres);
    if(velrhop != NULL) cudaFreeHost(velrhop);
    if(idep != NULL) cudaFreeHost(idep);

    if(str != NULL) cudaFreeHost(str);
    if(stn != NULL) cudaFreeHost(stn);  

    if(float4_cache != NULL) cudaFreeHost(str);
    if(tensor2d_cache != NULL) cudaFreeHost(stn); 
}


// Check if a parameter exists, if it exists, return its value
boost::variant<std::string, int, float> LoquatIo::CheckParameter(
	std::string parameter_name,
	std::map<std::string, boost::variant<std::string, int, float> > parameters_from_xml)
{
	if(parameters_from_xml.count(parameter_name) == 0)
	{
		std::cout << "Error! " << parameter_name << " does not exist in the xml file!" << std::endl;
		exit(EXIT_FAILURE);
	}
	return parameters_from_xml.at(parameter_name);
}


void LoquatIo::LoadParameters()
{
	std::cout << std::endl;
	std::cout << "Loading parameters ..." << std::endl;
	
	std::string xml_name(project_name);
	xml_name.append(".xml");
	
	tinyxml2::XMLDocument xml;
	xml.LoadFile(xml_name.c_str());
	
	if(xml.Error())
	{
		std::cout << "Error! Cannot load the parameter file: " << xml_name << std::endl;
		std::cout << "Check the XML file, the names much match!" << std::endl;
		exit(EXIT_FAILURE);
	}
	else
	{
		std::cout << "The parameter file \"" << xml_name <<"\" is loaded sucessfully" << std::endl;
		std::cout << "Parsing the parameters" << std::endl;
		std::cout << std::endl;
	}
	
	tinyxml2::XMLElement* xml_parameters = xml.FirstChildElement();
	std::map<std::string, boost::variant<std::string, int, float> > parameters_from_xml;
	for(tinyxml2::XMLElement* iterator = xml_parameters->FirstChildElement(); iterator != NULL; iterator = iterator->NextSiblingElement())
	{
		// The name of the parameter
		std::string parameter_name(iterator->Name());
		
		// The actual value of the parameter
		boost::variant<std::string, int, float> parameter_value;
		if(iterator->Attribute("type", "int"))
		{
			parameter_value = atoi(iterator->GetText());
		}
		else if(iterator->Attribute("type", "float"))
		{
			parameter_value = (float)atof(iterator->GetText());
		}
		else
		{
			std::cout << "Wrong data type for " << parameter_name << "!!!" << std::endl;
            exit(EXIT_FAILURE);
		}
		
		std::cout << parameter_name << " = " << parameter_value << std::endl;

		parameters_from_xml.insert(std::pair<std::string, boost::variant<std::string, int, float> >(parameter_name, parameter_value));
	}

	par = new Parameters;

	// Particle number
	par->np = boost::get<int>(CheckParameter("number-of-particles", parameters_from_xml));
	par->npb = boost::get<int>(CheckParameter("number-of-boundary-particles", parameters_from_xml));
	par->npm = boost::get<int>(CheckParameter("number-of-material-particles", parameters_from_xml));
	if(par->np != par->npb + par->npm)
	{
		std::cout << "Error! Wrong number of particles ..." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Computational domain
	par->domain_size_x = boost::get<float>(CheckParameter("domain-size-x", parameters_from_xml));
	par->domain_size_y = boost::get<float>(CheckParameter("domain-size-y", parameters_from_xml));
	par->domain_size_z = boost::get<float>(CheckParameter("domain-size-z", parameters_from_xml));
	par->domain_min_x = boost::get<float>(CheckParameter("domain-min-x", parameters_from_xml));
	par->domain_min_y = boost::get<float>(CheckParameter("domain-min-y", parameters_from_xml));
	par->domain_min_z = boost::get<float>(CheckParameter("domain-min-z", parameters_from_xml));

	// Computational resolution
	par->dr = boost::get<float>(CheckParameter("particle-size", parameters_from_xml));
	par->hdr = boost::get<float>(CheckParameter("h-over-dr", parameters_from_xml));

	// Material property
	par->rho0 = boost::get<float>(CheckParameter("reference-density", parameters_from_xml));
	par->cs = boost::get<float>(CheckParameter("speed-of-sound", parameters_from_xml));
	par->cfl = boost::get<float>(CheckParameter("cfl-number", parameters_from_xml));
	par->constitutive_model = boost::get<int>(CheckParameter("constitutive-model", parameters_from_xml));
	if(par->constitutive_model == 1) // D-P model
	{
		par->model_par_1 = boost::get<float>(CheckParameter("elastic-K", parameters_from_xml));
		par->model_par_2 = boost::get<float>(CheckParameter("elastic-G", parameters_from_xml));
		par->model_par_3 = boost::get<float>(CheckParameter("DP-kphi", parameters_from_xml));
		par->model_par_4 = boost::get<float>(CheckParameter("DP-kpsi", parameters_from_xml));
		par->model_par_5 = boost::get<float>(CheckParameter("DP-kcohesion", parameters_from_xml));
	}
	else if(par->constitutive_model == 2) // Hypoplastic model (Wang 2009)
	{
		par->model_par_1 = boost::get<float>(CheckParameter("hypo-par-c1", parameters_from_xml));
		par->model_par_2 = boost::get<float>(CheckParameter("hypo-par-c2", parameters_from_xml));
		par->model_par_3 = boost::get<float>(CheckParameter("hypo-par-c3", parameters_from_xml));
		par->model_par_4 = boost::get<float>(CheckParameter("hypo-par-c4", parameters_from_xml));
		par->model_par_5 = boost::get<float>(CheckParameter("cohesion", parameters_from_xml));
	}
	
	// Boundary condition
	par->acc_x = boost::get<float>(CheckParameter("acceleration-x", parameters_from_xml));
	par->acc_y = boost::get<float>(CheckParameter("acceleration-y", parameters_from_xml));
	par->acc_z = boost::get<float>(CheckParameter("acceleration-z", parameters_from_xml));

	// Time related
	par->time_max = boost::get<float>(CheckParameter("time-max", parameters_from_xml));
	par->time = static_cast<double>(boost::get<float>(CheckParameter("time", parameters_from_xml)));
	par->dt = boost::get<float>(CheckParameter("time-step", parameters_from_xml));
	par->output_frequency = boost::get<float>(CheckParameter("output-frequency", parameters_from_xml));
	par->output_number = boost::get<int>(CheckParameter("output-number", parameters_from_xml));

	//Computational configuration
	par->density_shepard_filter = boost::get<int>(CheckParameter("density-shepard-filter", parameters_from_xml));
	par->stress_shepard_filter = boost::get<int>(CheckParameter("stress-shepard-filter", parameters_from_xml));
	par->dynamic_time_step = boost::get<int>(CheckParameter("dynamic-time-step", parameters_from_xml));
	par->integration_method = boost::get<int>(CheckParameter("integration-method", parameters_from_xml));
	par->viscosity_type = boost::get<int>(CheckParameter("viscosity-type", parameters_from_xml));
	if(par->viscosity_type == 1)
	{
		par->artificial_viscosity_alpha = boost::get<float>(CheckParameter("artificial-viscosity-alpha", parameters_from_xml));
	}

	par->artificial_pressure = boost::get<int>(CheckParameter("artificial-pressure", parameters_from_xml));
	if(par->artificial_pressure == 1)
	{
		par->artificial_pressure_coefficient = boost::get<float>(CheckParameter("artificial-pressure-coefficient", parameters_from_xml));
	}

	par->output_format = boost::get<int>(CheckParameter("output-format", parameters_from_xml));

	std::cout << " " << std::endl;
	std::cout << "Finish reading parameters" << std::endl;
	std::cout << " " << std::endl;
}


void LoquatIo::LoadParticles()
{
	std::cout << "Loading particle data ..." << std::endl;

    cudaHostAlloc((void **)&pospres, par->np * sizeof(float4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&velrhop, par->np * sizeof(float4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&idep, par->np * sizeof(float4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&str, par->np * sizeof(tensor2d), cudaHostAllocDefault);
    cudaHostAlloc((void **)&stn, par->np * sizeof(tensor2d), cudaHostAllocDefault);

    cudaHostAlloc((void **)&float4_cache, par->np * sizeof(float4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&tensor2d_cache, par->np * sizeof(tensor2d), cudaHostAllocDefault);

	std::string dat_name(project_name);
    dat_name.append(".dat");

	std::ifstream input_stream;
	input_stream.open(dat_name.c_str());
	if(input_stream)
	{
		for(int i = 0; i < par->np; i++)
		{
			input_stream >> pospres[i].x >> pospres[i].y >> pospres[i].z >> pospres[i].w >> velrhop[i].x >> velrhop[i].y >> velrhop[i].z >> velrhop[i].w >> idep[i].x >> idep[i].y >> str[i].xx >> str[i].yy >> str[i].zz >> str[i].xy >> str[i].xz >> str[i].yz >> stn[i].xx >> stn[i].yy >> stn[i].zz >> stn[i].xy >> stn[i].xz >> stn[i].yz;
		}
		input_stream.close();
	}
	else
	{
		std::cout << "Error! Cannot open " << dat_name << " !" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::cout << "Finish loading particle data ..." << std::endl;
}


void LoquatIo::SaveXml()
{
	std::stringstream ss;
	ss << project_name << "/output_" << par->output_number << ".xml";
	std::string filename = ss.str();

	std::ofstream ofs(filename.c_str(), std::ofstream::out);
	if(ofs)
	{
		ofs << "<?xml version=\"1.0\" encoding=\"utf-8\" ?>" << std::endl;
		ofs << "<parameters>" << std::endl;
		ofs << "  <number-of-particles type=\"int\">" << par->np << "</number-of-particles>" << std::endl;
		ofs << "  <number-of-boundary-particles type=\"int\">" << par->npb << "</number-of-boundary-particles>" << std::endl;
		ofs << "  <number-of-material-particles type=\"int\">" << par->npm << "</number-of-material-particles>" << std::endl;
		
		ofs << "  <domain-size-x type=\"float\">" << par->domain_size_x << "</domain-size-x>" << std::endl;
		ofs << "  <domain-size-y type=\"float\">" << par->domain_size_y << "</domain-size-y>" << std::endl;
		ofs << "  <domain-size-z type=\"float\">" << par->domain_size_z << "</domain-size-z>" << std::endl;
		ofs << "  <domain-min-x type=\"float\">" << par->domain_min_x << "</domain-min-x>" << std::endl;
		ofs << "  <domain-min-y type=\"float\">" << par->domain_min_y << "</domain-min-y>" << std::endl;
		ofs << "  <domain-min-z type=\"float\">" << par->domain_min_z << "</domain-min-z>" << std::endl;

		ofs << "  <particle-size type=\"float\">" << par->dr << "</particle-size>" << std::endl;
		ofs << "  <h-over-dr type=\"float\">" << par->hdr << "</h-over-dr>" << std::endl;

		ofs << "  <reference-density type=\"float\">" << par->rho0 << "</reference-density>" << std::endl;
		ofs << "  <speed-of-sound type=\"float\">" << par->cs << "</speed-of-sound>" << std::endl;
		ofs << "  <cfl-number type=\"float\">" << par->cfl << "</cfl-number>" << std::endl;
		ofs << "  <constitutive-model type=\"int\">" << par->constitutive_model << "</constitutive-model>" << std::endl;
		if(par->constitutive_model == 1) // D-P model
		{
			ofs << "  <elastic-K type=\"float\">" << par->model_par_1 << "</elastic-K>" << std::endl;
			ofs << "  <poisson-G type=\"float\">" << par->model_par_2 << "</poisson-G>" << std::endl;
			ofs << "  <DP-kphi type=\"float\">" << par->model_par_3 << "</DP-kphi>" << std::endl;
			ofs << "  <DP-kpsi type=\"float\">" << par->model_par_4 << "</DP-kpsi>" << std::endl;
			ofs << "  <DP-kcohesion type=\"float\">" << par->model_par_5 << "</DP-kcohesion>" << std::endl;
		}
		else if(par->constitutive_model == 2) // Hypoplastic model
		{
			ofs << "  <hypo-par-c1 type=\"float\">" << par->model_par_1 << "</hypo-par-c1>" << std::endl;
			ofs << "  <hypo-par-c2 type=\"float\">" << par->model_par_2 << "</hypo-par-c2>" << std::endl;
			ofs << "  <hypo-par-c3 type=\"float\">" << par->model_par_3 << "</hypo-par-c3>" << std::endl;
			ofs << "  <hypo-par-c4 type=\"float\">" << par->model_par_4 << "</hypo-par-c4>" << std::endl;
			ofs << "  <cohesion type=\"float\">" << par->model_par_5 << "</cohesion>" << std::endl;
		}

		ofs << "  <acceleration-x type=\"float\">" << par->acc_x << "</acceleration-x>" << std::endl;
		ofs << "  <acceleration-y type=\"float\">" << par->acc_y << "</acceleration-y>" << std::endl;
		ofs << "  <acceleration-z type=\"float\">" << par->acc_z << "</acceleration-z>" << std::endl;

		ofs << "  <time-max type=\"float\">" << par->time_max << "</time-max>" << std::endl;
		ofs << "  <time type=\"float\">" << static_cast<float>(par->time) << "</time>" << std::endl;
		ofs << "  <time-step type=\"float\">" << par->dt << "</time-step>" << std::endl;
		ofs << "  <output-frequency type=\"float\">" << par->output_frequency << "</output-frequency>" << std::endl;
		ofs << "  <output-number type=\"int\">" << par->output_number << "</output-number>" << std::endl;

		ofs << "  <density-shepard-filter type=\"int\">" << par->density_shepard_filter << "</density-shepard-filter>" << std::endl;
		ofs << "  <stress-shepard-filter type=\"int\">" << par->stress_shepard_filter << "</stress-shepard-filter>" << std::endl;
		ofs << "  <dynamic-time-step type=\"int\">" << par->dynamic_time_step << "</dynamic-time-step>" << std::endl;
		ofs << "  <integration-method type=\"int\">" << par->integration_method << "</integration-method>" << std::endl;
		ofs << "  <viscosity-type type=\"int\">" << par->viscosity_type << "</viscosity-type>" << std::endl;
		if(par->viscosity_type == 1)
		{
			ofs << "  <artificial-viscosity-alpha type=\"float\">" << par->artificial_viscosity_alpha << "</artificial-viscosity-alpha>" << std::endl;
		}

		ofs << "  <artificial-pressure type=\"int\">" << par->artificial_pressure << "</artificial-pressure>" << std::endl;
		if(par->artificial_pressure == 1)
		{
			ofs << "  <artificial-pressure-coefficient type=\"float\">" << par->artificial_pressure_coefficient << "</artificial-pressure-coefficient>" << std::endl;
		}

		ofs << "  <output-format type=\"int\">" << par->output_format << "</output-format>" << std::endl;

		ofs << "</parameters>" << std::endl;

		ofs.close();
	}
	else
	{
		std::cout << "Error! Cannot write file " << filename << " !" << std::endl;
	}
}


void LoquatIo::SortFloat4(int np, std::vector<int> sort_idx, float4* data, float4* cache)
{
    memcpy(cache, data, np * sizeof(float4));
    for(int i = 0; i < np; i++)
    {
        data[i] = cache[sort_idx[i]];
    }
}


void LoquatIo::SortTensor2d(int np, std::vector<int> sort_idx, tensor2d* data, tensor2d* cache)
{
    memcpy(cache, data, np * sizeof(tensor2d));
    for(int i = 0; i < np; i++)
    {
        data[i] = cache[sort_idx[i]];
    }
}


void LoquatIo::SaveParticles()
{
	#if COMPUTE_DISPLACEMENT
	GetDisplacement();
	#endif

    if(par->output_format == 1) // Save as plain data file
    {
        std::stringstream ss;
        ss << project_name << "/" << par->output_number << ".dat" << std::endl;
        std::string filename = ss.str();
    
        std::ofstream ofs(filename.c_str(), std::ofstream::out);
        if(ofs)
        {
            for(int i = 0; i < par->np; i++)
            {
                ofs << pospres[i].x << " " << pospres[i].y << " " << pospres[i].z << " " << pospres[i].w << " " << velrhop[i].x << " " << velrhop[i].y << " " << velrhop[i].z << " " << velrhop[i].w << " " << idep[i].x << " " << idep[i].y << " " << str[i].xx << " " << str[i].yy << " " << str[i].zz << " " << str[i].xy << " " << str[i].xz << " " << str[i].yz << " " << stn[i].xx << " " << stn[i].yy << " " << stn[i].zz << " " << stn[i].xy << " " << stn[i].xz << " " << stn[i].yz << std::endl;
            }
            ofs.close();
        }
        else
        {
            std::cout << "Error! Cannot write in " << filename << " !" << std::endl;
        }
    }

    else if(par->output_format == 2) // Save as vtu file for ParaView
    {
		std::clock_t clock_sorting = std::clock();

	    std::vector<int> sort_key;
    	std::vector<int> sort_idx;
	    sort_key.resize(par->np);
    	sort_idx.resize(par->np);

	    for(int i = 0; i < par->np; i++)
	    {
		    sort_key[i] = static_cast<int>(idep[i].x);
		    sort_idx[i] = i;
	    }

	    thrust::sort_by_key(&sort_key[0], &sort_key[0] + par->np, &sort_idx[0], thrust::greater<int>());

	    SortFloat4(par->np, sort_idx, pospres, float4_cache);
    	SortFloat4(par->np, sort_idx, velrhop, float4_cache);
	    SortFloat4(par->np, sort_idx, idep, float4_cache);
	    SortTensor2d(par->np, sort_idx, str, tensor2d_cache);
	    SortTensor2d(par->np, sort_idx, stn, tensor2d_cache);

	    std::vector<int> ids;
	    std::vector<int> id_positions;
	    int current_id = static_cast<int>(idep[0].x);
	    ids.push_back(current_id);
	    id_positions.push_back(0);
	    for(int i = 0; i < par->np; i++)
	    {
            int id_i = static_cast<int>(idep[i].x);
		    if(current_id != id_i)
    		{
			    current_id = id_i;
			    ids.push_back(current_id);
			    id_positions.push_back(i);
		    }
	    }

	    std::vector<int> number_of_particles;
	    for(int i = 0; i < ids.size(); i++)
	    {
		    if(i < ids.size() - 1)
		    {
			    number_of_particles.push_back(id_positions[i + 1] - id_positions[i]);
		    }
		    else if(i == ids.size() - 1)
		    {
			    number_of_particles.push_back(par->np - id_positions[i]);
		    }
	    }

		double time_sorting = (std::clock() - clock_sorting) / static_cast<double>(CLOCKS_PER_SEC);

		std::clock_t clock_writing = std::clock();
	    for(int i = 0; i < ids.size(); i++)
    	{
		    std::stringstream ss;
		    ss << project_name << "/" << ids[i] << "_" << par->output_number << ".vtu";
		    std::string filename = ss.str();

		    std::ofstream ofs(filename.c_str(), std::ofstream::out);
		    if(ofs)
		    {
			    ofs << "<?xml version=\"1.0\"?>" << std::endl;
			    ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">" << std::endl;
			    ofs << "  <UnstructuredGrid GhostLevel=\"0\">" << std::endl;
			    ofs << "    <Piece NumberOfPoints=\"" << number_of_particles[i] << "\" NumberOfCells=\"" << number_of_particles[i] << "\" Name=\"" << ids[i] << "\">" << std::endl;
    			ofs << "      <PointData>" << std::endl;

			    // Pressure
			    ofs << "        <DataArray type=\"Float32\" Name=\"Pressure\">" << std::endl;
			    for(int j = 0; j < number_of_particles[i]; j++)
			    {
				    int index = id_positions[i] + j;
				    ofs << pospres[index].w << " ";
    			}
    			ofs << "</DataArray>" << std::endl;

    			// Velocity
			    ofs << "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\">" << std::endl;
			    for(int j = 0; j < number_of_particles[i]; j++)
			    {
				    int index = id_positions[i] + j;
				    ofs << velrhop[index].x << " " << velrhop[index].y << " " << velrhop[index].z << " ";
			    }
    			ofs << "</DataArray>" << std::endl;

    			// Density
			    ofs << "        <DataArray type=\"Float32\" Name=\"Density\">" << std::endl;
			    for(int j = 0; j < number_of_particles[i]; j++)
			    {
				    int index = id_positions[i] + j;
				    ofs << velrhop[index].w << " ";
    			}
			    ofs << "</DataArray>" << std::endl;

                // Equivalent plastic strain
			    ofs << "        <DataArray type=\"Float32\" Name=\"PlasticStrain\">" << std::endl;
			    for(int j = 0; j < number_of_particles[i]; j++)
			    {
				    int index = id_positions[i] + j;
				    ofs << idep[index].y << " ";
    			}
			    ofs << "</DataArray>" << std::endl;

    			// Stress
    			ofs << "        <DataArray type=\"Float32\" Name=\"Stress\" NumberOfComponents=\"6\">" << std::endl;
			    for(int j = 0; j < number_of_particles[i]; j++)
			    {
				    int index = id_positions[i] + j;
                    ofs << str[index].xx << " " << str[index].yy << " " << str[index].zz << " " << str[index].xy << " " << str[index].xz << " " << str[index].yz << " ";
                }
                ofs << "</DataArray>" << std::endl;

                // Strain
                ofs << "        <DataArray type=\"Float32\" Name=\"Strain\" NumberOfComponents=\"6\">" << std::endl;
                for(int j = 0; j < number_of_particles[i]; j++)
                {
                    int index = id_positions[i] + j;
                    ofs << stn[index].xx << " " << stn[index].yy << " " << stn[index].zz << " " << stn[index].xy << " " << stn[index].xz << " " << stn[index].yz << " ";
                }
                ofs << "</DataArray>" << std::endl;

                ofs << "      </PointData>" << std::endl;

                ofs << "      <CellData>" << std::endl;
                ofs << "      </CellData>" << std::endl;

                // Position
                ofs << "      <Points>" << std::endl;
                ofs << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\">" << std::endl;
                for(int j = 0; j < number_of_particles[i]; j++)
                {
                    int index = id_positions[i] + j;
                    ofs << pospres[index].x << " " << pospres[index].y << " " << pospres[index].z << " ";
                }
                ofs << "</DataArray>" << std::endl;
                ofs << "      </Points>" << std::endl;

                // Cell connectivity
                ofs << "      <Cells>" << std::endl;
                ofs << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
                for(int j = 0; j < number_of_particles[i]; j++)
                {
                    ofs << j << " ";
                }
                ofs << "</DataArray>" << std::endl;
                ofs << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\"> " << std::endl;
                for(int j = 0; j < number_of_particles[i]; j++)
                {
                    ofs << j+1 << " ";
                }
                ofs << "</DataArray>" << std::endl;
                ofs << "        <DataArray type=\"Int64\" Name=\"types\" format=\"ascii\">" << std::endl;
                for(int j = 0; j < number_of_particles[i]; j++)
                {
                    ofs << 1 << " ";
                }
                ofs << "</DataArray>" << std::endl;
                ofs << "      </Cells>" << std::endl;

                ofs << "    </Piece>" << std::endl;
                ofs << "  </UnstructuredGrid>" << std::endl;
                ofs << "</VTKFile>" << std::endl;

                ofs.close();
		    }
		    else
		    {
			    std::cout << "Error! Cannot write file " << filename << " !" << std::endl;
    		}
		}
		
		double time_writing = (std::clock() - clock_writing) / static_cast<double>(CLOCKS_PER_SEC);
		printf("Save time: sorting = %0.4f, writing = %0.4f\n", time_sorting, time_writing);
    }
}


void LoquatIo::ResizeArraysLength(int new_np)
{
    if(pospres != NULL) cudaFreeHost(pospres);
    if(velrhop != NULL) cudaFreeHost(velrhop);
    if(idep != NULL) cudaFreeHost(idep);

    if(str != NULL) cudaFreeHost(str);
    if(stn != NULL) cudaFreeHost(stn);  

    if(float4_cache != NULL) cudaFreeHost(str);
    if(tensor2d_cache != NULL) cudaFreeHost(stn);  

    cudaHostAlloc((void **)&pospres, new_np * sizeof(float4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&velrhop, new_np * sizeof(float4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&idep, new_np * sizeof(float4), cudaHostAllocDefault);

    cudaHostAlloc((void **)&str, new_np * sizeof(tensor2d), cudaHostAllocDefault);
    cudaHostAlloc((void **)&stn, new_np * sizeof(tensor2d), cudaHostAllocDefault);

    cudaHostAlloc((void **)&float4_cache, new_np * sizeof(float4), cudaHostAllocDefault);
    cudaHostAlloc((void **)&tensor2d_cache, new_np * sizeof(tensor2d), cudaHostAllocDefault);
}