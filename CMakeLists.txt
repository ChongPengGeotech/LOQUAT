cmake_minimum_required(VERSION 3.11) 
project(LOQUAT VERSION 1.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)
find_package(CUDA REQUIRED)

include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_SOURCES
src/adami_boundary.cu
src/loquat.cu
src/loquat_io.cu
src/loquat_grid.cu
src/predictor_corrector.cu
src/field_smoothing.cu
src/particle_interaction.cu
)

set(CPP_SOURCES
src/tinyxml2.cpp
)

#link_directories(lib)

add_executable(LOQUAT src/main.cpp ${CUDA_SOURCES} ${CPP_SOURCES})
#target_link_libraries(LOQUAT ${Boost_LIBRARIES} loquat_grid particle_interaction)
set_property(TARGET LOQUAT PROPERTY CUDA_ARCHITECTURES "75")
