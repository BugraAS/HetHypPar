project(inner)


find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)

# Find CUDA package
find_package(CUDA REQUIRED)

add_subdirectory(deps)

set(C_SOURCE_FILES
		partition.c
		split.c
		cuSparse.c   
    spmxv_wrapper.c
)
include_directories(
    headers
    precompiled_deps/headers
    ${CUDA_INCLUDE_DIRS}
)

link_directories(
		precompiled_deps
)
link_libraries(
		m
		spmxv
		mmio
		patoh
		MPI::MPI_C
		${CUDA_LIBRARIES}
		${CUDA_cusparse_LIBRARY}
)
include_directories(
		headers
		precompiled_deps/headers
)


add_executable(${PROJECT_NAME} main.c ${C_SOURCE_FILES})   # SpMxV 

add_executable(patpart main_patoh.c ${C_SOURCE_FILES})  # PaToH  

add_executable(mxv-gpu main_gpu.c ${C_SOURCE_FILES})  # GPU-CPU hybrid SpMxV
