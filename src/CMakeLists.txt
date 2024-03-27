project(inner)


find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)

add_subdirectory(deps)

set(C_SOURCE_FILES
		partition.c
    spmxv_wrapper.c
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
)
include_directories(
		headers
		precompiled_deps/headers
)


add_executable(${PROJECT_NAME} main.c ${C_SOURCE_FILES})   # SpMxV 

add_executable(patpart main_patoh.c ${C_SOURCE_FILES})  # PaToH  


