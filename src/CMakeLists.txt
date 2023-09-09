project(inner)


find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)

add_subdirectory(deps)

set(C_SOURCE_FILES
		main.c
		partition.c
    spmxv_wrapper.c
)

add_executable(${PROJECT_NAME} ${C_SOURCE_FILES})

target_link_directories(${PROJECT_NAME}
	PRIVATE
		precompiled_deps
)
target_link_libraries(${PROJECT_NAME}
	PRIVATE
		m
		spmxv
		mmio
		patoh
		MPI::MPI_C
)
target_include_directories(${PROJECT_NAME}
	PRIVATE
		headers
		precompiled_deps/headers
)
