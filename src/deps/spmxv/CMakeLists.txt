cmake_minimum_required(VERSION 3.25)

project(spmxv
    LANGUAGES C
    VERSION 0.0.1
)


find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)


set(C_SOURCE_FILES
		buVector.c
		matrix.c
        mxv.c
        setup.c
)

add_library(${PROJECT_NAME} STATIC
    ${C_SOURCE_FILES}
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        MPI::MPI_C
        m
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/headers
)