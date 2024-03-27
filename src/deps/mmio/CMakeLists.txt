cmake_minimum_required(VERSION 3.11)

project(mmio
    LANGUAGES C
    VERSION 0.0.1
)

set(C_SOURCE_FILES
		mmio.c
)

add_library(${PROJECT_NAME} STATIC
    ${C_SOURCE_FILES}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/headers
)