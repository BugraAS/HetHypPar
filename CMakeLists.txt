cmake_minimum_required(VERSION 3.11)
project(outer)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)