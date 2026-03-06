# ------------------------------------------------------------------------------
# -               V E R S I O N   I D E N T I F I C A T I O N                  -
# ------------------------------------------------------------------------------
# File:                    %name: CMakeLists.txt %
# Date Created:            %date_created: Sat Feb 08 17:16:42 2026 %
# CM Version               %version: 1 %
# CM Instance              %instance: 1 %
# Author:                  %created_by: MBusato %
# ------------------------------------------------------------------------------
# -                          D E S C R I P T I O N                             -
# ------------------------------------------------------------------------------
# CMake template for C++ projects with integrated logging from Shared-Libraries.
#
# QUICK START GUIDE:
#   1. Create a 'build' Directory:       mkdir -p build/cmake && cd build/cmake
#   2. Configure the Project:            cmake -DCMAKE_BUILD_TYPE=Debug ../..
#                                        cmake -DCMAKE_BUILD_TYPE=Release ../..
#   3. Build the Project:                cmake --build .
#   4. Run the Executable:               ./../bin/My_Project (Linux/WSL)
#                                        ..\bin\My_Project.exe (Windows)
#
# CUSTOMISATION:
#   - Change PROJECT_NAME Below to the Current Project Name.
#   - Add any Source Files to add_executable().
#   - Add Include Directories to target_include_directories().
#   - Add Libraries to target_link_libraries().
#
# ------------------------------------------------------------------------------
# -                          L I C E N S I N G                                 -
# ------------------------------------------------------------------------------
# No licensing restrictions are applicable to the software below.
# ------------------------------------------------------------------------------

# Set Minimum Required CMake Version.
cmake_minimum_required(VERSION 3.16)

# Set C++ Standard (C++17 for Good Balance of Features & Compatibility).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable Export of Compile Commands.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable Verbose Makefile Output.
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set Build Type to Debug if Not Specified.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()


# ------------------------------------------------------------------------------
#                              PROJECT SETUP
# ------------------------------------------------------------------------------
# Set Project Name (CHANGE THIS to Current Project Name).
project("My_Project" VERSION 1.0.0 LANGUAGES CXX)

# Print Configuration Information.
message(STATUS "")
message(STATUS "=== Building ${PROJECT_NAME} v${PROJECT_VERSION} ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         OUTPUT DIRECTORY CONFIGURATION
# ------------------------------------------------------------------------------
# Set Output Directories (Relative to Build Directory).
# Executables go to ../bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Static Libraries (.lib/.a) go to ../lib/
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)

# Shared Libraries (.dll/.so) go to ../bin/
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Debug Symbols (.pdb) go to ../temp/
set(CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../temp)

# Create Directories if they Don't Exist.
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
file(MAKE_DIRECTORY ${CMAKE_PDB_OUTPUT_DIRECTORY})
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         COMPILER FLAGS & OPTIONS
# ------------------------------------------------------------------------------
# Cross-Platform Compiler Flags.
if(MSVC)
  # Microsoft Visual C++ Compiler Flags.
  message(STATUS "Configuring for MSVC Compiler.")
  
  # Warning Level 4 (High Warning Level).
  add_compile_options(/W4)
  
  # Treat Warnings as Errors (Optional).
  # add_compile_options(/WX)
  
  # Disable Specific Warnings (Add as Needed).
  # add_compile_options(/wd4996)  # Disable Deprecated Function Warnings.
  
  # Enable Multi-Processor Compilation (Faster Builds).
  add_compile_options(/MP)
  
  # Disable Incremental Linking (Prevents .ilk Files).
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /INCREMENTAL:NO")
  set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /INCREMENTAL:NO")
  
  # Add Windows-Specific Definitions.
  add_compile_definitions(_WIN32 WIN32_LEAN_AND_MEAN NOMINMAX)
  
  # Unicode Support (Optional).
  add_compile_definitions(UNICODE _UNICODE)

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # GCC or Clang Compiler Flags (Linux, WSL, macOS).
  message(STATUS "Configuring for ${CMAKE_CXX_COMPILER_ID} Compiler.")
  
  # Enable All Warnings.
  add_compile_options(-Wall)
  
  # Enable Extra Warnings.
  add_compile_options(-Wextra)
  
  # Enable Pedantic Mode (Strict Standard Compliance).
  add_compile_options(-pedantic)
  
  # Treat Warnings as Errors (Optional).
  # add_compile_options(-Werror)
  
  # Optimisation Flags.
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    # Debug: No Optimisation (Include Debug Symbols).
    add_compile_options(-O0 -g)
  else()
    # Release: Optimise for Speed.
    add_compile_options(-O2)
  endif()
  
  # Enable Position Independent Code (Required for Shared Libraries).
  add_compile_options(-fPIC)

else()
  message(WARNING "Unknown Compiler: ${CMAKE_CXX_COMPILER_ID}.")
endif()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                    FETCH LOGGER FROM GITHUB REPOSITORY
# ------------------------------------------------------------------------------
# Include FetchContent Module to Download External Dependencies.
include(FetchContent)

# Print Message for the Modules being Targeted. 
message(STATUS "Fetching Logging Module from GitHub...")

# Fetch the Shared-Libraries Repository.
FetchContent_Declare(
  shared_libraries
  GIT_REPOSITORY git@github.com:MaxB-RnD/Shared-Libraries.git
  GIT_TAG main  # Use 'main' Branch (Specify a Commit Hash for Stability).
  SOURCE_SUBDIR Loggers  # Only Use the Loggers Subdirectory.
)

# Make the Content Available.
FetchContent_MakeAvailable(shared_libraries)

# Get the Source Directory of the Fetched Content.
FetchContent_GetProperties(shared_libraries)
if(NOT shared_libraries_POPULATED)
  FetchContent_Populate(shared_libraries)
endif()

# Create a Static Library for the Logger.
add_library(logger STATIC
  ${shared_libraries_SOURCE_DIR}/Loggers/logging.cpp
)

# Set Include Directories for the Logger.
target_include_directories(logger PUBLIC
  ${shared_libraries_SOURCE_DIR}/Loggers
)

# Add Compiler Definitions for the Logger Based on Platform.
if(WIN32)
  target_compile_definitions(logger PRIVATE _WIN32)
endif()

# Print Sucess Message. 
message(STATUS "Logger Library Configured Successfully.")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         BUILD PROJECT EXECUTABLE
# ------------------------------------------------------------------------------
# Build Executable for Project.
add_executable(${PROJECT_NAME}
  # Add 'main' Source File Here:
  "src/main.cpp"
  
  # Windows Resource File.
  # "resources/my_project_resources.rc"
)

# Link the Shared Libraries to this Project.
target_link_libraries(${PROJECT_NAME} PRIVATE
  logger  # Custom Logger from GitHub.
)

# Add Include Directories for Easier Includes in Source Files.
target_include_directories(${PROJECT_NAME} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  # Add Other Include Directories as Needed
)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                     BUILD CUSTOM PROJECT LIBRARY
# ------------------------------------------------------------------------------
# # Set Library Name.
# set(LIBRARY_NAME "common")

# # List Headers Explicitly.
# set(LIBRARY_HEADERS
#   "include/header_file.h"
# )

# # List Sources Explicitly.
# set(LIBRARY_SOURCES
#   "src/source_file.cpp"
# )

# # Create the Static Library.
# add_library(${LIBRARY_NAME} STATIC
#   ${LIBRARY_HEADERS}
#   ${LIBRARY_SOURCES}
# )

# # Include Directories for this Library.
# target_include_directories(${LIBRARY_NAME} PUBLIC
#   "${CMAKE_CURRENT_SOURCE_DIR}/include"
# )

# # Link Library Dependencies.
# target_link_libraries(${LIBRARY_NAME} PRIVATE
#   logger
# )

# # Link Custom Library to the Main Project.
# target_link_libraries(${PROJECT_NAME} PRIVATE
#   ${LIBRARY_NAME}
# )
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                    LINK SYSTEM LIBRARIES (PLATFORM-SPECIFIC)
# ------------------------------------------------------------------------------
# Link Platform-Specific System Libraries.
if(WIN32)
  # Windows-Specific Libraries.
  target_link_libraries(${PROJECT_NAME} PRIVATE
    ws2_32  # Winsock (Networking).
    winmm   # Windows Multimedia (for timeBeginPeriod, etc.).
  )
elseif(UNIX AND NOT APPLE)
  # Linux-Specific Libraries.
  target_link_libraries(${PROJECT_NAME} PRIVATE
    pthread  # POSIX Threads.
    dl       # Dynamic Linking.
  )
elseif(APPLE)
  # MacOS-Specific Libraries.
  target_link_libraries(${PROJECT_NAME} PRIVATE
    "-framework CoreFoundation"
  )
endif()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                    COPY FILES TO BUILD DIRECTORY
# ------------------------------------------------------------------------------
# # Copy a Single File.
# add_custom_command(
#   TARGET ${PROJECT_NAME}
#   POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E copy
#   ${CMAKE_CURRENT_SOURCE_DIR}/data/config.txt
#   $<TARGET_FILE_DIR:${PROJECT_NAME}>/config.txt
#   COMMENT "Copying config.txt to Build Directory."
# )

# # Copy an Entire Folder.
# add_custom_command(
#   TARGET ${PROJECT_NAME}
#   POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E copy_directory
#   ${CMAKE_CURRENT_SOURCE_DIR}/assets
#   $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets
#   COMMENT "Copying assets Folder to Build Directory."
# )

# # Extract a .zip File to the Build Directory.
# if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/archive.zip)
#   add_custom_command(
#     TARGET ${PROJECT_NAME}
#     POST_BUILD
#     # Extract the zip file.
#     COMMAND ${CMAKE_COMMAND} -E tar xf ${CMAKE_CURRENT_SOURCE_DIR}/data/archive.zip
#     WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>
#     COMMENT "Extracting archive.zip to Build Directory"
#   )
# endif()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                           ADD SUBDIRECTORIES
# ------------------------------------------------------------------------------
# # Include Subdirectory with its Oown CMakeLists.txt.
# add_subdirectory("external/third_party_lib")
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#                         BUILD SUMMARY
# # ------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== Build Configuration Complete ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Executable will be built to: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "To build, run: cmake --build .")
message(STATUS "")
# ------------------------------------------------------------------------------
