# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.5/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.5/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/gflags

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/gflags/build

# Utility rule file for gflags_nothreads.

# Include any custom commands dependencies for this target.
include CMakeFiles/gflags_nothreads.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/gflags_nothreads.dir/progress.make

gflags_nothreads: CMakeFiles/gflags_nothreads.dir/build.make
.PHONY : gflags_nothreads

# Rule to build all files generated by this target.
CMakeFiles/gflags_nothreads.dir/build: gflags_nothreads
.PHONY : CMakeFiles/gflags_nothreads.dir/build

CMakeFiles/gflags_nothreads.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gflags_nothreads.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gflags_nothreads.dir/clean

CMakeFiles/gflags_nothreads.dir/depend:
	cd /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/gflags/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/gflags /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/gflags /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/gflags/build /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/gflags/build /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/gflags/build/CMakeFiles/gflags_nothreads.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gflags_nothreads.dir/depend

