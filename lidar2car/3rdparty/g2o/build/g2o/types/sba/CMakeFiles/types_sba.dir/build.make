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
CMAKE_SOURCE_DIR = /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build

# Include any dependencies generated for this target.
include g2o/types/sba/CMakeFiles/types_sba.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include g2o/types/sba/CMakeFiles/types_sba.dir/compiler_depend.make

# Include the progress variables for this target.
include g2o/types/sba/CMakeFiles/types_sba.dir/progress.make

# Include the compile flags for this target's objects.
include g2o/types/sba/CMakeFiles/types_sba.dir/flags.make

g2o/types/sba/CMakeFiles/types_sba.dir/types_sba.cpp.o: g2o/types/sba/CMakeFiles/types_sba.dir/flags.make
g2o/types/sba/CMakeFiles/types_sba.dir/types_sba.cpp.o: ../g2o/types/sba/types_sba.cpp
g2o/types/sba/CMakeFiles/types_sba.dir/types_sba.cpp.o: g2o/types/sba/CMakeFiles/types_sba.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object g2o/types/sba/CMakeFiles/types_sba.dir/types_sba.cpp.o"
	cd /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/g2o/types/sba && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT g2o/types/sba/CMakeFiles/types_sba.dir/types_sba.cpp.o -MF CMakeFiles/types_sba.dir/types_sba.cpp.o.d -o CMakeFiles/types_sba.dir/types_sba.cpp.o -c /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/g2o/types/sba/types_sba.cpp

g2o/types/sba/CMakeFiles/types_sba.dir/types_sba.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/types_sba.dir/types_sba.cpp.i"
	cd /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/g2o/types/sba && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/g2o/types/sba/types_sba.cpp > CMakeFiles/types_sba.dir/types_sba.cpp.i

g2o/types/sba/CMakeFiles/types_sba.dir/types_sba.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/types_sba.dir/types_sba.cpp.s"
	cd /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/g2o/types/sba && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/g2o/types/sba/types_sba.cpp -o CMakeFiles/types_sba.dir/types_sba.cpp.s

g2o/types/sba/CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.o: g2o/types/sba/CMakeFiles/types_sba.dir/flags.make
g2o/types/sba/CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.o: ../g2o/types/sba/types_six_dof_expmap.cpp
g2o/types/sba/CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.o: g2o/types/sba/CMakeFiles/types_sba.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object g2o/types/sba/CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.o"
	cd /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/g2o/types/sba && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT g2o/types/sba/CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.o -MF CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.o.d -o CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.o -c /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/g2o/types/sba/types_six_dof_expmap.cpp

g2o/types/sba/CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.i"
	cd /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/g2o/types/sba && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/g2o/types/sba/types_six_dof_expmap.cpp > CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.i

g2o/types/sba/CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.s"
	cd /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/g2o/types/sba && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/g2o/types/sba/types_six_dof_expmap.cpp -o CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.s

# Object files for target types_sba
types_sba_OBJECTS = \
"CMakeFiles/types_sba.dir/types_sba.cpp.o" \
"CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.o"

# External object files for target types_sba
types_sba_EXTERNAL_OBJECTS =

../lib/libg2o_types_sba.so: g2o/types/sba/CMakeFiles/types_sba.dir/types_sba.cpp.o
../lib/libg2o_types_sba.so: g2o/types/sba/CMakeFiles/types_sba.dir/types_six_dof_expmap.cpp.o
../lib/libg2o_types_sba.so: g2o/types/sba/CMakeFiles/types_sba.dir/build.make
../lib/libg2o_types_sba.so: ../lib/libg2o_types_slam3d.so
../lib/libg2o_types_sba.so: ../lib/libg2o_core.so
../lib/libg2o_types_sba.so: ../lib/libg2o_stuff.so
../lib/libg2o_types_sba.so: ../lib/libg2o_opengl_helper.so
../lib/libg2o_types_sba.so: /usr/lib/x86_64-linux-gnu/libGLU.so
../lib/libg2o_types_sba.so: /usr/lib/x86_64-linux-gnu/libGL.so
../lib/libg2o_types_sba.so: /usr/lib/x86_64-linux-gnu/libGL.so
../lib/libg2o_types_sba.so: g2o/types/sba/CMakeFiles/types_sba.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library ../../../../lib/libg2o_types_sba.so"
	cd /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/g2o/types/sba && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/types_sba.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
g2o/types/sba/CMakeFiles/types_sba.dir/build: ../lib/libg2o_types_sba.so
.PHONY : g2o/types/sba/CMakeFiles/types_sba.dir/build

g2o/types/sba/CMakeFiles/types_sba.dir/clean:
	cd /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/g2o/types/sba && $(CMAKE_COMMAND) -P CMakeFiles/types_sba.dir/cmake_clean.cmake
.PHONY : g2o/types/sba/CMakeFiles/types_sba.dir/clean

g2o/types/sba/CMakeFiles/types_sba.dir/depend:
	cd /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/g2o/types/sba /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/g2o/types/sba /home/sensetime/ws/unified-senseauto/sensetime-main/LiDAR-SLAM/3rdparty/g2o/build/g2o/types/sba/CMakeFiles/types_sba.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : g2o/types/sba/CMakeFiles/types_sba.dir/depend
