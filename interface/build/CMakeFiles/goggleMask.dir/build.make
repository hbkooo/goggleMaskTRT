# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hbk/project/goggleMaskRt/interface

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hbk/project/goggleMaskRt/interface/build

# Include any dependencies generated for this target.
include CMakeFiles/goggleMask.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/goggleMask.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/goggleMask.dir/flags.make

CMakeFiles/goggleMask.dir/main.cpp.o: CMakeFiles/goggleMask.dir/flags.make
CMakeFiles/goggleMask.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hbk/project/goggleMaskRt/interface/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/goggleMask.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/goggleMask.dir/main.cpp.o -c /home/hbk/project/goggleMaskRt/interface/main.cpp

CMakeFiles/goggleMask.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/goggleMask.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hbk/project/goggleMaskRt/interface/main.cpp > CMakeFiles/goggleMask.dir/main.cpp.i

CMakeFiles/goggleMask.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/goggleMask.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hbk/project/goggleMaskRt/interface/main.cpp -o CMakeFiles/goggleMask.dir/main.cpp.s

# Object files for target goggleMask
goggleMask_OBJECTS = \
"CMakeFiles/goggleMask.dir/main.cpp.o"

# External object files for target goggleMask
goggleMask_EXTERNAL_OBJECTS =

goggleMask: CMakeFiles/goggleMask.dir/main.cpp.o
goggleMask: CMakeFiles/goggleMask.dir/build.make
goggleMask: /usr/local/cuda/lib64/libcudart_static.a
goggleMask: /usr/lib/x86_64-linux-gnu/librt.so
goggleMask: /usr/local/lib/libopencv_videostab.so.3.1.0
goggleMask: /usr/local/lib/libopencv_superres.so.3.1.0
goggleMask: /usr/local/lib/libopencv_stitching.so.3.1.0
goggleMask: /usr/local/lib/libopencv_shape.so.3.1.0
goggleMask: /usr/local/lib/libopencv_photo.so.3.1.0
goggleMask: /usr/local/lib/libopencv_objdetect.so.3.1.0
goggleMask: /usr/local/lib/libopencv_calib3d.so.3.1.0
goggleMask: ../lib/libgogglemask.so
goggleMask: /usr/local/lib/libopencv_features2d.so.3.1.0
goggleMask: /usr/local/lib/libopencv_ml.so.3.1.0
goggleMask: /usr/local/lib/libopencv_highgui.so.3.1.0
goggleMask: /usr/local/lib/libopencv_videoio.so.3.1.0
goggleMask: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
goggleMask: /usr/local/lib/libopencv_flann.so.3.1.0
goggleMask: /usr/local/lib/libopencv_video.so.3.1.0
goggleMask: /usr/local/lib/libopencv_imgproc.so.3.1.0
goggleMask: /usr/local/lib/libopencv_core.so.3.1.0
goggleMask: CMakeFiles/goggleMask.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hbk/project/goggleMaskRt/interface/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable goggleMask"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/goggleMask.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/goggleMask.dir/build: goggleMask

.PHONY : CMakeFiles/goggleMask.dir/build

CMakeFiles/goggleMask.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/goggleMask.dir/cmake_clean.cmake
.PHONY : CMakeFiles/goggleMask.dir/clean

CMakeFiles/goggleMask.dir/depend:
	cd /home/hbk/project/goggleMaskRt/interface/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hbk/project/goggleMaskRt/interface /home/hbk/project/goggleMaskRt/interface /home/hbk/project/goggleMaskRt/interface/build /home/hbk/project/goggleMaskRt/interface/build /home/hbk/project/goggleMaskRt/interface/build/CMakeFiles/goggleMask.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/goggleMask.dir/depend

