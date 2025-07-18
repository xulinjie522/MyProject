# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nvidia/Desktop/xulinjie/yolop-yolo11

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/Desktop/xulinjie/yolop-yolo11/build

# Include any dependencies generated for this target.
include yolo11/CMakeFiles/yolo11_det.dir/depend.make

# Include the progress variables for this target.
include yolo11/CMakeFiles/yolo11_det.dir/progress.make

# Include the compile flags for this target's objects.
include yolo11/CMakeFiles/yolo11_det.dir/flags.make

yolo11/CMakeFiles/yolo11_det.dir/yolo11_det.cpp.o: yolo11/CMakeFiles/yolo11_det.dir/flags.make
yolo11/CMakeFiles/yolo11_det.dir/yolo11_det.cpp.o: ../yolo11/yolo11_det.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/xulinjie/yolop-yolo11/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object yolo11/CMakeFiles/yolo11_det.dir/yolo11_det.cpp.o"
	cd /home/nvidia/Desktop/xulinjie/yolop-yolo11/build/yolo11 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolo11_det.dir/yolo11_det.cpp.o -c /home/nvidia/Desktop/xulinjie/yolop-yolo11/yolo11/yolo11_det.cpp

yolo11/CMakeFiles/yolo11_det.dir/yolo11_det.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo11_det.dir/yolo11_det.cpp.i"
	cd /home/nvidia/Desktop/xulinjie/yolop-yolo11/build/yolo11 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/xulinjie/yolop-yolo11/yolo11/yolo11_det.cpp > CMakeFiles/yolo11_det.dir/yolo11_det.cpp.i

yolo11/CMakeFiles/yolo11_det.dir/yolo11_det.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo11_det.dir/yolo11_det.cpp.s"
	cd /home/nvidia/Desktop/xulinjie/yolop-yolo11/build/yolo11 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/xulinjie/yolop-yolo11/yolo11/yolo11_det.cpp -o CMakeFiles/yolo11_det.dir/yolo11_det.cpp.s

# Object files for target yolo11_det
yolo11_det_OBJECTS = \
"CMakeFiles/yolo11_det.dir/yolo11_det.cpp.o"

# External object files for target yolo11_det
yolo11_det_EXTERNAL_OBJECTS =

yolo11/yolo11_det: yolo11/CMakeFiles/yolo11_det.dir/yolo11_det.cpp.o
yolo11/yolo11_det: yolo11/CMakeFiles/yolo11_det.dir/build.make
yolo11/yolo11_det: yolo11/libyolo11_lib.a
yolo11/yolo11_det: yolo11/libyolo11_plugins.so
yolo11/yolo11_det: /usr/local/cuda-11.4/lib64/libcudart_static.a
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/librt.so
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5.4
yolo11/yolo11_det: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5.4
yolo11/yolo11_det: yolo11/CMakeFiles/yolo11_det.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/Desktop/xulinjie/yolop-yolo11/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable yolo11_det"
	cd /home/nvidia/Desktop/xulinjie/yolop-yolo11/build/yolo11 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolo11_det.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
yolo11/CMakeFiles/yolo11_det.dir/build: yolo11/yolo11_det

.PHONY : yolo11/CMakeFiles/yolo11_det.dir/build

yolo11/CMakeFiles/yolo11_det.dir/clean:
	cd /home/nvidia/Desktop/xulinjie/yolop-yolo11/build/yolo11 && $(CMAKE_COMMAND) -P CMakeFiles/yolo11_det.dir/cmake_clean.cmake
.PHONY : yolo11/CMakeFiles/yolo11_det.dir/clean

yolo11/CMakeFiles/yolo11_det.dir/depend:
	cd /home/nvidia/Desktop/xulinjie/yolop-yolo11/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/Desktop/xulinjie/yolop-yolo11 /home/nvidia/Desktop/xulinjie/yolop-yolo11/yolo11 /home/nvidia/Desktop/xulinjie/yolop-yolo11/build /home/nvidia/Desktop/xulinjie/yolop-yolo11/build/yolo11 /home/nvidia/Desktop/xulinjie/yolop-yolo11/build/yolo11/CMakeFiles/yolo11_det.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : yolo11/CMakeFiles/yolo11_det.dir/depend

