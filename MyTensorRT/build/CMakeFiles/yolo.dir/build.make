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
CMAKE_SOURCE_DIR = /home/nvidia/Desktop/xulinjie

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/Desktop/xulinjie/build

# Include any dependencies generated for this target.
include CMakeFiles/yolo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/yolo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolo.dir/flags.make

CMakeFiles/yolo.dir/src/main.cpp.o: CMakeFiles/yolo.dir/flags.make
CMakeFiles/yolo.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/xulinjie/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolo.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolo.dir/src/main.cpp.o -c /home/nvidia/Desktop/xulinjie/src/main.cpp

CMakeFiles/yolo.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/xulinjie/src/main.cpp > CMakeFiles/yolo.dir/src/main.cpp.i

CMakeFiles/yolo.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/xulinjie/src/main.cpp -o CMakeFiles/yolo.dir/src/main.cpp.s

CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.o: CMakeFiles/yolo.dir/flags.make
CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.o: ../src/BaseTRTInfer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/xulinjie/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.o -c /home/nvidia/Desktop/xulinjie/src/BaseTRTInfer.cpp

CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/xulinjie/src/BaseTRTInfer.cpp > CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.i

CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/xulinjie/src/BaseTRTInfer.cpp -o CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.s

CMakeFiles/yolo.dir/src/YOLO.cpp.o: CMakeFiles/yolo.dir/flags.make
CMakeFiles/yolo.dir/src/YOLO.cpp.o: ../src/YOLO.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/xulinjie/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/yolo.dir/src/YOLO.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolo.dir/src/YOLO.cpp.o -c /home/nvidia/Desktop/xulinjie/src/YOLO.cpp

CMakeFiles/yolo.dir/src/YOLO.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo.dir/src/YOLO.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/xulinjie/src/YOLO.cpp > CMakeFiles/yolo.dir/src/YOLO.cpp.i

CMakeFiles/yolo.dir/src/YOLO.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo.dir/src/YOLO.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/xulinjie/src/YOLO.cpp -o CMakeFiles/yolo.dir/src/YOLO.cpp.s

CMakeFiles/yolo.dir/src/YOLOInfer.cpp.o: CMakeFiles/yolo.dir/flags.make
CMakeFiles/yolo.dir/src/YOLOInfer.cpp.o: ../src/YOLOInfer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/xulinjie/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/yolo.dir/src/YOLOInfer.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolo.dir/src/YOLOInfer.cpp.o -c /home/nvidia/Desktop/xulinjie/src/YOLOInfer.cpp

CMakeFiles/yolo.dir/src/YOLOInfer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo.dir/src/YOLOInfer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/xulinjie/src/YOLOInfer.cpp > CMakeFiles/yolo.dir/src/YOLOInfer.cpp.i

CMakeFiles/yolo.dir/src/YOLOInfer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo.dir/src/YOLOInfer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/xulinjie/src/YOLOInfer.cpp -o CMakeFiles/yolo.dir/src/YOLOInfer.cpp.s

CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.o: CMakeFiles/yolo.dir/flags.make
CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.o: ../src/YOLOPostprocess.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/xulinjie/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.o -c /home/nvidia/Desktop/xulinjie/src/YOLOPostprocess.cpp

CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/xulinjie/src/YOLOPostprocess.cpp > CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.i

CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/xulinjie/src/YOLOPostprocess.cpp -o CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.s

CMakeFiles/yolo.dir/src/YOLOPostprocess.cu.o: CMakeFiles/yolo.dir/flags.make
CMakeFiles/yolo.dir/src/YOLOPostprocess.cu.o: ../src/YOLOPostprocess.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/xulinjie/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object CMakeFiles/yolo.dir/src/YOLOPostprocess.cu.o"
	/usr/local/cuda-11.4/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/nvidia/Desktop/xulinjie/src/YOLOPostprocess.cu -o CMakeFiles/yolo.dir/src/YOLOPostprocess.cu.o

CMakeFiles/yolo.dir/src/YOLOPostprocess.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/yolo.dir/src/YOLOPostprocess.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/yolo.dir/src/YOLOPostprocess.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/yolo.dir/src/YOLOPostprocess.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.o: CMakeFiles/yolo.dir/flags.make
CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.o: ../src/YOLOPreprocess.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/xulinjie/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.o -c /home/nvidia/Desktop/xulinjie/src/YOLOPreprocess.cpp

CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/xulinjie/src/YOLOPreprocess.cpp > CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.i

CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/xulinjie/src/YOLOPreprocess.cpp -o CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.s

CMakeFiles/yolo.dir/src/YOLOPreprocess.cu.o: CMakeFiles/yolo.dir/flags.make
CMakeFiles/yolo.dir/src/YOLOPreprocess.cu.o: ../src/YOLOPreprocess.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/xulinjie/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CUDA object CMakeFiles/yolo.dir/src/YOLOPreprocess.cu.o"
	/usr/local/cuda-11.4/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/nvidia/Desktop/xulinjie/src/YOLOPreprocess.cu -o CMakeFiles/yolo.dir/src/YOLOPreprocess.cu.o

CMakeFiles/yolo.dir/src/YOLOPreprocess.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/yolo.dir/src/YOLOPreprocess.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/yolo.dir/src/YOLOPreprocess.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/yolo.dir/src/YOLOPreprocess.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/yolo.dir/src/IModel.cpp.o: CMakeFiles/yolo.dir/flags.make
CMakeFiles/yolo.dir/src/IModel.cpp.o: ../src/IModel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/xulinjie/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/yolo.dir/src/IModel.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolo.dir/src/IModel.cpp.o -c /home/nvidia/Desktop/xulinjie/src/IModel.cpp

CMakeFiles/yolo.dir/src/IModel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo.dir/src/IModel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/xulinjie/src/IModel.cpp > CMakeFiles/yolo.dir/src/IModel.cpp.i

CMakeFiles/yolo.dir/src/IModel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo.dir/src/IModel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/xulinjie/src/IModel.cpp -o CMakeFiles/yolo.dir/src/IModel.cpp.s

# Object files for target yolo
yolo_OBJECTS = \
"CMakeFiles/yolo.dir/src/main.cpp.o" \
"CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.o" \
"CMakeFiles/yolo.dir/src/YOLO.cpp.o" \
"CMakeFiles/yolo.dir/src/YOLOInfer.cpp.o" \
"CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.o" \
"CMakeFiles/yolo.dir/src/YOLOPostprocess.cu.o" \
"CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.o" \
"CMakeFiles/yolo.dir/src/YOLOPreprocess.cu.o" \
"CMakeFiles/yolo.dir/src/IModel.cpp.o"

# External object files for target yolo
yolo_EXTERNAL_OBJECTS =

CMakeFiles/yolo.dir/cmake_device_link.o: CMakeFiles/yolo.dir/src/main.cpp.o
CMakeFiles/yolo.dir/cmake_device_link.o: CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.o
CMakeFiles/yolo.dir/cmake_device_link.o: CMakeFiles/yolo.dir/src/YOLO.cpp.o
CMakeFiles/yolo.dir/cmake_device_link.o: CMakeFiles/yolo.dir/src/YOLOInfer.cpp.o
CMakeFiles/yolo.dir/cmake_device_link.o: CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.o
CMakeFiles/yolo.dir/cmake_device_link.o: CMakeFiles/yolo.dir/src/YOLOPostprocess.cu.o
CMakeFiles/yolo.dir/cmake_device_link.o: CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.o
CMakeFiles/yolo.dir/cmake_device_link.o: CMakeFiles/yolo.dir/src/YOLOPreprocess.cu.o
CMakeFiles/yolo.dir/cmake_device_link.o: CMakeFiles/yolo.dir/src/IModel.cpp.o
CMakeFiles/yolo.dir/cmake_device_link.o: CMakeFiles/yolo.dir/build.make
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/local/cuda-11.4/lib64/libcudart_static.a
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/librt.so
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/local/cuda-11.4/lib64/libcublas.so
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5.4
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/librt.so
CMakeFiles/yolo.dir/cmake_device_link.o: /usr/local/cuda-11.4/lib64/libcublas.so
CMakeFiles/yolo.dir/cmake_device_link.o: CMakeFiles/yolo.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/Desktop/xulinjie/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CUDA device code CMakeFiles/yolo.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolo.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolo.dir/build: CMakeFiles/yolo.dir/cmake_device_link.o

.PHONY : CMakeFiles/yolo.dir/build

# Object files for target yolo
yolo_OBJECTS = \
"CMakeFiles/yolo.dir/src/main.cpp.o" \
"CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.o" \
"CMakeFiles/yolo.dir/src/YOLO.cpp.o" \
"CMakeFiles/yolo.dir/src/YOLOInfer.cpp.o" \
"CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.o" \
"CMakeFiles/yolo.dir/src/YOLOPostprocess.cu.o" \
"CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.o" \
"CMakeFiles/yolo.dir/src/YOLOPreprocess.cu.o" \
"CMakeFiles/yolo.dir/src/IModel.cpp.o"

# External object files for target yolo
yolo_EXTERNAL_OBJECTS =

../bin/yolo: CMakeFiles/yolo.dir/src/main.cpp.o
../bin/yolo: CMakeFiles/yolo.dir/src/BaseTRTInfer.cpp.o
../bin/yolo: CMakeFiles/yolo.dir/src/YOLO.cpp.o
../bin/yolo: CMakeFiles/yolo.dir/src/YOLOInfer.cpp.o
../bin/yolo: CMakeFiles/yolo.dir/src/YOLOPostprocess.cpp.o
../bin/yolo: CMakeFiles/yolo.dir/src/YOLOPostprocess.cu.o
../bin/yolo: CMakeFiles/yolo.dir/src/YOLOPreprocess.cpp.o
../bin/yolo: CMakeFiles/yolo.dir/src/YOLOPreprocess.cu.o
../bin/yolo: CMakeFiles/yolo.dir/src/IModel.cpp.o
../bin/yolo: CMakeFiles/yolo.dir/build.make
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.5.4
../bin/yolo: /usr/local/cuda-11.4/lib64/libcudart_static.a
../bin/yolo: /usr/lib/aarch64-linux-gnu/librt.so
../bin/yolo: /usr/local/cuda-11.4/lib64/libcublas.so
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5.4
../bin/yolo: /usr/lib/aarch64-linux-gnu/librt.so
../bin/yolo: /usr/local/cuda-11.4/lib64/libcublas.so
../bin/yolo: CMakeFiles/yolo.dir/cmake_device_link.o
../bin/yolo: CMakeFiles/yolo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/Desktop/xulinjie/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX executable ../bin/yolo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolo.dir/build: ../bin/yolo

.PHONY : CMakeFiles/yolo.dir/build

CMakeFiles/yolo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolo.dir/clean

CMakeFiles/yolo.dir/depend:
	cd /home/nvidia/Desktop/xulinjie/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/Desktop/xulinjie /home/nvidia/Desktop/xulinjie /home/nvidia/Desktop/xulinjie/build /home/nvidia/Desktop/xulinjie/build /home/nvidia/Desktop/xulinjie/build/CMakeFiles/yolo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yolo.dir/depend

