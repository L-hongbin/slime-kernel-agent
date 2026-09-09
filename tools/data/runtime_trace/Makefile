# External NVBit release and all generated objects stay in local_artifacts.
NVBIT_ROOT ?= ../../../local_artifacts/component_tracking/nvbit_release_x86_64
BUILD_DIR ?= ../../../local_artifacts/component_tracking/build
NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_90
FLAGS = -I$(NVBIT_ROOT)/core -std=c++17 -arch=$(ARCH) -Xcompiler -fPIC
all: $(BUILD_DIR)/runtime_trace.so
$(BUILD_DIR):
	mkdir -p $@
$(BUILD_DIR)/inject_funcs.o: inject_funcs.cu common.h | $(BUILD_DIR)
	$(NVCC) $(FLAGS) -Xptxas -astoolspatch --keep-device-functions -c $< -o $@
$(BUILD_DIR)/runtime_trace.o: runtime_trace.cu common.h | $(BUILD_DIR)
	$(NVCC) $(FLAGS) -dc -Xptxas -cloning=no -c $< -o $@
$(BUILD_DIR)/runtime_trace.so: $(BUILD_DIR)/runtime_trace.o $(BUILD_DIR)/inject_funcs.o
	$(NVCC) $(FLAGS) -shared $^ -L$(NVBIT_ROOT)/core -lnvbit -lcuda -lcudart_static -o $@
