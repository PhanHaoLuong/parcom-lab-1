# ParCom Lab 1 -- 2D convolution (heat diffusion) with OpenMP.
#
# Linux first, because the course requires it. The same file also works on
# macOS and under MSYS2. Nothing here needs CUDA: GPU offloading is opt-in via
# GPU=1 and uses only the compiler's own nvptx backend.
#
#   make                 build build/convo (CPU only)
#   make run             build, then run 100 iterations
#   make bench           full sweep -> results/ (see run_bench.sh)
#   make clean
#
# Options:
#   GPU=1                enable OpenMP target offloading (+ PARCOM_GPU)
#   CUDA_ARCH=sm_75      nvptx target; default "native" for the local GPU
#   NATIVE=0             drop -march=native to get a portable binary
#   ARGS="--iters 200"   extra arguments for `make run`

CXX       ?= g++
BUILD_DIR := build
TARGET    := $(BUILD_DIR)/convo

SRCS := src/main.cpp src/heat_matrix.cpp src/gpu_offload.cpp
OBJS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

OPT      ?= -O3
NATIVE   ?= 1
GPU      ?= 0
CUDA_ARCH ?= native

CPPFLAGS += -Isrc -MMD -MP
CXXFLAGS += -std=c++17 -fopenmp -Wall -Wextra -Wno-unused-parameter $(OPT)
LDFLAGS  += -fopenmp

ifeq ($(NATIVE),1)
CXXFLAGS += -march=native
endif

ifeq ($(GPU),1)
CXXFLAGS += -DPARCOM_GPU -foffload=nvptx-none
ifneq ($(CUDA_ARCH),native)
CXXFLAGS += -foffload-options=nvptx-none=-misa=$(CUDA_ARCH)
endif
endif

.PHONY: all clean run bench dirs

all: $(TARGET)

$(TARGET): $(OBJS) | dirs
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LDLIBS)
	@echo "built $@"

$(BUILD_DIR)/%.o: src/%.cpp | dirs
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

dirs:
	@mkdir -p $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET) $(ARGS)

bench: $(TARGET)
	./run_bench.sh $(ARGS)

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
