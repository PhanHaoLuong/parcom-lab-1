THREADS ?= 1
N ?= 1

CXX = g++

CXXFLAGS = -O3 -fopenmp \
           -fcf-protection=none \
           -fno-stack-protector \
           -foffload=nvptx-none \
           -foffload-options=nvptx-none=-misa=sm_80
					 -march=native

default:
	$(CXX) $(CXXFLAGS) ./convo.cpp ./load_heat_matrix.cpp -o convo

run: default
	OMP_TARGET_OFFLOAD=MANDATORY OMP_NUM_THREADS=$(THREADS) ./convo $(N)

clean:
	rm -f convo

.PHONY: default run clean
