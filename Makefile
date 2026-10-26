THREADS ?= 1

default: convo.cpp
	g++ -O2 -fopenmp ./convo.cpp ./load_heat_matrix.cpp -o convo

run: default
	OMP_NUM_THREADS=$(THREADS) ./convo
