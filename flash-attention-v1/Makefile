NVCC ?= nvcc
ARCH ?= sm_120
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo

.PHONY: all check

all: flash-attention test

flash-attention: flash-attention.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

test: test.cu flash-attention.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

benchmark: benchmark.cu flash-attention.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

check: test
	./test
