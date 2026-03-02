NVCC ?= nvcc
ARCH ?= sm_120
NVCCFLAGS ?= -O3 -std=c++17
KERNELS := vector_add
BINARIES := $(addprefix build/,$(KERNELS))

.PHONY: all check matmul
all: $(BINARIES)

build:
	mkdir -p $@

define kernel_rule
build/$(1): $(1)/$(1).cu common/check.cuh | build
	$$(NVCC) $$(NVCCFLAGS) -arch=$$(ARCH) $$< -o $$@
endef
$(foreach kernel,$(KERNELS),$(eval $(call kernel_rule,$(kernel))))

matmul: build/matmul

build/matmul: matmul/matmul.cu | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@ -lcublas

check: all
	@set -e; for kernel in $(BINARIES); do ./$$kernel; done
