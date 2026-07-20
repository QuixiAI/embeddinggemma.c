# embeddinggemma.c CPU, Metal, CUDA, ROCm, and XPU SYCL build
CC      ?= cc
CXX     ?= c++
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Werror -g
LDLIBS  ?= -lm -pthread
NVCC    ?= nvcc
CUDA_HOME ?= /usr/local/cuda
NVCCFLAGS ?= -std=c++17 -O3 --use_fast_math -lineinfo
# Build native code for every architecture supported by the installed compiler,
# plus PTX for forward compatibility. Set CUDA_ARCHS=86 for a faster local build.
ifdef CUDA_ARCH
CUDA_ARCHS ?= $(CUDA_ARCH)
else
CUDA_ARCHS ?= $(patsubst sm_%,%,$(shell $(NVCC) --list-gpu-code 2>/dev/null))
endif
CUDA_PTX_ARCH ?= $(lastword $(CUDA_ARCHS))
CUDA_GENCODE := $(foreach arch,$(CUDA_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch)) \
	-gencode arch=compute_$(CUDA_PTX_ARCH),code=compute_$(CUDA_PTX_ARCH)
CUDA_LDLIBS := -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcublas -lcudart
HIPCC ?= hipcc
ROCM_HOME ?= /opt/rocm
HIPFLAGS ?= -std=c++17 -O3 -ffast-math
# Build one portable Instinct binary by default. These are the production LLVM
# targets for CDNA1 through CDNA4; ROCM_ARCH remains a developer-only override.
ROCM_CDNA_ARCHS ?= gfx908 gfx90a gfx942 gfx950
ifdef ROCM_ARCH
ROCM_ARCHS ?= $(ROCM_ARCH)
else
ROCM_ARCHS ?= $(ROCM_CDNA_ARCHS)
endif
ROCM_GENCODE := $(foreach arch,$(ROCM_ARCHS),--offload-arch=$(arch))
ROCM_LDLIBS := -L$(ROCM_HOME)/lib -Wl,-rpath,$(ROCM_HOME)/lib -lhipblas -lamdhip64
SYCL_CXX ?= icpx
SYCLFLAGS ?= -std=c++17 -O3 -fsycl -ffast-math
SYCL_TARGETS ?=
SYCL_LINK_TARGET_FLAGS :=
ifneq ($(strip $(SYCL_TARGETS)),)
SYCLFLAGS += -fsycl-targets=$(SYCL_TARGETS)
SYCL_LINK_TARGET_FLAGS += -fsycl-targets=$(SYCL_TARGETS)
endif
SYCL_LDLIBS ?= -fsycl -qmkl
BUILD   := build
DIST    ?= dist
MODEL   ?= model/embeddinggemma-300M-qat-Q4_0.gguf
VERSION := $(strip $(shell cat VERSION 2>/dev/null))

# Prefer a full Xcode installation for Metal without requiring a global
# xcode-select change. An environment-provided DEVELOPER_DIR still wins.
ifeq ($(origin DEVELOPER_DIR), undefined)
ifneq ($(wildcard /Applications/Xcode.app/Contents/Developer),)
DEVELOPER_DIR := /Applications/Xcode.app/Contents/Developer
export DEVELOPER_DIR
endif
endif

# QuixiCore-Metal-style build: compile every Metal translation unit into one
# metallib, then embed it into each Metal executable as a Mach-O section.
METAL_FLAGS ?= -std=metal3.1 -O2 -Wall -Wextra -fno-fast-math
METAL_INCLUDE_DIR := src/metal/include
METAL_SOURCES := \
	src/metal/kernels/embedding.metal \
	src/metal/kernels/qgemv.metal \
	src/metal/kernels/norm.metal \
	src/metal/kernels/rope.metal \
	src/metal/kernels/attention.metal \
	src/metal/kernels/elementwise.metal \
	src/metal/kernels/pool.metal
METAL_HEADERS := $(wildcard $(METAL_INCLUDE_DIR)/*.metal)
METALLIB := $(BUILD)/embeddinggemma.metallib
METALLIB_LDFLAGS := -Wl,-sectcreate,__DATA,__metallib,$(abspath $(METALLIB))
OBJC ?= clang
OBJCFLAGS ?= -O2 -Wall -Wextra -Werror -fobjc-arc
METAL_FRAMEWORKS := -framework Foundation -framework Metal

SRCS_CORE := src/gguf.c
SRCS_MODEL := $(SRCS_CORE) src/model.c
SRCS_TOKENIZER := $(SRCS_MODEL) src/tokenizer.c
SRCS_ENGINE := $(SRCS_TOKENIZER) src/quants.c src/kernels.c src/parallel.c src/engine.c
SRCS_SERVICE := src/inference_service.c
SRCS_SERVER := src/server.c src/response_cache.c

CUDA_CORE_OBJS := $(patsubst src/%.c,$(BUILD)/cuda/%.o,$(SRCS_ENGINE))
CUDA_SERVICE_OBJS := $(patsubst src/%.c,$(BUILD)/cuda/%.o,$(SRCS_SERVICE))
CUDA_SERVER_OBJS := $(patsubst src/%.c,$(BUILD)/cuda/%.o,$(SRCS_SERVER))
CUDA_ENGINE_OBJ := $(BUILD)/cuda/engine_cuda.o
ROCM_CORE_OBJS := $(patsubst src/%.c,$(BUILD)/rocm/%.o,$(SRCS_ENGINE))
ROCM_SERVICE_OBJS := $(patsubst src/%.c,$(BUILD)/rocm/%.o,$(SRCS_SERVICE))
ROCM_SERVER_OBJS := $(patsubst src/%.c,$(BUILD)/rocm/%.o,$(SRCS_SERVER))
ROCM_ENGINE_OBJ := $(BUILD)/rocm/engine_rocm.o
XPU_CORE_OBJS := $(patsubst src/%.c,$(BUILD)/xpu/%.o,$(SRCS_ENGINE))
XPU_SERVICE_OBJS := $(patsubst src/%.c,$(BUILD)/xpu/%.o,$(SRCS_SERVICE))
XPU_SERVER_OBJS := $(patsubst src/%.c,$(BUILD)/xpu/%.o,$(SRCS_SERVER))
XPU_XE2_FLASH ?= 0
XPU_ONEDNN ?= 0
XPU_DEPS_DIR ?= $(CURDIR)/.xpu-deps
ifeq ($(origin VLLM_XPU_KERNELS),undefined)
VLLM_XPU_KERNELS := $(XPU_DEPS_DIR)/vllm-xpu-kernels
CUTLASS_SYCL ?= $(XPU_DEPS_DIR)/sycl-tla
XPU_MANAGED_DEPS := 1
else
CUTLASS_SYCL ?= $(VLLM_XPU_KERNELS)/.deps/cutlass-sycl-src
XPU_MANAGED_DEPS := 0
endif
XPU_DEPS_STAMP := $(XPU_DEPS_DIR)/.pinned-xe2-deps
XPU_DEPS_PREREQ := $(if $(filter 1,$(XPU_MANAGED_DEPS)),$(XPU_DEPS_STAMP))
XPU_ENGINE_DEFS :=
XPU_ENGINE_TARGET_FLAGS :=
XPU_EXTRA_OBJS :=
XPU_LINK_FLAGS := $(SYCL_LINK_TARGET_FLAGS)
ifeq ($(XPU_XE2_FLASH),1)
XPU_ENGINE_DEFS += -DEI_XPU_XE2_FLASH -DEI_XPU_XE2_W4
XPU_ENGINE_TARGET_FLAGS += -fsycl-targets=spir64_gen
XPU_EXTRA_OBJS += $(BUILD)/xpu/engine_xpu_flash.o $(BUILD)/xpu/engine_xpu_w4.o
XPU_FLASH_FLAGS := -DEI_XPU_XE2_FLASH -DEI_XPU_XE2_W4 -DVLLM_XPU_ENABLE_XE2 \
	-fsycl-targets=spir64_gen -DCUTLASS_ENABLE_HEADERS_ONLY \
	-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED \
	-fno-sycl-instrument-device-code -ftemplate-backtrace-limit=0 \
	-include $(VLLM_XPU_KERNELS)/csrc/sycl_first.h \
	-I$(VLLM_XPU_KERNELS)/csrc \
	-I$(VLLM_XPU_KERNELS)/csrc/xpu/attn/xe_2 \
	-I$(VLLM_XPU_KERNELS) -I$(CUTLASS_SYCL)/include \
	-I$(CUTLASS_SYCL)/tools/util/include -I$(CUTLASS_SYCL)/applications
XPU_LINK_FLAGS += -fsycl-max-parallel-link-jobs=16 -flink-huge-device-code \
	-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
	-fsycl-targets=spir64_gen \
	-Xsycl-target-backend=spir64_gen \
	"-device pvc,bmg,bmg-g21-a0,bmg-g31-a0 -internal_options -cl-intel-256-GRF-per-thread" \
	-lze_loader -Wno-unused-command-line-argument
endif
ifeq ($(XPU_ONEDNN),1)
XPU_ENGINE_DEFS += -DEI_XPU_ONEDNN
XPU_EXTRA_OBJS += $(BUILD)/xpu/engine_xpu_onednn.o
XPU_LINK_FLAGS += -ldnnl
endif
XPU_ENGINE_VARIANT := $(if $(filter 1,$(XPU_XE2_FLASH)),-xe2)$(if $(filter 1,$(XPU_ONEDNN)),-onednn)
XPU_ENGINE_OBJ := $(BUILD)/xpu/engine_xpu$(XPU_ENGINE_VARIANT).o
XPU_ENGINE_OBJS := $(XPU_ENGINE_OBJ) $(XPU_EXTRA_OBJS)
XPU_TEST_OBJ := $(BUILD)/xpu/test_xpu$(XPU_ENGINE_VARIANT).o

.PHONY: all FORCE_XPU_LINK xpu-deps
all: $(BUILD)/embeddinggemma

FORCE_XPU_LINK:

xpu-deps: $(XPU_DEPS_STAMP)

$(XPU_DEPS_STAMP): scripts/fetch-xpu-deps.sh
	./scripts/fetch-xpu-deps.sh $(XPU_DEPS_DIR)
	@touch $@

$(BUILD)/test_gguf: src/test_gguf.c $(SRCS_CORE) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/test_gguf.c $(SRCS_CORE) $(LDLIBS)

$(BUILD)/test_tokenizer: src/test_tokenizer.c $(SRCS_TOKENIZER) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/test_tokenizer.c $(SRCS_TOKENIZER) $(LDLIBS)

$(BUILD)/test_embed: src/test_embed.c $(SRCS_ENGINE) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/test_embed.c $(SRCS_ENGINE) $(LDLIBS)

$(BUILD)/test_batch: src/test_batch.c $(SRCS_ENGINE) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/test_batch.c $(SRCS_ENGINE) $(LDLIBS)

$(BUILD)/test_kernels: src/test_kernels.c src/quants.c src/kernels.c src/gguf.c src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/test_kernels.c src/quants.c src/kernels.c src/gguf.c $(LDLIBS)

$(BUILD)/embeddinggemma: $(SRCS_SERVER) $(SRCS_ENGINE) $(SRCS_SERVICE) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(SRCS_SERVER) $(SRCS_ENGINE) $(SRCS_SERVICE) $(LDLIBS)

$(BUILD)/test_inference_service: src/test_inference_service.c $(SRCS_SERVICE) src/tokenizer.c src/gguf.c src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/test_inference_service.c $(SRCS_SERVICE) src/tokenizer.c src/gguf.c $(LDLIBS)

$(BUILD)/test_response_cache: src/test_response_cache.c src/response_cache.c src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/test_response_cache.c src/response_cache.c src/gguf.c $(LDLIBS)

$(BUILD)/cuda/%.o: src/%.c src/*.h | $(BUILD)/cuda
	$(CC) $(CFLAGS) -DEI_ENABLE_CUDA -Isrc -c -o $@ $<

$(CUDA_ENGINE_OBJ): src/engine_cuda.cu src/engine_cuda.h src/model.h src/gguf.h | $(BUILD)/cuda
	$(NVCC) $(NVCCFLAGS) $(CUDA_GENCODE) -Xptxas=-warn-spills -Isrc -c -o $@ $<

$(BUILD)/cuda/test_embed.o: src/test_embed.c src/*.h | $(BUILD)/cuda
	$(CC) $(CFLAGS) -DEI_ENABLE_CUDA -Isrc -c -o $@ $<

$(BUILD)/cuda/test_batch.o: src/test_batch.c src/*.h | $(BUILD)/cuda
	$(CC) $(CFLAGS) -DEI_ENABLE_CUDA -Isrc -c -o $@ $<

$(BUILD)/cuda/test_cuda.o: src/test_cuda.c src/*.h | $(BUILD)/cuda
	$(CC) $(CFLAGS) -DEI_ENABLE_CUDA -Isrc -c -o $@ $<

$(BUILD)/cuda/perf_engine.o: perf/harness/bench_engine.c src/*.h | $(BUILD)/cuda
	$(CC) $(CFLAGS) -DEI_ENABLE_CUDA -Isrc -c -o $@ $<

$(BUILD)/cuda/perf_batch.o: perf/harness/bench_batch.c src/*.h | $(BUILD)/cuda
	$(CC) $(CFLAGS) -DEI_ENABLE_CUDA -Isrc -c -o $@ $<

$(BUILD)/cuda/perf_concurrency.o: perf/harness/bench_concurrency.c src/*.h | $(BUILD)/cuda
	$(CC) $(CFLAGS) -DEI_ENABLE_CUDA -Isrc -c -o $@ $<

$(BUILD)/embeddinggemma-cuda: $(CUDA_SERVER_OBJS) $(CUDA_CORE_OBJS) $(CUDA_SERVICE_OBJS) $(CUDA_ENGINE_OBJ) | $(BUILD)
	$(CXX) -o $@ $(CUDA_SERVER_OBJS) $(CUDA_CORE_OBJS) $(CUDA_SERVICE_OBJS) \
		$(CUDA_ENGINE_OBJ) $(LDLIBS) $(CUDA_LDLIBS)

$(BUILD)/test_embed_cuda: $(BUILD)/cuda/test_embed.o $(CUDA_CORE_OBJS) $(CUDA_ENGINE_OBJ) | $(BUILD)
	$(CXX) -o $@ $^ $(LDLIBS) $(CUDA_LDLIBS)

$(BUILD)/test_batch_cuda: $(BUILD)/cuda/test_batch.o $(CUDA_CORE_OBJS) $(CUDA_ENGINE_OBJ) | $(BUILD)
	$(CXX) -o $@ $^ $(LDLIBS) $(CUDA_LDLIBS)

$(BUILD)/test_cuda: $(BUILD)/cuda/test_cuda.o $(CUDA_CORE_OBJS) $(CUDA_ENGINE_OBJ) | $(BUILD)
	$(CXX) -o $@ $^ $(LDLIBS) $(CUDA_LDLIBS)

$(BUILD)/perf_engine_cuda: $(BUILD)/cuda/perf_engine.o $(CUDA_CORE_OBJS) $(CUDA_ENGINE_OBJ) | $(BUILD)
	$(CXX) -o $@ $^ $(LDLIBS) $(CUDA_LDLIBS)

$(BUILD)/perf_batch_cuda: $(BUILD)/cuda/perf_batch.o $(CUDA_CORE_OBJS) $(CUDA_ENGINE_OBJ) | $(BUILD)
	$(CXX) -o $@ $^ $(LDLIBS) $(CUDA_LDLIBS)

$(BUILD)/perf_concurrency_cuda: $(BUILD)/cuda/perf_concurrency.o $(CUDA_CORE_OBJS) $(CUDA_SERVICE_OBJS) $(CUDA_ENGINE_OBJ) | $(BUILD)
	$(CXX) -o $@ $^ $(LDLIBS) $(CUDA_LDLIBS)

$(BUILD)/rocm/%.o: src/%.c src/*.h | $(BUILD)/rocm
	$(CC) $(CFLAGS) -DEI_ENABLE_ROCM -Isrc -c -o $@ $<

$(ROCM_ENGINE_OBJ): src/engine_rocm.hip src/engine_rocm.h src/model.h src/gguf.h | $(BUILD)/rocm
	$(HIPCC) $(HIPFLAGS) $(ROCM_GENCODE) -Isrc -c -o $@ $<

$(BUILD)/rocm/test_embed.o: src/test_embed.c src/*.h | $(BUILD)/rocm
	$(CC) $(CFLAGS) -DEI_ENABLE_ROCM -Isrc -c -o $@ $<

$(BUILD)/rocm/test_batch.o: src/test_batch.c src/*.h | $(BUILD)/rocm
	$(CC) $(CFLAGS) -DEI_ENABLE_ROCM -Isrc -c -o $@ $<

$(BUILD)/rocm/test_rocm.o: src/test_rocm.c src/*.h | $(BUILD)/rocm
	$(CC) $(CFLAGS) -DEI_ENABLE_ROCM -Isrc -c -o $@ $<

$(BUILD)/rocm/perf_engine.o: perf/harness/bench_engine.c src/*.h | $(BUILD)/rocm
	$(CC) $(CFLAGS) -DEI_ENABLE_ROCM -Isrc -c -o $@ $<

$(BUILD)/rocm/perf_batch.o: perf/harness/bench_batch.c src/*.h | $(BUILD)/rocm
	$(CC) $(CFLAGS) -DEI_ENABLE_ROCM -Isrc -c -o $@ $<

$(BUILD)/rocm/perf_concurrency.o: perf/harness/bench_concurrency.c src/*.h | $(BUILD)/rocm
	$(CC) $(CFLAGS) -DEI_ENABLE_ROCM -Isrc -c -o $@ $<

$(BUILD)/embeddinggemma-rocm: $(ROCM_SERVER_OBJS) $(ROCM_CORE_OBJS) $(ROCM_SERVICE_OBJS) $(ROCM_ENGINE_OBJ) | $(BUILD)
	$(HIPCC) -o $@ $^ $(LDLIBS) $(ROCM_LDLIBS)

$(BUILD)/test_embed_rocm: $(BUILD)/rocm/test_embed.o $(ROCM_CORE_OBJS) $(ROCM_ENGINE_OBJ) | $(BUILD)
	$(HIPCC) -o $@ $^ $(LDLIBS) $(ROCM_LDLIBS)

$(BUILD)/test_batch_rocm: $(BUILD)/rocm/test_batch.o $(ROCM_CORE_OBJS) $(ROCM_ENGINE_OBJ) | $(BUILD)
	$(HIPCC) -o $@ $^ $(LDLIBS) $(ROCM_LDLIBS)

$(BUILD)/test_rocm: $(BUILD)/rocm/test_rocm.o $(ROCM_CORE_OBJS) $(ROCM_ENGINE_OBJ) | $(BUILD)
	$(HIPCC) -o $@ $^ $(LDLIBS) $(ROCM_LDLIBS)

$(BUILD)/perf_engine_rocm: $(BUILD)/rocm/perf_engine.o $(ROCM_CORE_OBJS) $(ROCM_ENGINE_OBJ) | $(BUILD)
	$(HIPCC) -o $@ $^ $(LDLIBS) $(ROCM_LDLIBS)

$(BUILD)/perf_batch_rocm: $(BUILD)/rocm/perf_batch.o $(ROCM_CORE_OBJS) $(ROCM_ENGINE_OBJ) | $(BUILD)
	$(HIPCC) -o $@ $^ $(LDLIBS) $(ROCM_LDLIBS)

$(BUILD)/perf_concurrency_rocm: $(BUILD)/rocm/perf_concurrency.o $(ROCM_CORE_OBJS) $(ROCM_SERVICE_OBJS) $(ROCM_ENGINE_OBJ) | $(BUILD)
	$(HIPCC) -o $@ $^ $(LDLIBS) $(ROCM_LDLIBS)

$(BUILD)/xpu/%.o: src/%.c src/*.h | $(BUILD)/xpu
	$(CC) $(CFLAGS) -DEI_ENABLE_XPU -Isrc -c -o $@ $<

$(XPU_ENGINE_OBJ): src/engine_xpu.cpp src/engine_xpu.h src/model.h src/gguf.h | $(BUILD)/xpu
	$(SYCL_CXX) $(SYCLFLAGS) $(XPU_ENGINE_TARGET_FLAGS) $(XPU_ENGINE_DEFS) -Isrc -c -o $@ $<

$(BUILD)/xpu/engine_xpu_flash.o: src/engine_xpu_flash.cpp src/engine_xpu_flash.h $(XPU_DEPS_PREREQ) | $(BUILD)/xpu
	$(SYCL_CXX) $(filter-out -ffast-math,$(SYCLFLAGS)) $(XPU_FLASH_FLAGS) -Isrc -c -o $@ $<

$(BUILD)/xpu/engine_xpu_w4.o: src/engine_xpu_w4.cpp src/engine_xpu_w4.h $(XPU_DEPS_PREREQ) | $(BUILD)/xpu
	$(SYCL_CXX) $(filter-out -ffast-math,$(SYCLFLAGS)) $(XPU_FLASH_FLAGS) -Isrc -c -o $@ $<

$(BUILD)/xpu/test_xpu_w4.o: src/test_xpu_w4.cpp src/engine_xpu_w4.h src/model.h $(XPU_DEPS_PREREQ) | $(BUILD)/xpu
	$(SYCL_CXX) $(filter-out -ffast-math,$(SYCLFLAGS)) $(XPU_FLASH_FLAGS) -Isrc -c -o $@ $<

$(BUILD)/test_xpu_w4: $(BUILD)/xpu/test_xpu_w4.o $(BUILD)/xpu/engine_xpu_w4.o $(BUILD)/xpu/gguf.o $(BUILD)/xpu/model.o | $(BUILD)
	$(SYCL_CXX) -o $@ $^ $(LDLIBS) $(SYCL_LDLIBS) $(XPU_LINK_FLAGS)

$(BUILD)/xpu/engine_xpu_onednn.o: src/engine_xpu_onednn.cpp src/engine_xpu_onednn.h | $(BUILD)/xpu
	$(SYCL_CXX) $(SYCLFLAGS) $(XPU_ENGINE_TARGET_FLAGS) -DEI_XPU_ONEDNN -Isrc -c -o $@ $<

$(BUILD)/xpu/test_embed.o: src/test_embed.c src/*.h | $(BUILD)/xpu
	$(CC) $(CFLAGS) -DEI_ENABLE_XPU -Isrc -c -o $@ $<

$(BUILD)/xpu/test_batch.o: src/test_batch.c src/*.h | $(BUILD)/xpu
	$(CC) $(CFLAGS) -DEI_ENABLE_XPU -Isrc -c -o $@ $<

$(XPU_TEST_OBJ): src/test_xpu.c src/*.h | $(BUILD)/xpu
	$(CC) $(CFLAGS) -DEI_ENABLE_XPU $(XPU_ENGINE_DEFS) -Isrc -c -o $@ $<

$(BUILD)/xpu/perf_engine.o: perf/harness/bench_engine.c src/*.h | $(BUILD)/xpu
	$(CC) $(CFLAGS) -DEI_ENABLE_XPU -Isrc -c -o $@ $<

$(BUILD)/xpu/perf_batch.o: perf/harness/bench_batch.c src/*.h | $(BUILD)/xpu
	$(CC) $(CFLAGS) -DEI_ENABLE_XPU -Isrc -c -o $@ $<

$(BUILD)/xpu/perf_concurrency.o: perf/harness/bench_concurrency.c src/*.h | $(BUILD)/xpu
	$(CC) $(CFLAGS) -DEI_ENABLE_XPU -Isrc -c -o $@ $<

$(BUILD)/embeddinggemma-xpu: $(XPU_SERVER_OBJS) $(XPU_CORE_OBJS) $(XPU_SERVICE_OBJS) $(XPU_ENGINE_OBJS) FORCE_XPU_LINK | $(BUILD)
	$(SYCL_CXX) -o $@ $(XPU_SERVER_OBJS) $(XPU_CORE_OBJS) $(XPU_SERVICE_OBJS) \
		$(XPU_ENGINE_OBJS) $(LDLIBS) $(SYCL_LDLIBS) $(XPU_LINK_FLAGS)

$(BUILD)/test_embed_xpu: $(BUILD)/xpu/test_embed.o $(XPU_CORE_OBJS) $(XPU_ENGINE_OBJS) FORCE_XPU_LINK | $(BUILD)
	$(SYCL_CXX) -o $@ $(filter %.o,$^) $(LDLIBS) $(SYCL_LDLIBS) $(XPU_LINK_FLAGS)

$(BUILD)/test_batch_xpu: $(BUILD)/xpu/test_batch.o $(XPU_CORE_OBJS) $(XPU_ENGINE_OBJS) FORCE_XPU_LINK | $(BUILD)
	$(SYCL_CXX) -o $@ $(filter %.o,$^) $(LDLIBS) $(SYCL_LDLIBS) $(XPU_LINK_FLAGS)

$(BUILD)/test_xpu: $(XPU_TEST_OBJ) $(XPU_CORE_OBJS) $(XPU_ENGINE_OBJS) FORCE_XPU_LINK | $(BUILD)
	$(SYCL_CXX) -o $@ $(filter %.o,$^) $(LDLIBS) $(SYCL_LDLIBS) $(XPU_LINK_FLAGS)

$(BUILD)/perf_engine_xpu: $(BUILD)/xpu/perf_engine.o $(XPU_CORE_OBJS) $(XPU_ENGINE_OBJS) FORCE_XPU_LINK | $(BUILD)
	$(SYCL_CXX) -o $@ $(filter %.o,$^) $(LDLIBS) $(SYCL_LDLIBS) $(XPU_LINK_FLAGS)

$(BUILD)/perf_batch_xpu: $(BUILD)/xpu/perf_batch.o $(XPU_CORE_OBJS) $(XPU_ENGINE_OBJS) FORCE_XPU_LINK | $(BUILD)
	$(SYCL_CXX) -o $@ $(filter %.o,$^) $(LDLIBS) $(SYCL_LDLIBS) $(XPU_LINK_FLAGS)

$(BUILD)/perf_concurrency_xpu: $(BUILD)/xpu/perf_concurrency.o $(XPU_CORE_OBJS) $(XPU_SERVICE_OBJS) $(XPU_ENGINE_OBJS) FORCE_XPU_LINK | $(BUILD)
	$(SYCL_CXX) -o $@ $(filter %.o,$^) $(LDLIBS) $(SYCL_LDLIBS) $(XPU_LINK_FLAGS)

$(BUILD)/engine_metal.o: src/engine_metal.m src/engine_metal.h src/model.h src/gguf.h | $(BUILD)
	$(OBJC) $(OBJCFLAGS) -Isrc -c -o $@ $<

$(BUILD)/embeddinggemma-metal: $(SRCS_SERVER) $(SRCS_ENGINE) $(SRCS_SERVICE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -o $@ $(SRCS_SERVER) $(SRCS_ENGINE) \
		$(SRCS_SERVICE) $(BUILD)/engine_metal.o $(LDLIBS) \
		$(METALLIB_LDFLAGS) $(METAL_FRAMEWORKS)

$(BUILD)/test_embed_metal: src/test_embed.c $(SRCS_ENGINE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -o $@ src/test_embed.c $(SRCS_ENGINE) \
		$(BUILD)/engine_metal.o $(LDLIBS) $(METALLIB_LDFLAGS) $(METAL_FRAMEWORKS)

$(BUILD)/test_backend_parity_metal: src/test_backend_parity.c $(SRCS_ENGINE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -o $@ src/test_backend_parity.c $(SRCS_ENGINE) \
		$(BUILD)/engine_metal.o $(LDLIBS) $(METALLIB_LDFLAGS) $(METAL_FRAMEWORKS)

$(BUILD)/test_batch_metal: src/test_batch.c $(SRCS_ENGINE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -o $@ src/test_batch.c $(SRCS_ENGINE) \
		$(BUILD)/engine_metal.o $(LDLIBS) $(METALLIB_LDFLAGS) $(METAL_FRAMEWORKS)

$(BUILD)/perf_kernels: perf/harness/bench_kernels.c src/quants.c src/kernels.c src/gguf.c src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $@ perf/harness/bench_kernels.c src/quants.c src/kernels.c src/gguf.c $(LDLIBS)

$(BUILD)/perf_engine: perf/harness/bench_engine.c $(SRCS_ENGINE) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $@ perf/harness/bench_engine.c $(SRCS_ENGINE) $(LDLIBS)

$(BUILD)/perf_engine_metal: perf/harness/bench_engine.c $(SRCS_ENGINE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -Isrc -o $@ perf/harness/bench_engine.c $(SRCS_ENGINE) \
		$(BUILD)/engine_metal.o $(LDLIBS) $(METALLIB_LDFLAGS) $(METAL_FRAMEWORKS)

$(BUILD)/perf_concurrency: perf/harness/bench_concurrency.c $(SRCS_ENGINE) $(SRCS_SERVICE) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $@ perf/harness/bench_concurrency.c $(SRCS_ENGINE) $(SRCS_SERVICE) $(LDLIBS)

$(BUILD)/perf_concurrency_metal: perf/harness/bench_concurrency.c $(SRCS_ENGINE) $(SRCS_SERVICE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -Isrc -o $@ perf/harness/bench_concurrency.c $(SRCS_ENGINE) \
		$(SRCS_SERVICE) $(BUILD)/engine_metal.o $(LDLIBS) $(METALLIB_LDFLAGS) \
		$(METAL_FRAMEWORKS)

$(BUILD)/perf_batch: perf/harness/bench_batch.c $(SRCS_ENGINE) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $@ perf/harness/bench_batch.c $(SRCS_ENGINE) $(LDLIBS)

$(BUILD)/perf_batch_metal: perf/harness/bench_batch.c $(SRCS_ENGINE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -Isrc -o $@ perf/harness/bench_batch.c $(SRCS_ENGINE) \
		$(BUILD)/engine_metal.o $(LDLIBS) $(METALLIB_LDFLAGS) $(METAL_FRAMEWORKS)

$(BUILD)/perf_tokenization: perf/harness/bench_tokenization.c $(SRCS_TOKENIZER) $(SRCS_SERVICE) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $@ perf/harness/bench_tokenization.c $(SRCS_TOKENIZER) \
		$(SRCS_SERVICE) $(LDLIBS)

$(METALLIB): $(METAL_SOURCES) $(METAL_HEADERS) | $(BUILD)
	@xcrun --find metal >/dev/null 2>&1 || { \
		echo "error: Metal compiler not found; install/select full Xcode with xcode-select" >&2; \
		exit 1; \
	}
	xcrun -sdk macosx metal $(METAL_FLAGS) -I$(METAL_INCLUDE_DIR) $(METAL_SOURCES) -o $@

$(BUILD)/test_metal: src/test_metal.m | $(BUILD)
	$(OBJC) $(OBJCFLAGS) -framework Foundation -framework Metal -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/cuda:
	mkdir -p $(BUILD)/cuda

$(BUILD)/rocm:
	mkdir -p $(BUILD)/rocm

$(BUILD)/xpu:
	mkdir -p $(BUILD)/xpu

.PHONY: test test-unit check test-http test-http-metal test-http-cuda test-http-rocm test-http-xpu test-metal test-cuda test-rocm test-xpu perf perf-engine perf-engine-metal perf-engine-cuda perf-engine-rocm perf-engine-xpu perf-concurrency perf-concurrency-cuda perf-concurrency-rocm perf-concurrency-xpu perf-dimensions perf-batch perf-tokenization metal metal-kernels cuda rocm xpu check-scripts clean clean-cpu clean-metal clean-cuda clean-rocm clean-xpu release-darwin release-linux-cpu release-linux-cuda release-linux-rocm release-linux-xpu release-checksums release-verify release-ready release-info help
test: $(BUILD)/embeddinggemma $(BUILD)/test_gguf $(BUILD)/test_tokenizer $(BUILD)/test_kernels $(BUILD)/test_embed $(BUILD)/test_batch $(BUILD)/test_inference_service $(BUILD)/test_response_cache
	python3 testdata/test_model_manifest.py --binary ./$(BUILD)/test_gguf \
		--model $(MODEL) --manifest testdata/model-manifest.json
	./$(BUILD)/test_tokenizer $(MODEL) testdata/goldens-tokens.json
	./$(BUILD)/test_kernels
	./$(BUILD)/test_embed $(MODEL) testdata/goldens-llamacpp.json
	./$(BUILD)/test_batch $(MODEL) cpu
	./$(BUILD)/test_inference_service
	./$(BUILD)/test_response_cache
	python3 testdata/test_http_dimensions.py --binary ./$(BUILD)/embeddinggemma \
		--model $(MODEL) --backend cpu

test-unit: $(BUILD)/test_kernels $(BUILD)/test_inference_service $(BUILD)/test_response_cache
	./$(BUILD)/test_kernels
	./$(BUILD)/test_inference_service
	./$(BUILD)/test_response_cache

check: check-scripts test-unit

test-http: $(BUILD)/embeddinggemma
	python3 testdata/test_http_dimensions.py --binary ./$(BUILD)/embeddinggemma \
		--model $(MODEL) --backend cpu

test-http-metal: $(BUILD)/embeddinggemma-metal
	python3 testdata/test_http_dimensions.py --binary ./$(BUILD)/embeddinggemma-metal \
		--model $(MODEL) --backend metal

test-http-cuda: $(BUILD)/embeddinggemma-cuda
	python3 testdata/test_http_dimensions.py --binary ./$(BUILD)/embeddinggemma-cuda \
		--model $(MODEL) --backend cuda

test-http-rocm: $(BUILD)/embeddinggemma-rocm
	python3 testdata/test_http_dimensions.py --binary ./$(BUILD)/embeddinggemma-rocm \
		--model $(MODEL) --backend rocm

test-http-xpu: $(BUILD)/embeddinggemma-xpu
	python3 testdata/test_http_dimensions.py --binary ./$(BUILD)/embeddinggemma-xpu \
		--model $(MODEL) --backend xpu

perf: $(BUILD)/perf_kernels
	python3 perf/bench_kernels.py --no-build --binary ./$(BUILD)/perf_kernels --preset quick

perf-engine: $(BUILD)/perf_engine
	python3 perf/bench_engine.py --no-build --model $(MODEL) --backend cpu

perf-engine-metal: $(BUILD)/perf_engine_metal
	python3 perf/bench_engine.py --no-build --model $(MODEL) --backend metal

perf-engine-cuda: $(BUILD)/perf_engine_cuda
	python3 perf/bench_engine.py --no-build --model $(MODEL) --backend cuda

perf-engine-rocm: $(BUILD)/perf_engine_rocm
	python3 perf/bench_engine.py --no-build --model $(MODEL) --backend rocm

perf-engine-xpu: $(BUILD)/perf_engine_xpu
	python3 perf/bench_engine.py --no-build --model $(MODEL) --backend xpu

perf-concurrency: $(BUILD)/perf_concurrency $(BUILD)/perf_concurrency_metal
	python3 perf/bench_concurrency.py --no-build --model $(MODEL) --backend both

perf-concurrency-cuda: $(BUILD)/perf_concurrency_cuda
	python3 perf/bench_concurrency.py --no-build --model $(MODEL) --backend cuda

perf-concurrency-rocm: $(BUILD)/perf_concurrency_rocm
	python3 perf/bench_concurrency.py --no-build --model $(MODEL) --backend rocm

perf-concurrency-xpu: $(BUILD)/perf_concurrency_xpu
	python3 perf/bench_concurrency.py --no-build --model $(MODEL) --backend xpu

perf-dimensions: $(BUILD)/embeddinggemma $(BUILD)/embeddinggemma-metal
	python3 perf/bench_dimensions.py --no-build --model $(MODEL) --backend cpu
	python3 perf/bench_dimensions.py --no-build --model $(MODEL) --backend metal

perf-batch: $(BUILD)/perf_batch $(BUILD)/perf_batch_metal
	./$(BUILD)/perf_batch --model $(MODEL) --backend cpu
	./$(BUILD)/perf_batch_metal --model $(MODEL) --backend metal

perf-tokenization: $(BUILD)/perf_tokenization
	./$(BUILD)/perf_tokenization --model $(MODEL)

metal-kernels: $(METALLIB)

metal: $(BUILD)/embeddinggemma-metal

cuda: $(BUILD)/embeddinggemma-cuda

rocm: $(BUILD)/embeddinggemma-rocm

xpu: $(BUILD)/embeddinggemma-xpu

test-metal: $(METALLIB) $(BUILD)/test_metal $(BUILD)/test_embed_metal $(BUILD)/test_backend_parity_metal $(BUILD)/test_batch_metal
	./$(BUILD)/test_metal $(METALLIB)
	EI_BACKEND=metal ./$(BUILD)/test_embed_metal $(MODEL) testdata/goldens-llamacpp.json
	./$(BUILD)/test_backend_parity_metal $(MODEL)
	./$(BUILD)/test_batch_metal $(MODEL) metal

test-cuda: $(BUILD)/test_embed_cuda $(BUILD)/test_cuda $(BUILD)/test_batch_cuda $(BUILD)/embeddinggemma-cuda
	EI_BACKEND=cuda ./$(BUILD)/test_embed_cuda $(MODEL) testdata/goldens-llamacpp.json
	./$(BUILD)/test_cuda $(MODEL)
	./$(BUILD)/test_batch_cuda $(MODEL) cuda
	python3 testdata/test_http_dimensions.py --binary ./$(BUILD)/embeddinggemma-cuda \
		--model $(MODEL) --backend cuda

test-rocm: $(BUILD)/test_embed_rocm $(BUILD)/test_rocm $(BUILD)/test_batch_rocm $(BUILD)/embeddinggemma-rocm
	EI_BACKEND=rocm ./$(BUILD)/test_embed_rocm $(MODEL) testdata/goldens-llamacpp.json
	./$(BUILD)/test_rocm $(MODEL)
	./$(BUILD)/test_batch_rocm $(MODEL) rocm
	python3 testdata/test_http_dimensions.py --binary ./$(BUILD)/embeddinggemma-rocm \
		--model $(MODEL) --backend rocm

test-xpu: $(BUILD)/test_embed_xpu $(BUILD)/test_xpu $(BUILD)/test_batch_xpu $(BUILD)/embeddinggemma-xpu
	EI_BACKEND=xpu ./$(BUILD)/test_embed_xpu $(MODEL) testdata/goldens-llamacpp.json
	./$(BUILD)/test_xpu $(MODEL)
	./$(BUILD)/test_batch_xpu $(MODEL) xpu
	python3 testdata/test_http_dimensions.py --binary ./$(BUILD)/embeddinggemma-xpu \
		--model $(MODEL) --backend xpu

print-cc-version:
	@$(CC) --version 2>/dev/null | sed -n '1p'

check-scripts:
	sh -n install.sh scripts/fetch-xpu-deps.sh scripts/stage-release.sh scripts/release-assets.sh

clean-cpu:
	rm -f $(BUILD)/embeddinggemma

clean-metal:
	rm -f $(METALLIB) $(BUILD)/engine_metal.o $(BUILD)/*metal $(BUILD)/*metal.o

clean-cuda:
	rm -rf $(BUILD)/cuda
	rm -f $(BUILD)/*cuda

clean-rocm:
	rm -rf $(BUILD)/rocm
	rm -f $(BUILD)/*rocm

clean-xpu:
	rm -rf $(BUILD)/xpu
	rm -f $(BUILD)/*xpu $(BUILD)/test_xpu_w4

release-darwin:
	$(MAKE) clean-cpu clean-metal
	$(MAKE) all
	$(MAKE) metal
	./scripts/stage-release.sh cpu $(BUILD)/embeddinggemma $(DIST)
	./scripts/stage-release.sh metal $(BUILD)/embeddinggemma-metal $(DIST)

release-linux-cpu:
	$(MAKE) clean-cpu
	$(MAKE) all
	./scripts/stage-release.sh cpu $(BUILD)/embeddinggemma $(DIST)

release-linux-cuda:
	@test "$(origin CUDA_ARCH)" = undefined && \
		test "$(origin CUDA_ARCHS)" = file && \
		test "$(origin CUDA_PTX_ARCH)" = file || { \
		echo 'error: CUDA architecture overrides are forbidden for release builds' >&2; exit 1; }
	$(MAKE) clean-cuda
	$(MAKE) cuda
	./scripts/stage-release.sh cuda $(BUILD)/embeddinggemma-cuda $(DIST)

release-linux-rocm:
	@test "$(origin ROCM_ARCH)" = undefined && \
		test "$(origin ROCM_ARCHS)" = file || { \
		echo 'error: ROCm architecture overrides are forbidden for release builds' >&2; exit 1; }
	$(MAKE) clean-rocm
	$(MAKE) rocm
	./scripts/stage-release.sh rocm $(BUILD)/embeddinggemma-rocm $(DIST)

release-linux-xpu:
	$(MAKE) clean-xpu
	$(MAKE) xpu XPU_XE2_FLASH=1
	./scripts/stage-release.sh xpu $(BUILD)/embeddinggemma-xpu $(DIST)

release-checksums:
	./scripts/release-assets.sh checksums $(DIST)

release-verify:
	./scripts/release-assets.sh verify $(DIST)

release-ready: check-scripts release-verify
	@test -n "$(VERSION)" || { echo 'error: VERSION is empty' >&2; exit 1; }
	@printf '%s\n' "$(VERSION)" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+$$' || { \
		echo 'error: VERSION must use vMAJOR.MINOR.PATCH' >&2; exit 1; }
	@test -z "$$(git status --porcelain)" || { \
		echo 'error: release worktree is dirty' >&2; git status --short >&2; exit 1; }
	@git grep -q -- "--version $(VERSION)" README.md || { \
		echo 'error: README pinned installer example does not match VERSION' >&2; exit 1; }
	@if git rev-parse -q --verify "refs/tags/$(VERSION)" >/dev/null; then \
		test "$$(git rev-list -n 1 "$(VERSION)")" = "$$(git rev-parse HEAD)" || { \
			echo 'error: existing VERSION tag does not point to HEAD' >&2; exit 1; }; \
	fi
	@printf 'Release %s is ready at commit %s\n' "$(VERSION)" "$$(git rev-parse --short=12 HEAD)"

release-info:
	@printf 'version=%s\ncommit=%s\ndist=%s\n' "$(VERSION)" \
		"$$(git rev-parse HEAD 2>/dev/null || printf unknown)" "$(DIST)"

help:
	@printf '%s\n' \
		'make | metal | cuda | rocm | xpu  Build a server backend' \
		'make check                         Run model-free unit and script checks' \
		'make test | test-BACKEND          Run correctness and HTTP tests' \
		'make perf-*                       Run kernel, engine, batch, or concurrency benchmarks' \
		'make clean-BACKEND                Remove backend objects before changing build flags' \
		'make release-darwin               Stage Darwin ARM64 CPU and Metal executables' \
		'make release-linux-BACKEND        Stage a Linux x86_64 executable' \
		'make release-checksums            Write checksums for all six executables' \
		'make release-ready                Verify scripts, assets, checksums, version, and git state'

clean:
	rm -rf $(BUILD) $(DIST)
