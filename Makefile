# embeddinggemma.c CPU and Metal build
CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Werror -g
LDLIBS  ?= -lm -pthread
CURL_CFLAGS ?= $(shell curl-config --cflags 2>/dev/null)
CURL_LIBS ?= $(shell curl-config --libs 2>/dev/null || printf '%s' '-lcurl')
BUILD   := build
MODEL   ?= model/embeddinggemma-300M-qat-Q4_0.gguf

# Prefer a full Xcode installation for Metal without requiring a global
# xcode-select change. An environment-provided DEVELOPER_DIR still wins.
ifeq ($(origin DEVELOPER_DIR), undefined)
ifneq ($(wildcard /Applications/Xcode.app/Contents/Developer),)
DEVELOPER_DIR := /Applications/Xcode.app/Contents/Developer
export DEVELOPER_DIR
endif
endif

# QuixiCore-Metal-style build: compile every Metal translation unit into one
# colocated metallib. Keep the source list explicit and track included headers.
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
OBJC ?= clang
OBJCFLAGS ?= -O2 -Wall -Wextra -Werror -fobjc-arc
METAL_FRAMEWORKS := -framework Foundation -framework Metal

SRCS_CORE := src/gguf.c
SRCS_MODEL := $(SRCS_CORE) src/model.c
SRCS_TOKENIZER := $(SRCS_MODEL) src/tokenizer.c
SRCS_ENGINE := $(SRCS_TOKENIZER) src/quants.c src/kernels.c src/parallel.c src/engine.c
SRCS_SERVICE := src/inference_service.c
SRCS_SERVER := src/server.c src/response_cache.c

.PHONY: all
all: $(BUILD)/embeddinggemma

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
	$(CC) $(CFLAGS) $(CURL_CFLAGS) -o $@ $(SRCS_SERVER) $(SRCS_ENGINE) $(SRCS_SERVICE) $(LDLIBS) $(CURL_LIBS)

$(BUILD)/test_inference_service: src/test_inference_service.c $(SRCS_SERVICE) src/tokenizer.c src/gguf.c src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/test_inference_service.c $(SRCS_SERVICE) src/tokenizer.c src/gguf.c $(LDLIBS)

$(BUILD)/test_response_cache: src/test_response_cache.c src/response_cache.c src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ src/test_response_cache.c src/response_cache.c src/gguf.c $(LDLIBS)

$(BUILD)/engine_metal.o: src/engine_metal.m src/engine_metal.h src/model.h src/gguf.h | $(BUILD)
	$(OBJC) $(OBJCFLAGS) -Isrc -c -o $@ $<

$(BUILD)/embeddinggemma-metal: $(SRCS_SERVER) $(SRCS_ENGINE) $(SRCS_SERVICE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL $(CURL_CFLAGS) -o $@ $(SRCS_SERVER) $(SRCS_ENGINE) \
		$(SRCS_SERVICE) $(BUILD)/engine_metal.o $(LDLIBS) $(CURL_LIBS) $(METAL_FRAMEWORKS)

$(BUILD)/test_embed_metal: src/test_embed.c $(SRCS_ENGINE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -o $@ src/test_embed.c $(SRCS_ENGINE) \
		$(BUILD)/engine_metal.o $(LDLIBS) $(METAL_FRAMEWORKS)

$(BUILD)/test_backend_parity_metal: src/test_backend_parity.c $(SRCS_ENGINE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -o $@ src/test_backend_parity.c $(SRCS_ENGINE) \
		$(BUILD)/engine_metal.o $(LDLIBS) $(METAL_FRAMEWORKS)

$(BUILD)/test_batch_metal: src/test_batch.c $(SRCS_ENGINE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -o $@ src/test_batch.c $(SRCS_ENGINE) \
		$(BUILD)/engine_metal.o $(LDLIBS) $(METAL_FRAMEWORKS)

$(BUILD)/perf_kernels: perf/harness/bench_kernels.c src/quants.c src/kernels.c src/gguf.c src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $@ perf/harness/bench_kernels.c src/quants.c src/kernels.c src/gguf.c $(LDLIBS)

$(BUILD)/perf_engine: perf/harness/bench_engine.c $(SRCS_ENGINE) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $@ perf/harness/bench_engine.c $(SRCS_ENGINE) $(LDLIBS)

$(BUILD)/perf_engine_metal: perf/harness/bench_engine.c $(SRCS_ENGINE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -Isrc -o $@ perf/harness/bench_engine.c $(SRCS_ENGINE) \
		$(BUILD)/engine_metal.o $(LDLIBS) $(METAL_FRAMEWORKS)

$(BUILD)/perf_concurrency: perf/harness/bench_concurrency.c $(SRCS_ENGINE) $(SRCS_SERVICE) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $@ perf/harness/bench_concurrency.c $(SRCS_ENGINE) $(SRCS_SERVICE) $(LDLIBS)

$(BUILD)/perf_concurrency_metal: perf/harness/bench_concurrency.c $(SRCS_ENGINE) $(SRCS_SERVICE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -Isrc -o $@ perf/harness/bench_concurrency.c $(SRCS_ENGINE) \
		$(SRCS_SERVICE) $(BUILD)/engine_metal.o $(LDLIBS) $(METAL_FRAMEWORKS)

$(BUILD)/perf_batch: perf/harness/bench_batch.c $(SRCS_ENGINE) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $@ perf/harness/bench_batch.c $(SRCS_ENGINE) $(LDLIBS)

$(BUILD)/perf_batch_metal: perf/harness/bench_batch.c $(SRCS_ENGINE) $(BUILD)/engine_metal.o src/*.h $(METALLIB) | $(BUILD)
	$(CC) $(CFLAGS) -DEI_ENABLE_METAL -Isrc -o $@ perf/harness/bench_batch.c $(SRCS_ENGINE) \
		$(BUILD)/engine_metal.o $(LDLIBS) $(METAL_FRAMEWORKS)

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

.PHONY: test test-http test-http-metal test-metal perf perf-engine perf-engine-metal perf-concurrency perf-dimensions perf-batch perf-tokenization metal metal-kernels clean
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

test-http: $(BUILD)/embeddinggemma
	python3 testdata/test_http_dimensions.py --binary ./$(BUILD)/embeddinggemma \
		--model $(MODEL) --backend cpu

test-http-metal: $(BUILD)/embeddinggemma-metal
	python3 testdata/test_http_dimensions.py --binary ./$(BUILD)/embeddinggemma-metal \
		--model $(MODEL) --backend metal

perf: $(BUILD)/perf_kernels
	python3 perf/bench_kernels.py --no-build --binary ./$(BUILD)/perf_kernels --preset quick

perf-engine: $(BUILD)/perf_engine
	python3 perf/bench_engine.py --no-build --model $(MODEL) --backend cpu

perf-engine-metal: $(BUILD)/perf_engine_metal
	python3 perf/bench_engine.py --no-build --model $(MODEL) --backend metal

perf-concurrency: $(BUILD)/perf_concurrency $(BUILD)/perf_concurrency_metal
	python3 perf/bench_concurrency.py --no-build --model $(MODEL) --backend both

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

test-metal: $(METALLIB) $(BUILD)/test_metal $(BUILD)/test_embed_metal $(BUILD)/test_backend_parity_metal $(BUILD)/test_batch_metal
	./$(BUILD)/test_metal $(METALLIB)
	EI_BACKEND=metal ./$(BUILD)/test_embed_metal $(MODEL) testdata/goldens-llamacpp.json
	./$(BUILD)/test_backend_parity_metal $(MODEL)
	./$(BUILD)/test_batch_metal $(MODEL) metal

print-cc-version:
	@$(CC) --version 2>/dev/null | sed -n '1p'

clean:
	rm -rf $(BUILD)
