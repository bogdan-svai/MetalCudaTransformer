NVCC ?= nvcc
CPPFLAGS := -Iinclude
OPT := -O2
OBJECTS := build/main.o build/kernels.o build/sha256.o

.PHONY: compile clean

compile: miner

miner: $(OBJECTS)
	$(NVCC) $(OBJECTS) $(OPT) -o $@

build/main.o: app/main.cpp include/sha256.h
	@mkdir -p build
	$(NVCC) -dc $< $(CPPFLAGS) $(OPT) -MMD -MF $@.d -o $@

build/kernels.o: src/kernels.cu include/sha256.h
	@mkdir -p build
	$(NVCC) -dc $< $(CPPFLAGS) $(OPT) -MMD -MF $@.d -o $@

build/sha256.o: src/sha256.cu include/sha256.h
	@mkdir -p build
	$(NVCC) -dc $< $(CPPFLAGS) $(OPT) -MMD -MF $@.d -o $@

-include $(OBJECTS:%=%.d)

clean:
	rm -rf build miner miner.mcu
