bin=pngmf mf-benchmark
include ../common/Makefile.openmp

pngmf: pngmf.o mf.o pngio.o error.o
	$(CXX) $(LDFLAGS) $^ -lpng -o $@

mf-benchmark: mf-benchmark.o mf.o error.o
	$(CXX) $(LDFLAGS) $^ -o $@

include Makefile.dep
