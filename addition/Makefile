
NVCC        = /usr/local/cuda/bin/nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = addition
OBJ	        = addition.o

default: $(EXE)

addition.o: addition.cu
	nvcc -c -o $@ addition.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	nvcc $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)