CPPFLAGS = -O3 -std=c++0x
CUFLAGS = -O3 -use_fast_math -c -Xptxas -v -gencode arch=compute_13,code=sm_13 -gencode arch=compute_20,code=sm_20
#CUFLAGS = -O3 -use_fast_math -c -Xptxas -v -arch=sm_20
TARGET = minhash

.PHONY: clean

all: $(TARGET)

$(TARGET): characteristicMatrix.h characteristicMatrix.cpp main.cpp kernel.o
	g++ $(CPPFLAGS) characteristicMatrix.h characteristicMatrix.cpp main.cpp kernel.o -o $(TARGET) -L/usr/local/cuda/lib64 -lcudart

kernel.o: kernel.cu kernel.h
	nvcc $(CUFLAGS) -o $@ $<
	@ echo "--------------------------------------------------------------------------------"

clean:
	rm -rf $(CUOBJ) $(TARGET)
