CPPFLAGS = -O3 -std=c++0x
TARGET = minhash

.PHONY: clean

all: $(TARGET)

$(TARGET): main.cpp characteristicMatrix.h characteristicMatrix.cpp
	g++ $(CPPFLAGS) characteristicMatrix.h characteristicMatrix.cpp main.cpp -o $(TARGET) 

clean:
	rm -rf $(TARGET)
