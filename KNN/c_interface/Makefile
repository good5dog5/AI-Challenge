main: main.cpp kernel.h kernel.cpp
	g++ -O3 -std=c++11 -fopenmp main.cpp kernel.cpp

libkernel: kernel.h kernel.cpp
	g++ -O2 -std=c++11 -fopenmp -shared -fPIC kernel.cpp -o libkernel.so