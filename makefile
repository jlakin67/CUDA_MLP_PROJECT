build: mlp.cpp util.cpp
	nvcc -std c++14 main.cu mlp.cpp util.cpp
