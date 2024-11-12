#pragma once
#include <iostream>
#include <vector>

//start at r = 0, c = 0
inline unsigned getIndex(unsigned r, unsigned c, unsigned cols) {
    return r * cols + c;
}

void printArray(float arr[], unsigned len);

void print2DArray(float arr[], unsigned rows, unsigned cols);

std::vector<float> createSquareIdentityMatrix(unsigned dim);

char* loadPGM(std::string path, int& width, int& height, int& maxValue);

void writePGM(std::string path, unsigned width, unsigned height, unsigned maxValue, const char* data);