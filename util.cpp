//
//  util.cpp
//  mlptest
//
//  Created by Josh Lakin on 12/4/23.
//

#include "util.h"
#include <fstream>
#include <string>

void printArray(float arr[], unsigned len) {
    std::cout << "[";
    for (unsigned i = 0; i < len - 1; i++) {
        std::cout << arr[i] << ", ";
    }
    std::cout << arr[len - 1] << "]";
}

void print2DArray(float arr[], unsigned rows, unsigned cols) {
    std::cout << "[";
    for (unsigned i = 0; i < rows - 1; i++) {
        printArray(&arr[cols*i], cols);
        std::cout << ", ";
    }
    printArray(&arr[(rows-1)*cols], cols);
    std::cout << "]" << std::endl;
}

std::vector<float> createSquareIdentityMatrix(unsigned dim) {
    std::vector<float> ret;
    ret.resize(dim * dim);
    for (unsigned i = 0; i < dim; i++) {
        for (unsigned j = 0; j < dim; j++) {
            unsigned index = getIndex(i, j, dim);
            if (i == j) ret[index] = 1.0f;
            else ret[index] = 0.0f;
        }
    }
    return ret;
}

char* loadPGM(std::string path, int& width, int& height, int& maxValue) {
    //https://netpbm.sourceforge.net/doc/ppm.html file format specified here
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return nullptr;
    }
    char c;
    file >> c;
    if (c != 'P') return nullptr;
    file >> c;
    if (c != '5') return nullptr; //5 for grayscale

    std::string widthStr;
    file >> c;
    widthStr.append(1, c);
    do {
        file.read(&c, 1);
        if (!isspace(static_cast<unsigned char>(c))) widthStr.append(1, c);
    } while (!isspace(static_cast<unsigned char>(c)));
    width = std::stoi(widthStr);

    std::string heightStr;
    file >> c;
    heightStr.append(1, c);
    do {
        file.read(&c, 1);
        if (!isspace(static_cast<unsigned char>(c))) heightStr.append(1, c);
    } while (!isspace(static_cast<unsigned char>(c)));
    height = std::stoi(heightStr);
    
    std::string maxValueStr;
    file >> c;
    maxValueStr.append(1, c);
    do {
        file.read(&c, 1);
        if (!isspace(static_cast<unsigned char>(c))) maxValueStr.append(1, c);
    } while (!isspace(static_cast<unsigned char>(c)));
    maxValue = std::stoi(maxValueStr);

    char* ret = new char[width * height];

    file.ignore(1);

    file.read(ret, width * height);

    file.close();

    return ret;
}

void writePGM(std::string path, unsigned width, unsigned height, unsigned maxValue, const char* data) {
    std::ofstream file{ path , std::ios::binary };
    file << 'P' << '5' << '\n';
    std::string widthStr = std::to_string(width);
    for (char c : widthStr) {
        file << c;
    }
    file << ' ';
    std::string heightStr = std::to_string(height);
    for (char c : heightStr) {
        file << c;
    }
    file << '\n';
    std::string maxValueStr = std::to_string(maxValue);
    for (char c : maxValueStr) {
        file << c;
    }
    file << '\n';
    file.write(data, width * height);

    file.close();
}
