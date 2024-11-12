//
//  mlp.hpp
//  mlptest
//
//  Created by Josh Lakin on 12/4/23.
//

#pragma once
#include <vector>
#include <random>
#include <chrono>
#include "util.h"
#include <cassert>
#include <math.h>

enum ActivationFunction {
    ACTIVATION_FUNCTION_NONE = 0,
    ACTIVATION_FUNCTION_RELU,
    ACTIVATION_FUNCTION_SIGMOID,
    NUM_ACTIVATION_FUNCTIONS
};

//length of vector must be cols
//length of bias must be rows, is optional
std::vector<float> matrixVectorMul(const std::vector<float>& matrix, unsigned rows, unsigned cols, const std::vector<float>& vector, float* bias, bool transpose);

std::vector<float> relu(const std::vector<float>& vector);

std::vector<float> reluDerivative(const std::vector<float>& vector);

std::vector<float> sigmoid(const std::vector<float>& vector);

std::vector<float> sigmoidDerivative(const std::vector<float>& vector);

struct Layer {
    std::vector<float> weightMatrix; //2d array
    std::vector<float> biasVector; //1d array, size=weightMatrixRows
    unsigned weightMatrixRows = 0;
    unsigned weightMatrixCols = 0;
    void init(unsigned numRows, unsigned numCols) {
        //glorot initialization
        weightMatrixRows = numRows;
        weightMatrixCols = numCols;
        
        weightMatrix.resize(numRows*numCols);
        biasVector.resize(numRows);
        
        const unsigned long long seed = std::chrono::high_resolution_clock::now()
                                       .time_since_epoch()
                                       .count();
        double sd = std::sqrt(2.0 / (weightMatrixRows + weightMatrixCols));
        std::mt19937_64 engine{seed};
        std::normal_distribution<float> dist(0, sd);
        for (unsigned i = 0; i < numRows; i++) {
            for (unsigned j = 0; j < numCols; j++) {
                unsigned index = getIndex(i, j, numCols);
                weightMatrix.at(index) = dist(engine);
            }
        }
        for (unsigned i = 0; i < numRows; i++) {
            biasVector.at(i) = dist(engine);
        }
    }
    void print() {
        if (!weightMatrix.empty() && !biasVector.empty()) {
            std::cout << "W = ";
            print2DArray(weightMatrix.data(), weightMatrixRows, weightMatrixCols);
            std::cout << "b = ";
            printArray(biasVector.data(), weightMatrixCols);
        }
    }
    ~Layer() {
    }
    ActivationFunction activationFunction = ACTIVATION_FUNCTION_NONE;
};

struct MultilayerPerceptron {
    unsigned numLayers = 0;
    std::vector<Layer> layers;
    void init(unsigned numLayers) {
        this->numLayers = numLayers;
        layers.resize(numLayers);
    }
    ~MultilayerPerceptron() {
    }
    void print() {
        if (!layers.empty()) {
            for (unsigned i = 0; i < numLayers; i++) {
                std::cout << "Layer " << i << ":" << std::endl;
                layers[i].print();
                std::cout << std::endl;
            }
        }
    }
};

struct LayerInference {
    std::vector<float> z; //pre-activation
    std::vector<float> a; //post-activation
};

using Datapoint = std::pair<std::vector<float>, float>;

inline float loss1D(float approx, float y) {
    float temp = approx - y;
    return 0.5f*temp*temp;
}

inline float loss1DDerivative(float approx, float y) {
    return approx - y;
};

std::vector<LayerInference> getInference(const std::vector<float>& input, MultilayerPerceptron& mlp);

void sgd(MultilayerPerceptron& mlp, const std::vector<Datapoint>& data, float lr, unsigned batchSize, unsigned numEpochs);

inline void arrayComponentAdd(std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    for (int i = 0; i < a.size(); i++) {
        a.at(i) = a.at(i) + b.at(i);
    }
}

inline void arrayComponentAddScaled(std::vector<float>& a, const std::vector<float>& b, float scale) {
    assert(a.size() == b.size());
    for (int i = 0; i < a.size(); i++) {
        a.at(i) = a.at(i) + scale*b.at(i);
    }
}

inline void arrayComponentSubtract(std::vector<float>& a, const std::vector<float>& b, float b_scalar) {
    assert(a.size() == b.size());
    for (int i = 0; i < a.size(); i++) {
        a.at(i) = a.at(i) - b_scalar*b.at(i);
    }
}

inline void arrayComponentMult(std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    for (int i = 0; i < a.size(); i++) {
        a.at(i) = a.at(i) * b.at(i);
    }
}

inline void arrayScale(std::vector<float>& a, float scale) {
    for (int i = 0; i < a.size(); i++) {
        a.at(i) *= scale;
    }
}

inline std::vector<float> outerProduct(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result;
    result.resize(a.size()*b.size());
    for (int c = 0; c < b.size(); c++) {
        for (int r = 0; r < a.size(); r++) {
            unsigned index = getIndex(r, c, static_cast<unsigned>(b.size()));
            result.at(index) = a.at(r)*b.at(c);
        }
    }
    return result;
}

//x and y must be floats normalized to range of [-1,1]
//https://arxiv.org/pdf/2003.08934.pdf
void positionalEncoding(float x, float y, std::vector<float>& out);

//make sure width and height are divisible by batchSize
std::vector<Datapoint> createDataset(const char* data, unsigned width, unsigned height, unsigned batchSizeX, unsigned batchSizeY);

//make sure width and height are divisible by batchSize
std::vector<Datapoint> createDataset32(const char* data, unsigned width, unsigned height, unsigned batchSizeX, unsigned batchSizeY);

char* reconstructImage(MultilayerPerceptron& mlp, unsigned width, unsigned height);