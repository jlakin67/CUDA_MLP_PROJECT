//
//  mlp.cpp
//  mlptest
//
//  Created by Josh Lakin on 12/4/23.
//

#include "mlp.hpp"

//length of vector must be cols
//length of bias must be rows, is optional
std::vector<float> matrixVectorMul(const std::vector<float>& matrix, unsigned rows, unsigned cols, const std::vector<float>& vector, float* bias, bool transpose) {
    assert(matrix.size() == rows*cols);
    if (transpose) assert(vector.size() == rows);
    else assert(vector.size() == cols);
    std::vector<float> result;
    if (transpose) {
        for (unsigned c = 0; c < cols; c++) {
            float dp = 0.f;
            for (unsigned r = 0; r < rows; r++) {
                unsigned index = getIndex(r, c, cols);
                dp += matrix.at(index) * vector.at(r);
            }
            if (bias) dp += bias[c];
            result.push_back(dp);
        }
    }
    else {
        for (unsigned r = 0; r < rows; r++) {
            float dp = 0.f;
            for (unsigned c = 0; c < cols; c++) {
                unsigned index = getIndex(r, c, cols);
                dp += matrix.at(index) * vector.at(c);
            }
            if (bias) dp += bias[r];
            result.push_back(dp);
        }
    }

    return result;
}


std::vector<float> relu(const std::vector<float>& vector) {
    std::vector<float> result;
    for (unsigned i = 0; i < vector.size(); i++) {
        if (vector.at(i) >= 0.f) result.push_back(vector.at(i));
        else result.push_back(0.f);
    }
    return result;
}

std::vector<float> reluDerivative(const std::vector<float>& vector) {
    std::vector<float> result;
    for (unsigned i = 0; i < vector.size(); i++) {
        if (vector.at(i) >= 0.f) result.push_back(1.0f);
        else result.push_back(0.f);
    }
    return result;
}

std::vector<float> sigmoid(const std::vector<float>& vector) {
    std::vector<float> result;
    for (unsigned i = 0; i < vector.size(); i++) {
        result.push_back(1.0f / (1.0f + std::exp(-vector.at(i))) );
    }
    return result;
}

std::vector<float> sigmoidDerivative(const std::vector<float>& vector) {
    std::vector<float> result;
    for (unsigned i = 0; i < vector.size(); i++) {
        float ans = 1.0f / (1.0f + std::exp(-vector.at(i)));
        ans *= (1.0f - ans);
        result.push_back(ans);
    }
    return result;
}

std::vector<LayerInference> getInference(const std::vector<float>& input,  MultilayerPerceptron& mlp) {
    assert(!input.empty());
    assert(!mlp.layers.empty());
    assert(mlp.numLayers > 0);

    std::vector<LayerInference> inferences;

    LayerInference firstInference;
    firstInference.z = matrixVectorMul(mlp.layers.at(0).weightMatrix, mlp.layers.at(0).weightMatrixRows,
        mlp.layers.at(0).weightMatrixCols, input, mlp.layers.at(0).biasVector.data(), false);
    if (mlp.layers.at(0).activationFunction == ACTIVATION_FUNCTION_RELU) {
        firstInference.a = relu(firstInference.z);
    }
    else if (mlp.layers.at(0).activationFunction == ACTIVATION_FUNCTION_SIGMOID) {
        firstInference.a = sigmoid(firstInference.z);
    }
    inferences.push_back(firstInference);
    for (unsigned i = 1; i < mlp.numLayers; i++) {
        LayerInference inference;
        inference.z = matrixVectorMul(mlp.layers.at(i).weightMatrix,
            mlp.layers.at(i).weightMatrixRows,
            mlp.layers.at(i).weightMatrixCols,
            inferences.at(i - 1u).a,
            mlp.layers.at(i).biasVector.data(), false);
        if (mlp.layers.at(i).activationFunction == ACTIVATION_FUNCTION_RELU) {
            inference.a = relu(inference.z);
        }
        else if (mlp.layers.at(i).activationFunction == ACTIVATION_FUNCTION_SIGMOID) {
            inference.a = sigmoid(inference.z);
        }
        inferences.push_back(inference);
    }
    return inferences;
}

void sgd(MultilayerPerceptron& mlp, const std::vector<Datapoint>& data, float lr, unsigned batchSize, unsigned numEpochs) {
    
    for (unsigned i = 0; i < numEpochs; i++) {
        std::cout << "Epoch: " << i << std::endl;
        unsigned dataIndex = 0;
        while (dataIndex < data.size()) {
            unsigned j = 0;
            std::vector<std::vector<float>> layerWeightGradients;
            layerWeightGradients.resize(mlp.numLayers);
            std::vector<std::vector<float>> layerBiasGradients;
            layerBiasGradients.resize(mlp.numLayers);
            for (unsigned l = 0; l < mlp.numLayers; l++) {
                layerWeightGradients.at(l).resize(mlp.layers.at(l).weightMatrixRows * mlp.layers.at(l).weightMatrixCols, 0.f);
                layerBiasGradients.at(l).resize(mlp.layers.at(l).weightMatrixRows, 0.f);
            }
            float loss = 0.f;
            while (dataIndex < data.size() && j < batchSize) {
                const Datapoint& datapoint = data.at(dataIndex);
                auto inferences = getInference(datapoint.first, mlp);

                int layerIndex = mlp.numLayers - 1;
                std::vector<float> weightGradient;
                const Layer& lastLayer = mlp.layers.at(layerIndex);
                weightGradient.resize(lastLayer.weightMatrixRows * lastLayer.weightMatrixCols);
                std::vector<float> biasGradient;

                //do last layer separately outside loop
                float de_da_last = loss1DDerivative(inferences.at(layerIndex).a.at(0), datapoint.second);
                if (lastLayer.activationFunction == ACTIVATION_FUNCTION_RELU) {
                    biasGradient = reluDerivative(inferences.at(layerIndex).z);
                }
                else if (lastLayer.activationFunction == ACTIVATION_FUNCTION_SIGMOID) {
                    biasGradient = sigmoidDerivative(inferences.at(layerIndex).z);
                }
                biasGradient.at(0) *= de_da_last;
                for (unsigned c = 0; c < lastLayer.weightMatrixCols; c++) {
                    unsigned weightIndex = getIndex(0, c, lastLayer.weightMatrixCols);
                    weightGradient.at(weightIndex) = biasGradient.at(0) * inferences.at(layerIndex - 1).a.at(c);
                }
                arrayComponentAdd(layerWeightGradients.at(layerIndex), weightGradient);
                arrayComponentAddScaled(layerWeightGradients.at(layerIndex), lastLayer.weightMatrix, 0.0001f);
                arrayComponentAdd(layerBiasGradients.at(layerIndex), biasGradient);
                arrayComponentAddScaled(layerBiasGradients.at(layerIndex), lastLayer.biasVector, 0.0001f);
                layerIndex--;
                std::vector<float> de_da;
                while (layerIndex >= 0) {
                    const Layer& prevLayer = mlp.layers.at(layerIndex + 1);
                    const Layer& layer = mlp.layers.at(layerIndex);
                    weightGradient.clear();
                    weightGradient.resize(layer.weightMatrixRows * layer.weightMatrixCols);
                    de_da = matrixVectorMul(prevLayer.weightMatrix, prevLayer.weightMatrixRows, prevLayer.weightMatrixCols, biasGradient, nullptr, true);
                    if (layer.activationFunction == ACTIVATION_FUNCTION_RELU) {
                        biasGradient = reluDerivative(inferences.at(layerIndex).z);
                    }
                    else if (layer.activationFunction == ACTIVATION_FUNCTION_SIGMOID) {
                        biasGradient = sigmoidDerivative(inferences.at(layerIndex).z);
                    }
                    arrayComponentMult(biasGradient, de_da);
                    if (layerIndex == 0) {
                        weightGradient = outerProduct(biasGradient, datapoint.first);
                    }
                    else {
                        weightGradient = outerProduct(biasGradient, inferences.at(layerIndex - 1).a);
                    }
                    arrayComponentAdd(layerWeightGradients.at(layerIndex), weightGradient);
                    arrayComponentAddScaled(layerWeightGradients.at(layerIndex), layer.weightMatrix, 0.0001f);
                    arrayComponentAdd(layerBiasGradients.at(layerIndex), biasGradient);
                    arrayComponentAddScaled(layerBiasGradients.at(layerIndex), layer.biasVector, 0.0001f);

                    layerIndex--;
                }

                dataIndex++;
                j++;

                loss += loss1D(inferences.at(mlp.numLayers - 1).a.at(0), datapoint.second);

            }
            for (unsigned l = 0; l < mlp.numLayers; l++) {
                arrayScale(layerWeightGradients.at(l), 1.0f / std::max(j, 1u));
                arrayScale(layerBiasGradients.at(l), 1.0f / std::max(j, 1u));
                arrayComponentSubtract(mlp.layers.at(l).weightMatrix, layerWeightGradients.at(l), lr);
                arrayComponentSubtract(mlp.layers.at(l).biasVector, layerBiasGradients.at(l), lr);
            }
            std::cout << "Total loss : " << loss << std::endl;
        }
    }
    
    
}

void positionalEncoding(float x, float y, std::vector<float>& out) {

    constexpr const float PI = 3.14159265358979323846f;

    for (unsigned i = 0; i < 8; i++) {
        out.push_back(sinf(powf(2.0f, i) * PI * x));
        out.push_back(cosf(powf(2.0f, i) * PI * x));
    }
    for (unsigned i = 0; i < 8; i++) {
        out.push_back(sinf(powf(2.0f, i) * PI * y));
        out.push_back(cosf(powf(2.0f, i) * PI * y));
    }

}

std::vector<Datapoint> createDataset(const char* data, unsigned width, unsigned height, unsigned batchSizeX, unsigned batchSizeY) {
    assert(data);
    std::vector<Datapoint> result;
    unsigned blockWidth = width / batchSizeX;
    unsigned blockHeight = height / batchSizeY;

    for (unsigned bx = 0; bx < batchSizeX; bx++) {
        for (unsigned by = 0; by < batchSizeY; by++) {
            for (unsigned x = 0; x < blockWidth; x++) {
                for (unsigned y = 0; y < blockHeight; y++) {
                    unsigned imageIndexX = bx * blockWidth + x;
                    unsigned imageIndexY = by * blockHeight + y;
                    unsigned imageIndex = getIndex(imageIndexY, imageIndexX, width);
                    float xFloat = (static_cast<float>(imageIndexX) - (width / 2.0f)) / (width / 2.0f);
                    float yFloat = (static_cast<float>(imageIndexY) - (height / 2.0f)) / (height / 2.0f);
                    std::vector<float> input;
                    input.push_back(xFloat);
                    input.push_back(yFloat);
                    unsigned char pixelValByte = static_cast<unsigned char>(data[imageIndex]);
                    float pixelVal = pixelValByte / 255.0f;
                    result.push_back(std::make_pair(input, pixelVal));
                }
            }
        }
    }

    return result;
}

std::vector<Datapoint> createDataset32(const char* data, unsigned width, unsigned height, unsigned batchSizeX, unsigned batchSizeY) {
    assert(data);
    std::vector<Datapoint> result;
    unsigned blockWidth = width / batchSizeX;
    unsigned blockHeight = height / batchSizeY;

    for (unsigned bx = 0; bx < batchSizeX; bx++) {
        for (unsigned by = 0; by < batchSizeY; by++) {
            for (unsigned x = 0; x < blockWidth; x++) {
                for (unsigned y = 0; y < blockHeight; y++) {
                    unsigned imageIndexX = bx * blockWidth + x;
                    unsigned imageIndexY = by * blockHeight + y;
                    unsigned imageIndex = getIndex(imageIndexY, imageIndexX, width);
                    float xFloat = (static_cast<float>(imageIndexX) - (width / 2.0f)) / (width / 2.0f);
                    float yFloat = (static_cast<float>(imageIndexY) - (height / 2.0f)) / (height / 2.0f);
                    unsigned char pixelValByte = static_cast<unsigned char>(data[imageIndex]);
                    float pixelVal = pixelValByte / 255.0f;
                    std::vector<float> input;
                    positionalEncoding(xFloat, yFloat, input);
                    result.push_back(std::make_pair(input, pixelVal));
                }
            }
        }
    }

    return result;
}

char* reconstructImage(MultilayerPerceptron& mlp, unsigned width, unsigned height) {
    char* out = new char[width * height];
    std::vector<float> input;
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            input.clear();
            unsigned imageIndex = getIndex(r, c, width);
            float xFloat = (static_cast<float>(c) - (width / 2.0f)) / (width / 2.0f);
            float yFloat = (static_cast<float>(r) - (height / 2.0f)) / (height / 2.0f);
            positionalEncoding(xFloat, yFloat, input);
            auto inferences = getInference(input, mlp);
            unsigned char pixelVal = static_cast<unsigned char> (std::roundf(std::min(inferences.back().a.at(0) * 255.0f, 255.0f)));
            out[imageIndex] = static_cast<char>(pixelVal);
        }
    }

    return out;
}
