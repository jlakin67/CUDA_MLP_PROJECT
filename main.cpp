#include <vector>
#include <cassert>
#include "util.h"
#include "mlp.hpp"

constexpr const unsigned int INPUT_SIZE = 32;
constexpr const unsigned int OUTPUT_SIZE = 1;

int main() {
    MultilayerPerceptron mlp;
    
    mlp.init(3);
    mlp.layers.at(0).init(64, INPUT_SIZE);
    mlp.layers.at(0).activationFunction = ACTIVATION_FUNCTION_RELU;
    mlp.layers.at(1).init(64, 64);
    mlp.layers.at(1).activationFunction = ACTIVATION_FUNCTION_RELU;
    mlp.layers.at(2).init(OUTPUT_SIZE, 64);
    mlp.layers.at(2).activationFunction = ACTIVATION_FUNCTION_SIGMOID;
    /*
    mlp.init(7);
    mlp.layers.at(0).init(64, INPUT_SIZE);
    mlp.layers.at(0).activationFunction = ACTIVATION_FUNCTION_RELU;
    mlp.layers.at(1).init(64, 64);
    mlp.layers.at(1).activationFunction = ACTIVATION_FUNCTION_RELU;
    mlp.layers.at(2).init(128, 64);
    mlp.layers.at(2).activationFunction = ACTIVATION_FUNCTION_RELU;
    mlp.layers.at(3).init(64, 128);
    mlp.layers.at(3).activationFunction = ACTIVATION_FUNCTION_RELU;
    mlp.layers.at(4).init(32, 64);
    mlp.layers.at(4).activationFunction = ACTIVATION_FUNCTION_RELU;
    mlp.layers.at(5).init(16, 32);
    mlp.layers.at(5).activationFunction = ACTIVATION_FUNCTION_RELU;
    mlp.layers.at(6).init(1, 16);
    mlp.layers.at(6).activationFunction = ACTIVATION_FUNCTION_SIGMOID;
    */
    
    //std::vector<float> testInput;
    //testInput.resize(INPUT_SIZE, 0.5f);
    //auto inferences = getInference(testInput, 0.5f, mlp);
    //std::cout << "Output: " << inferences.back().a.at(0) << ", Loss: " << loss1D(inferences.back().a.at(0), 0.2f) << std::endl;

    int width = 0, height = 0, maxValue = 0;
    char* image = loadPGM("./feep.pgm", width, height, maxValue);
    std::vector<Datapoint> imageData = createDataset32(image, width, height, width, height);
    assert(imageData.size() == (width * height));
    sgd(mlp, imageData, 0.05f, width*height, 1000);
    char* out = reconstructImage(mlp, width, height);
    writePGM("out.pgm", width, height, maxValue, out);
}
