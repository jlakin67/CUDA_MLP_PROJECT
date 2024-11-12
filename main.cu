#include <cuda.h>
#include "mlp.hpp"

struct DeviceLayer {
	unsigned int weightMatrixRows = 0;
	unsigned int weightMatrixCols = 0;
	int activationFunction = 0;
	float* deviceWeightMatrix;
	float* deviceBiasVector;
};

struct DeviceInference {
	float* z;
	float* a;
};

DeviceLayer* allocateDeviceLayers(MultilayerPerceptron& mlp, DeviceLayer** hostArray) {
	assert(hostArray);
	DeviceLayer* deviceLayerArray = nullptr;
	cudaMalloc(&deviceLayerArray, mlp.numLayers * sizeof(DeviceLayer));
	DeviceLayer* deviceLayersHost = new DeviceLayer[mlp.numLayers];
	for (unsigned i = 0; i < mlp.numLayers; i++) {
		Layer& layer = mlp.layers.at(i);
		deviceLayersHost[i].activationFunction = layer.activationFunction;
		deviceLayersHost[i].weightMatrixRows = layer.weightMatrixRows;
		deviceLayersHost[i].weightMatrixCols = layer.weightMatrixCols;
		cudaMalloc(&deviceLayersHost[i].deviceWeightMatrix, layer.weightMatrixRows * layer.weightMatrixCols * sizeof(float));
		cudaMalloc(&deviceLayersHost[i].deviceBiasVector, layer.weightMatrixRows * sizeof(float));
		cudaMemcpy(deviceLayersHost[i].deviceWeightMatrix, layer.weightMatrix.data(),
			layer.weightMatrixRows * layer.weightMatrixCols * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceLayersHost[i].deviceBiasVector, layer.biasVector.data(),
			layer.weightMatrixRows * sizeof(float), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(deviceLayerArray, deviceLayersHost, mlp.numLayers * sizeof(DeviceLayer), cudaMemcpyHostToDevice);
	*hostArray = deviceLayersHost;
	return deviceLayerArray;
}

void retrieveLayersFromDevice(MultilayerPerceptron& mlp, DeviceLayer* hostDeviceLayers) {
	assert(hostDeviceLayers);
	for (unsigned i = 0; i < mlp.numLayers; i++) {
		Layer& layer = mlp.layers.at(i);
		cudaMemcpy(&layer.weightMatrix[0], hostDeviceLayers[i].deviceWeightMatrix,
			layer.weightMatrixRows * layer.weightMatrixCols * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&layer.biasVector[0], hostDeviceLayers[i].deviceBiasVector,
			layer.weightMatrixRows * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(hostDeviceLayers[i].deviceWeightMatrix);
		cudaFree(hostDeviceLayers[i].deviceBiasVector);
	}
}

__device__ unsigned getIndexDevice(unsigned r, unsigned c, unsigned cols) {
	return r * cols + c;
}

__device__ float loss1DDevice(float approx, float y) {
	float temp = approx - y;
	return 0.5f * temp * temp;
}

__device__ float loss1DDerivativeDevice(float approx, float y) {
	return approx - y;
};

__device__ float sigmoidDevice(float x) {
	return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoidDerivativeDevice(float x) {
	return sigmoidDevice(x) * sigmoidDevice(1.0f - x);
}

__device__ float reluDerivativeDevice(float x) {
	return (x < 0) ? 0.0f : 1.0f;
}

//assume blockDim = (64, 1, 1)
__global__ void getInferenceKernel(unsigned int numLayers, DeviceLayer* layers, unsigned inputSize, float* input, float* result) {
	__shared__ float a[64];
	
	if (threadIdx.x < inputSize) {
		a[threadIdx.x] = input[blockIdx.x*blockDim.x+threadIdx.x];
	}
	__syncthreads();

	for (unsigned l = 0; l < numLayers; l++) {

		float out = 0.f;
		if (threadIdx.x < layers[l].weightMatrixRows) {
			for (unsigned int c = 0; c < layers[l].weightMatrixCols; c++) {
				unsigned index = getIndexDevice(threadIdx.x, c, layers[l].weightMatrixCols);
				out = out + a[c] * layers[l].deviceWeightMatrix[index];
				out += layers[l].deviceBiasVector[threadIdx.x];
			}
			if (layers[l].activationFunction == ACTIVATION_FUNCTION_RELU) {
				out = fminf(0.f, out);
			}
			else if (layers[l].activationFunction == ACTIVATION_FUNCTION_SIGMOID) {
				out = sigmoidDevice(out);
			}
		}
		__syncthreads();
		a[threadIdx.x] = out;
		__syncthreads();

	}
	if (threadIdx.x == 0) result[blockIdx.x] = a[threadIdx.x];
}

constexpr const unsigned int MAX_BLOCK_SIZE = 1024;

//Because of the sequential update rule of stochastic gradient descent
//where wi+1 = wi - grad(f, wi), we need to do communication between threads.
//Unfortunately that means we can't do it in a massively parallel manner in one kernel, so
//we are restricted to 1 block or we have to split the process into multiple kernels.

//First attempt: This can only do a batch size of 1 which isn't very good
__global__ void sgdKernel(unsigned int numLabels, unsigned int inputSize, float* input, float* labels,
	unsigned int numLayers, DeviceLayer* layers, DeviceInference* temp,
	float lr, unsigned int numEpochs) {
	__shared__ float a[64]; //in backprop this will be de_db
	__shared__ float b[64 * 64]; //in backprop this will be de_dw

	for (unsigned i = 0; i < numEpochs; i++) {
		unsigned dataIndex = 0;
		float loss = 0.f;
		while (dataIndex < numLabels) {

			//first get inference
			if (threadIdx.x < inputSize) {
				a[threadIdx.x] = input[inputSize * dataIndex + threadIdx.x];
			}
			__syncthreads();

			for (unsigned l = 0; l < numLayers; l++) {

				float out = 0.f;
				if (threadIdx.x < layers[l].weightMatrixRows) {
					for (unsigned int c = 0; c < layers[l].weightMatrixCols; c++) {
						unsigned index = getIndexDevice(threadIdx.x, c, layers[l].weightMatrixCols);
						out = out + a[c] * layers[l].deviceWeightMatrix[index];
						out += layers[l].deviceBiasVector[threadIdx.x];
					}
					temp[l].z[threadIdx.x] = out;
					if (layers[l].activationFunction == ACTIVATION_FUNCTION_RELU) {
						out = fminf(0.f, out);
					}
					else if (layers[l].activationFunction == ACTIVATION_FUNCTION_SIGMOID) {
						out = sigmoidDevice(out);
					}
					temp[l].a[threadIdx.x] = out;
				}
				__syncthreads();
				a[threadIdx.x] = out;
			}

			if (threadIdx.x == 0) loss += loss1DDevice(a[0], labels[dataIndex]);

			//do backpropagation

			//last layer
			float de_da_last = loss1DDerivativeDevice(a[0], labels[dataIndex]);
			float df_dz_last;
			if (layers[numLayers - 1].activationFunction == ACTIVATION_FUNCTION_RELU) {
				df_dz_last = reluDerivativeDevice(temp[numLayers - 1].z[0]);
			}
			else {
				df_dz_last = sigmoidDerivativeDevice(temp[numLayers - 1].z[0]);
			}
			if (threadIdx.x == 0) {
				a[0] = de_da_last * df_dz_last;
				layers[numLayers - 1].deviceBiasVector[0] -= lr * a[0];
			}
			if (threadIdx.x < layers[numLayers - 2].weightMatrixRows) {
				b[threadIdx.x] = de_da_last * df_dz_last * temp[numLayers - 2].a[threadIdx.x];
			}
			__syncthreads();

			//in between layers
			for (int l = numLayers - 2; l >=0 ; l--) {
				//calculate de_dbl and update bl
				if (threadIdx.x < layers[l].weightMatrixRows) {
					float de_da = 0.f;
					for (unsigned r = 0; r < layers[l + 1].weightMatrixRows; r++) {
						unsigned index = getIndexDevice(r, threadIdx.x, layers[l + 1].weightMatrixCols);
						de_da += layers[l + 1].deviceWeightMatrix[index] * a[r];
					}
					float df_dz;
					if (layers[l].activationFunction == ACTIVATION_FUNCTION_RELU) {
						df_dz = reluDerivativeDevice(temp[l].z[threadIdx.x]);
					}
					else {
						df_dz = sigmoidDerivativeDevice(temp[l].z[threadIdx.x]);
					}
					a[threadIdx.x] = de_da * df_dz;
					layers[l].deviceBiasVector[threadIdx.x] -= lr * a[threadIdx.x];
				}
				//update de_dwl+1
				if (threadIdx.x < layers[l + 1].weightMatrixRows) {
					for (unsigned c = 0; c < layers[l + 1].weightMatrixCols; c++) {
						unsigned index = getIndexDevice(threadIdx.x, c, layers[l + 1].weightMatrixCols);
						layers[l + 1].deviceWeightMatrix[index] -= lr * b[index];
					}
				}
				__syncthreads();
				//calculate de_dwl
				if (threadIdx.x < layers[l].weightMatrixRows) {
					for (unsigned c = 0; c < layers[l].weightMatrixCols; c++) {
						unsigned index = getIndexDevice(threadIdx.x, c, layers[l].weightMatrixCols);
						if (l == 0) {
							b[index] = a[threadIdx.x] * input[inputSize * dataIndex + c];
						}
						else {
							b[index] = a[threadIdx.x] * temp[l - 1].a[c];
						}
					}
				}
				__syncthreads();
			}
			//update de_dwl0
			if (threadIdx.x < layers[0].weightMatrixRows) {
				for (unsigned c = 0; c < layers[0].weightMatrixCols; c++) {
					unsigned index = getIndexDevice(threadIdx.x, c, layers[0].weightMatrixCols);
					layers[0].deviceWeightMatrix[index] -= lr * b[index];
				}
			}
			__syncthreads();

			dataIndex++;

		}
		if (threadIdx.x == 0) printf("Epoch %d, Total Loss %f\n", i, loss);
	}
}

//Second attempt: Do backpropagation in two parts, this kernel will calculate gradients over batch.
//This one is much faster but consumes a lot of GPU memory for storing calculations
//gridDim.x == batchSize, blockIdx.x == indexInBatch
//blockDim.x == 64
__global__ void batchGradientKernel(unsigned startIndex, unsigned int inputSize, float input[], float labels[],
	unsigned int numLayers, DeviceLayer layers[], DeviceLayer gradients[], DeviceInference temp[], float losses[]) {
	__shared__ float a[64]; //in backprop this will be de_db
	__shared__ float b[64 * 64]; //in backprop this will be de_dw

	//first get inference
	if (threadIdx.x < inputSize) {
		a[threadIdx.x] = input[inputSize * (startIndex + blockIdx.x) + threadIdx.x];
	}
	__syncthreads();

	for (unsigned l = 0; l < numLayers; l++) {

		float out = 0.f;
		if (threadIdx.x < layers[l].weightMatrixRows) {
			for (unsigned int c = 0; c < layers[l].weightMatrixCols; c++) {
				unsigned index = getIndexDevice(threadIdx.x, c, layers[l].weightMatrixCols);
				out = out + a[c] * layers[l].deviceWeightMatrix[index];
				out += layers[l].deviceBiasVector[threadIdx.x];
			}
			temp[blockIdx.x*numLayers + l].z[threadIdx.x] = out;
			if (layers[l].activationFunction == ACTIVATION_FUNCTION_RELU) {
				out = fminf(0.f, out);
			}
			else if (layers[l].activationFunction == ACTIVATION_FUNCTION_SIGMOID) {
				out = sigmoidDevice(out);
			}
			temp[blockIdx.x * numLayers + l].a[threadIdx.x] = out;
		}
		__syncthreads();
		a[threadIdx.x] = out;
		__syncthreads();
	}

	if (threadIdx.x == 0) losses[blockIdx.x] = loss1DDevice(a[0], labels[startIndex+ blockIdx.x]);
	__syncthreads();

	//do backpropagation

	//last layer
	int layerIndex = numLayers - 1;
	float de_da_last = loss1DDerivativeDevice(a[0], labels[startIndex+blockIdx.x]);
	float df_dz_last;
	if (layers[layerIndex].activationFunction == ACTIVATION_FUNCTION_RELU) {
		df_dz_last = reluDerivativeDevice(temp[blockIdx.x * numLayers + layerIndex].z[0]);
	}
	else {
		df_dz_last = sigmoidDerivativeDevice(temp[blockIdx.x * numLayers + layerIndex].z[0]);
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		a[0] = de_da_last * df_dz_last;
		gradients[blockIdx.x * numLayers + layerIndex].deviceBiasVector[0] = a[0];
	}
	if (threadIdx.x < layers[layerIndex - 1].weightMatrixRows) {
		b[threadIdx.x] = de_da_last * df_dz_last * temp[blockIdx.x * numLayers + layerIndex - 1].a[threadIdx.x];
		gradients[blockIdx.x * numLayers + layerIndex].deviceWeightMatrix[threadIdx.x] = b[threadIdx.x];
	}
	__syncthreads();

	//remaining layers
	layerIndex--;
	while (layerIndex >= 0) {
		//calculate de_dbl
		float out = 0.f;
		if (threadIdx.x < layers[layerIndex].weightMatrixRows) {
			float de_da = 0.f;
			for (unsigned r = 0; r < layers[layerIndex + 1].weightMatrixRows; r++) {
				unsigned index = getIndexDevice(r, threadIdx.x, layers[layerIndex + 1].weightMatrixCols);
				de_da += layers[layerIndex + 1].deviceWeightMatrix[index] * a[r];
			}
			float df_dz;
			if (layers[layerIndex].activationFunction == ACTIVATION_FUNCTION_RELU) {
				df_dz = reluDerivativeDevice(temp[blockIdx.x * numLayers + layerIndex].z[threadIdx.x]);
			}
			else {
				df_dz = sigmoidDerivativeDevice(temp[blockIdx.x * numLayers + layerIndex].z[threadIdx.x]);
			}
			out = de_da * df_dz;
		}
		__syncthreads();
		a[threadIdx.x] = out;
		if (threadIdx.x < layers[layerIndex].weightMatrixRows) 
			gradients[blockIdx.x * numLayers + layerIndex].deviceBiasVector[threadIdx.x] = a[threadIdx.x];
		__syncthreads();
		//calculate de_dwl
		if (threadIdx.x < layers[layerIndex].weightMatrixRows) {
			for (unsigned c = 0; c < layers[layerIndex].weightMatrixCols; c++) {
				unsigned index = getIndexDevice(threadIdx.x, c, layers[layerIndex].weightMatrixCols);
				if (layerIndex == 0) {
					b[index] = a[threadIdx.x] * input[inputSize * (startIndex + blockIdx.x) + c];
				}
				else {
					b[index] = a[threadIdx.x] * temp[blockIdx.x * numLayers + layerIndex - 1].a[c];
				}
				gradients[blockIdx.x * numLayers + layerIndex].deviceWeightMatrix[index] = b[index];
			}
		}
		__syncthreads();
		layerIndex--;
	}

}

__device__ void parallelReduce(unsigned len, float arr[], int threadIndex) {
	unsigned currentLen = len;
	unsigned index = 2 * threadIndex;
	while (currentLen > 1) {
		float sum = 0.f;
		if (index < currentLen) sum = arr[index + 1] + arr[index];
		__syncthreads();
		if (index < currentLen) arr[index / 2] = sum;
		__syncthreads();
		currentLen /= 2;
	}
}

//Second attempt: Do backpropagation in two parts, the last part is to compute sum of gradients in parallel reduction then update
// blockDim.x = 128 = maxBatchSize/2
//numBlocks = 1
__global__ void gradientReduceBatch(unsigned int numLayers, unsigned int batchSize, 
	float losses[], DeviceLayer layers[], DeviceLayer gradients[], float lr, unsigned epoch, unsigned batchNumber) {
	__shared__ float a[256];
	unsigned int inputIndex = 2 * threadIdx.x;
	if (inputIndex < batchSize) a[inputIndex] = losses[inputIndex];
	else a[inputIndex] = 0.f;
	if (inputIndex + 1 < batchSize) a[inputIndex + 1] = losses[inputIndex + 1];
	else a[inputIndex + 1] = 0.f;
	__syncthreads();
	parallelReduce(256, a, threadIdx.x);
	if (threadIdx.x == 0) losses[0] = a[0];
	__syncthreads();
	for (unsigned int l = 0; l < numLayers; l++) {
		for (unsigned int r = 0; r < layers[l].weightMatrixRows; r++) {
			if (inputIndex < batchSize) a[inputIndex] = gradients[inputIndex * numLayers + l].deviceBiasVector[r];
			else a[inputIndex] = 0.f;
			if (inputIndex + 1 < batchSize) a[inputIndex+1] = gradients[(inputIndex + 1) * numLayers + l].deviceBiasVector[r];
			else a[inputIndex+1] = 0.f;
			__syncthreads();
			parallelReduce(256, a, threadIdx.x);
			if (threadIdx.x == 0) layers[l].deviceBiasVector[r] -= lr * (1.0f / (2.0f * batchSize)) * a[0];
			__syncthreads();
			for (unsigned int c = 0; c < layers[l].weightMatrixCols; c++) {
				unsigned matrixIndex = getIndexDevice(r, c, layers[l].weightMatrixCols);
				if (inputIndex < batchSize) a[inputIndex] = gradients[inputIndex * numLayers + l].deviceWeightMatrix[matrixIndex];
				else a[inputIndex] = 0.f;
				if (inputIndex + 1 < batchSize) a[inputIndex + 1] = gradients[(inputIndex + 1) * numLayers + l].deviceWeightMatrix[matrixIndex];
				else a[inputIndex + 1] = 0.f;
				__syncthreads();
				parallelReduce(256, a, threadIdx.x);
				if (threadIdx.x == 0) layers[l].deviceWeightMatrix[matrixIndex] -= lr * (1.0f / (2.0f * batchSize)) * a[0];
				__syncthreads();
			}
		}
	}
	if (threadIdx.x == 0) printf("Epoch %d, Batch number %d, Loss: %f\n", epoch, batchNumber, losses[0]);
}

void sgdDevice(unsigned int numLabels, unsigned int inputSize, float* inputs, float* labels, float lr, unsigned numEpochs,
	int numLayers, DeviceLayer* layers, DeviceLayer* hostLayers) {
	DeviceInference* temp = new DeviceInference[numLayers];
	for (int i = 0; i < numLayers; i++) {
		cudaMalloc(&temp[i].z, hostLayers[i].weightMatrixRows * sizeof(float));
		cudaMalloc(&temp[i].a, hostLayers[i].weightMatrixRows * sizeof(float));
	}
	DeviceInference* tempDevice = nullptr;
	cudaMalloc(&tempDevice, numLayers * sizeof(DeviceInference));
	cudaMemcpy(tempDevice, temp, numLayers * sizeof(DeviceInference), cudaMemcpyHostToDevice);
	sgdKernel << <1, 64 >> > (numLabels, inputSize, inputs, labels,
		numLayers, layers, tempDevice,
		lr, numEpochs);
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess) {
		std::cout << "sgd kernel error: " << cudaGetErrorString(cudaerr) << std::endl;
	}

	for (int i = 0; i < numLayers; i++) {
		cudaFree(temp[i].a);
		cudaFree(temp[i].z);
	}
	cudaFree(tempDevice);
}

//batches of size 256
void sgdDevice2 (unsigned int numLabels, unsigned int inputSize, float* inputs, float* labels, float lr, unsigned numEpochs,
	int numLayers, DeviceLayer* layers, DeviceLayer* hostLayers) {

	DeviceInference* temp = new DeviceInference[numLayers*256];
	for (unsigned i = 0; i < 256; i++) {
		for (int j = 0; j < numLayers; j++) {
			cudaMalloc(&temp[i*numLayers+j].z, hostLayers[j].weightMatrixRows * sizeof(float));
			cudaMalloc(&temp[i*numLayers+j].a, hostLayers[j].weightMatrixRows * sizeof(float));
		}
	}
	DeviceLayer* gradients = new DeviceLayer[numLayers * 256];
	for (unsigned i = 0; i < 256; i++) {
		for (int j = 0; j < numLayers; j++) {
			gradients[i * numLayers + j].weightMatrixRows = hostLayers[j].weightMatrixRows;
			gradients[i * numLayers + j].weightMatrixCols = hostLayers[j].weightMatrixCols;
			cudaMalloc(&gradients[i * numLayers + j].deviceWeightMatrix, hostLayers[j].weightMatrixCols * hostLayers[j].weightMatrixRows * sizeof(float));
			cudaMalloc(&gradients[i * numLayers + j].deviceBiasVector, hostLayers[j].weightMatrixRows * sizeof(float));
		}
	}
	DeviceInference* tempDevice = nullptr;
	DeviceLayer* gradientsDevice = nullptr;
	float* lossesDevice = nullptr;
	cudaMalloc(&tempDevice, numLayers* 256 * sizeof(DeviceInference));
	cudaMemcpy(tempDevice, temp, numLayers* 256 * sizeof(DeviceInference), cudaMemcpyHostToDevice);
	cudaMalloc(&gradientsDevice, numLayers * 256 * sizeof(DeviceLayer));
	cudaMemcpy(gradientsDevice, gradients, numLayers * 256 * sizeof(DeviceLayer), cudaMemcpyHostToDevice);
	cudaMalloc(&lossesDevice, 256 * sizeof(float));
	
	for (unsigned i = 0; i < numEpochs; i++) {
		unsigned currentBatch = 0;
		unsigned dataIndex = 0;
		while (dataIndex < numLabels) {
			unsigned currentBatchSize = std::min(256u, numLabels - dataIndex);

			//only using default cuda stream, so kernel calls are synchronized
			batchGradientKernel<<<currentBatchSize, 64>>> (dataIndex, inputSize, inputs, labels, numLayers, layers, gradientsDevice, tempDevice, lossesDevice);
			gradientReduceBatch <<<1, 128>>> (numLayers, currentBatchSize, lossesDevice, layers, gradientsDevice, lr, i, currentBatch);

			cudaError_t cudaerr = cudaDeviceSynchronize();
			if (cudaerr != cudaSuccess) {
				std::cout << "sgd kernel error: " << cudaGetErrorString(cudaerr) << std::endl;
			}

			dataIndex += 256;
			currentBatch++;
		}
	}

	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess) {
		std::cout << "sgd kernel error: " << cudaGetErrorString(cudaerr) << std::endl;
	}

	for (unsigned i = 0; i < 256; i++) {
		for (int j = 0; j < numLayers; j++) {
			cudaFree(temp[i * numLayers + j].z);
			cudaFree(temp[i * numLayers + j].a);
		}
	}
	for (unsigned i = 0; i < 256; i++) {
		for (int j = 0; j < numLayers; j++) {
			cudaFree(gradients[i * numLayers + j].deviceWeightMatrix);
			cudaFree(gradients[i * numLayers + j].deviceBiasVector);
		}
	}
	cudaFree(lossesDevice);
	delete[] gradients;
	delete[] temp;
}

constexpr const unsigned int INPUT_SIZE = 32;
constexpr const unsigned int OUTPUT_SIZE = 1;

float* getInferenceDevice(unsigned inputSize, unsigned numInputs, float* inputs, unsigned int numLayers, DeviceLayer* layers) {
	float* resultDevice = nullptr;
	float* result = new float[numInputs];
	float* inputBuffer = nullptr;
	cudaMalloc(&inputBuffer, inputSize * numInputs * sizeof(float));
	cudaMemcpy(inputBuffer, inputs, inputSize * numInputs * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&resultDevice, numInputs*sizeof(float));
	getInferenceKernel<<<numInputs, 64>>> (numLayers, layers, inputSize, inputBuffer, resultDevice);
	cudaDeviceSynchronize();
	cudaMemcpy(result, resultDevice, numInputs * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(inputBuffer);
	cudaFree(resultDevice);
	return result;
}

void createDatasetDevice(const char* image, unsigned width, unsigned height, float** inputs, float** labels) {
	assert(inputs);
	assert(labels);

	float* inputsPtr = new float[width * height * INPUT_SIZE];
	float* labelsPtr = new float[width * height];
	std::vector<float> encoding;
	for (unsigned x = 0; x < width; x++) {
		for (unsigned y = 0; y < height; y++) {
			encoding.clear();
			unsigned imageIndex = getIndex(y, x, width);
			float xFloat = (static_cast<float>(x) - (width / 2.0f)) / (width / 2.0f);
			float yFloat = (static_cast<float>(y) - (height / 2.0f)) / (height / 2.0f);
			unsigned char pixelValByte = static_cast<unsigned char>(image[imageIndex]);
			float pixelVal = pixelValByte / 255.0f;
			labelsPtr[imageIndex] = pixelVal;
			positionalEncoding(xFloat, yFloat, encoding);
			for (int i = 0; i < INPUT_SIZE; i++) {
				inputsPtr[(imageIndex * INPUT_SIZE) + i] = encoding.at(i);
			}
		}
	}
	cudaMalloc(inputs, width * height * INPUT_SIZE * sizeof(float));
	cudaMalloc(labels, width * height * sizeof(float));
	cudaMemcpy(*inputs, inputsPtr, width * height * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(*labels, labelsPtr, width * height * sizeof(float), cudaMemcpyHostToDevice);
}

char* reconstructImageDevice(unsigned numLayer, DeviceLayer* layers, unsigned width, unsigned height) {
	//float* getInferenceDevice(unsigned inputSize, unsigned numInputs, float* inputs, unsigned int numLayers, DeviceLayer * layers)
	char* out = new char[width * height];
	float* inputHost = new float[width * height * 32];
	std::vector<float> input;
	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			input.clear();
			unsigned imageIndex = getIndex(r, c, width);
			float xFloat = (static_cast<float>(c) - (width / 2.0f)) / (width / 2.0f);
			float yFloat = (static_cast<float>(r) - (height / 2.0f)) / (height / 2.0f);
			positionalEncoding(xFloat, yFloat, input);
			for (int i = 0; i < 32; i++) {
				inputHost[(imageIndex * INPUT_SIZE) + i] = input.at(i);
			}
		}
	}
	float* output = getInferenceDevice(32, width * height, inputHost, 3, layers);

	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			unsigned imageIndex = getIndex(r, c, width);
			unsigned char pixelVal = static_cast<unsigned char> (std::roundf(std::min(output[imageIndex] * 255.0f, 255.0f)));
			out[imageIndex] = static_cast<char>(pixelVal);
		}
	}


	return out;
}

int main(int argc, char* argv[]) {
	MultilayerPerceptron mlp;
	DeviceLayer* hostLayers = nullptr;
	mlp.init(3);
	mlp.layers.at(0).init(64, INPUT_SIZE);
	mlp.layers.at(0).activationFunction = ACTIVATION_FUNCTION_RELU;
	mlp.layers.at(1).init(64, 64);
	mlp.layers.at(1).activationFunction = ACTIVATION_FUNCTION_RELU;
	mlp.layers.at(2).init(OUTPUT_SIZE, 64);
	mlp.layers.at(2).activationFunction = ACTIVATION_FUNCTION_SIGMOID;

	DeviceLayer* deviceLayers = allocateDeviceLayers(mlp, &hostLayers);
	int width = 0, height = 0, maxValue = 0;
	char* image = loadPGM("./feep.pgm", width, height, maxValue);
	float* inputs = nullptr;
	float* labels = nullptr;
	createDatasetDevice(image, width, height, &inputs, &labels);
	//std::vector<Datapoint> imageData = createDataset32(image, width, height, width, height);
	//assert(imageData.size() == (width * height));
	//sgd(mlp, imageData, 1.0f, width * height, 10);
	sgdDevice2(width * height, INPUT_SIZE, inputs, labels, 0.01f, 100, mlp.numLayers, deviceLayers, hostLayers);
	
	char* out = reconstructImageDevice(mlp.numLayers, deviceLayers, width, height);
	//char* out = reconstructImage(mlp, width, height);
	writePGM("out.pgm", width, height, maxValue, out);
	delete[] out;

	return 0;
}