#include <iostream>
#include <cstdlib>
#include "NeuralNetwork.cudah"
#include "Random.cu"
#include <chrono>

__global__ void feedForwardLayer(float * values, float * biases, float * weights, int * dims, int * sumDims, int * sumWeights, int layer) {
    /* This gets the 'index' of the thread. The thread id is the id of the thread in the thread block
       The blockDim is the amount of threads per block and the block id is the id of the block this thread is in
       This basically gives the index of the thread so you can use it to access memory.
    */
    const int neuron = blockDim.x * blockIdx.x + threadIdx.x;

    // If the thread index is less than the size of the array.
    if(neuron < dims[layer]) {
        for(int subNeuron = sumDims[layer-1]; subNeuron < sumDims[layer]; subNeuron++) {
            values[sumDims[layer] + neuron] += values[subNeuron] * weights[sumWeights[layer-1] + dims[layer]*(subNeuron-sumDims[layer-1]) + neuron];
        }
        values[sumDims[layer] + neuron] += biases[sumDims[layer] + neuron];
    }
};

__global__ void getBackPropDeltas(float * deltaBiases, float * deltaWeights, float * values, float * weights, int * dims, int * sumDims, int * sumWeights, int layer) {
    /* This gets the 'index' of the thread. The thread id is the id of the thread in the thread block
       The blockDim is the amount of threads per block and the block id is the id of the block this thread is in
       This basically gives the index of the thread so you can use it to access memory.
    */
    // In this case, the neuron is the same value as the neuron in the iterator on the CPU
    int neuron = blockDim.x * blockIdx.x + threadIdx.x;

    // If the thread index is less than the size of the array.
    if(neuron < dims[layer]) {
        deltaBiases[neuron] = 0;
        for(int subNeuron = 0; subNeuron < dims[layer-1]; subNeuron++) {
            deltaWeights[sumWeights[layer-1] + dims[layer] * (subNeuron) + neuron] = values[sumDims[layer-1] + subNeuron] * deltaBiases[sumDims[layer] + neuron];
        }
    }
    // The reason for the separation is because you don't want multiple threads to change the same value at the same time.
    int subNeuron = neuron;
    if(subNeuron < dims[layer-1]) {
        for(int neuron = 0; neuron < dims[layer]; neuron++) {
            deltaBiases[sumDims[layer-1] + subNeuron] += weights[sumWeights[layer-1] + dims[layer] * (subNeuron) + neuron] * deltaBiases[sumDims[layer] + neuron];
        }
    }
};

__global__ void applyDeltas(float * deltaBiases, float * biases, float * deltaWeights, float * weights, int totalNNSize, int totalWeights, float learningRate) {
    /* This gets the 'index' of the thread. The thread id is the id of the thread in the thread block
       The blockDim is the amount of threads per block and the block id is the id of the block this thread is in
       This basically gives the index of the thread so you can use it to access memory.
    */
    int index = blockDim.x * blockIdx.x + threadIdx.x;


    if(index < totalWeights) {
        weights[index] += deltaWeights[index] * learningRate;
    }
    // If the thread index is less than the size of the array.
    if(index < totalNNSize) {
        biases[index] += deltaBiases[index] * learningRate;
    }
};

Network::Network() {
    read();
};

Network::Network(int dimsLen, int * neuronCountList, float learningRateInput) {
    // Set the learning rate
    learningRate = learningRateInput;
    // Sets dims and dimsLength
    dimsLength = dimsLen;
    dims = (int *) malloc(dimsLength * sizeof(int));
    for(int layer = 0; layer < dimsLength; layer++) {
        dims[layer] = neuronCountList[layer];
        std::<< "Dims[" << layer << "]: " << dims[layer] << std::endl;
    }
    printf("Dimsptr: %p\n", dims);
    initAll();
};

void Network::initAll() {
    // Gets the total size of the network.
    totalNNSize = 0;
    for(int layer = 0; layer < dimsLength; layer++) {
        totalNNSize += dims[layer];
    }
    // Gets the total amount of weights, and the amount of weights per layer
    weightDims = (int *) calloc(dimsLength, sizeof(int));
    weightDims[dimsLength-1] = 0;
    totalWeights = 0;
    for(int layer = 0; layer < dimsLength-1; layer++) {
        weightDims[layer] = dims[layer] * dims[layer+1];
        totalWeights += weightDims[layer];
    }

    // Gets the amount of weights from the 0th weight for each layer
    sumWeights = (int *) malloc((dimsLength) * sizeof(int));
    sumWeights[0] = 0;
    sumWeights[1] = weightDims[0];
    for(int layer = 2; layer < dimsLength; layer++) {
        sumWeights[layer] = sumWeights[layer-1] + weightDims[layer-1];
    }

    // Gets the amount of neuron from the 0th neuron for each layer
    sumDims = (int *) malloc((dimsLength+1) * sizeof(int));
    sumDims[0] = 0;
    sumDims[1] = dims[0];
    for(int layer = 2; layer < dimsLength+1; layer++) {
        sumDims[layer] = sumDims[layer-1] + dims[layer-1];
    }

    // Creates a new, empty value array.
    values = (float *) calloc(totalNNSize, sizeof(float));
    // Makes the delta variables for backpropagation
    deltaBiases = (float *) calloc(totalNNSize, sizeof(float));
    deltaWeights = (float *) calloc(totalWeights, sizeof(float));
    // Initializes the weights and biases with a value of 0.
    biases = (float *) calloc(totalNNSize, sizeof(float));
    weights = (float *) calloc(totalWeights, sizeof(float));
};

void Network::randomize() {
    srand(totalWeights);
    std::cout << "Randomizing biases... 0/" << totalNNSize <<std::endl;
    for(int neuron = 0; neuron < totalNNSize; neuron++) {
        biases[neuron] = (2 * (float(rand()) / (float) RAND_MAX) - 1);
    }

    std::cout << "Randomizing weights... 0/" << totalWeights <<std::endl;
    for(int weight = 0; weight < totalWeights; weight++) {
        weights[weight] = (2*rand()/RAND_MAX)-1;
    }
};

void Network::loadNetworkToGPU() {
    // These two do not need memcpy because they are initialized on the GPU.
    cudaMalloc(&deviceDeltaBiases, totalNNSize * sizeof(float));
    cudaMalloc(&deviceDeltaWeights, totalWeights * sizeof(float));

    cudaMalloc(&deviceBiases, totalNNSize * sizeof(float));
    cudaMalloc(&deviceWeights, totalWeights * sizeof(float));
    cudaMalloc(&deviceDims, dimsLength * sizeof(int));
    cudaMalloc(&deviceSumDims, (dimsLength+1) * sizeof(int));
    cudaMalloc(&deviceSumWeights, (dimsLength) * sizeof(int));
    cudaMalloc(&deviceValues, totalNNSize * sizeof(float));

    cudaMemcpy(deviceBiases, biases, totalNNSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceWeights, weights, totalWeights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDims, dims, dimsLength * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSumDims, sumDims, (dimsLength+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSumWeights, sumWeights, (dimsLength) * sizeof(int), cudaMemcpyHostToDevice);
    loaded = true;
};

float * Network::feedForwardGPU(float input[]) {
    // Creates a new, empty value array.
    for(int i = 0; i < totalNNSize; i++) {
        values[i] = 0;
    }

    // Puts the inputs into the values to be fed forward.
    for(int neuron = 0; neuron < dims[0]; neuron++) {
        values[neuron] = input[neuron];
    }

    // Allocates device memory for the values
    cudaMemcpy(deviceValues, values, totalNNSize * sizeof(float), cudaMemcpyHostToDevice);
    // For each layer, feed forward using a GPU thread instead of a for loop.
    for(int layer = 1; layer < dimsLength; layer++) {
        int dimGrid = (dims[layer] + THREADSPERBLOCK - 1)/THREADSPERBLOCK;
        feedForwardLayer <<< dimGrid, THREADSPERBLOCK >>> (deviceValues, deviceBiases, deviceWeights, deviceDims, deviceSumDims, deviceSumWeights, layer);
    }
    cudaMemcpy(values, deviceValues, totalNNSize * sizeof(float), cudaMemcpyDeviceToHost);

    float * lastLayer = (float *) malloc(sizeof(float) * dims[dimsLength-1]);
    for(int neuron = sumDims[dimsLength-1]; neuron < totalNNSize; neuron++) {
        lastLayer[neuron - sumDims[dimsLength-1]] = values[neuron];
    }
    return lastLayer;
};

void Network::backPropGPU(float expected[], float * result) {
    float * deltaBiasesTemp = (float *) calloc(totalNNSize, sizeof(float));
    for(int i = 0; i < dims[dimsLength-1]; i++) {;
        deltaBiasesTemp[sumDims[dimsLength-1] + i] = expected[i] - result[i];
    }
    cudaMemcpy(deviceDeltaBiases, deltaBiasesTemp, totalNNSize * sizeof(float), cudaMemcpyHostToDevice);
    int dimGrid;
    for(int layer = dimsLength-1; layer > 0; layer--) {
        int amountOfNeurons = dims[layer] > dims[layer-1] ? dims[layer] : dims[layer-1];
        dimGrid = (amountOfNeurons + THREADSPERBLOCK - 1)/THREADSPERBLOCK;
        getBackPropDeltas <<< dimGrid, THREADSPERBLOCK >>> (deviceDeltaBiases, deviceDeltaWeights, deviceValues, deviceWeights, deviceDims, deviceSumDims, deviceSumWeights, layer);
    }
    // dimGrid = (totalWeights + THREADSPERBLOCK - 1)/THREADSPERBLOCK;
    // applyDeltas <<< dimGrid, THREADSPERBLOCK >>> (deviceDeltaBiases, deviceBiases, deviceDeltaWeights, deviceWeights, totalNNSize, totalWeights, learningRate);
}

float * Network::feedForwardCPU(float input[]) {
    // Puts the inputs into the values to be fed forward.
    for(int neuron = 0; neuron < dims[0]; neuron++) {
        values[neuron] = input[neuron];
    }
    for(int layer = 1; layer < dimsLength; layer++) {
        // This for loop wouldn't exist if this was for the GPU since each GPU thread gets a neuron.
        for(int neuron = 0; neuron < dims[layer]; neuron++) {
            for(int subNeuron = sumDims[layer-1]; subNeuron < sumDims[layer]; subNeuron++) {
                values[sumDims[layer] + neuron] += values[subNeuron] * weights[sumWeights[layer-1] + dims[layer]*(subNeuron-sumDims[layer-1]) + neuron];
            }
            values[sumDims[layer] + neuron] += biases[sumDims[layer] + neuron];
        }
    }
    float * lastLayer = (float *) malloc(sizeof(float) * dims[dimsLength-1]);
    for(int neuron = sumDims[dimsLength-1]; neuron < totalNNSize; neuron++) {
        lastLayer[neuron - sumDims[dimsLength-1]] = values[neuron];
    }
    return lastLayer;
};

void Network::getBackPropDeltasCPU(float expected[], float * result) {
    for(int i = 0; i < dims[dimsLength-1]; i++) {
        deltaBiases[sumDims[dimsLength-1] + i] = expected[i] - result[i];
    }
    for(int layer = dimsLength-1; layer > 0; layer--) {
        // For every neuron in the layer
        for(int neuron = 0; neuron < dims[layer]; neuron++) {
            // For every neuron in the layer before
            for(int subNeuron = 0; subNeuron < dims[layer-1]; subNeuron++) {
                deltaWeights[sumWeights[layer-1] + dims[layer] * (subNeuron) + neuron] = values[sumDims[layer-1] + subNeuron] * deltaBiases[sumDims[layer] + neuron];
                deltaBiases[sumDims[layer-1] + subNeuron] += weights[sumWeights[layer-1] + dims[layer] * (subNeuron) + neuron] * deltaBiases[sumDims[layer] + neuron];
            }
        }
    }
}

void Network::backpropogateCPU(float expected[], float * result) {
    getBackPropDeltasCPU(expected, result);
    for(int neuron = 0; neuron < totalNNSize; neuron++) {
        biases[neuron] += deltaBiases[neuron] * learningRate;
    }
    for(int weight = 0; weight < totalWeights; weight++) {
        weights[weight] += deltaWeights[weight] * learningRate;
    }
}

void Network::unloadNetworkFromGPU() {
    cudaFree(deviceValues);
    cudaFree(deviceBiases);
    cudaFree(deviceWeights);
    cudaFree(deviceDims);
    cudaFree(deviceSumDims);
    cudaFree(deviceSumWeights);
    loaded = false;
};

Network::~Network() {
    free(biases);
    free(weights);
    free(dims);
    free(weightDims);
    free(sumDims);
    free(sumWeights);
    if(!loaded) {
        unloadNetworkFromGPU();
    }
};

void Network::print() {
    for(int layer = 0; layer < dimsLength; layer++) {
        std::cout << "Layer " << layer+1 << ":" << std::endl;
        for(int neuron = 0; neuron < dims[layer]; neuron++) {
            std::cout << "\tNeuron " << neuron+1 << ":" << std::endl;
            std::cout << "\t\tBias: " << biases[sumDims[layer] + neuron] << ":" << std::endl;
            std::cout << "\t\tDelta Bias: " << deltaBiases[sumDims[layer] + neuron] << ":" << std::endl;
            if(layer < (dimsLength-1)) {
                std::cout << "\t\tWeights: " << std::endl;
                for(int weight = 0; weight < dims[layer+1]; weight++) {
                    std::cout << "\t\t\tWeight " << weight+1 << ": " << weights[sumWeights[layer] + dims[layer+1]*neuron + weight] << ":" << std::endl;
                    std::cout << "\t\t\tDelta Weight " << weight+1 << ": " << deltaWeights[sumWeights[layer] + dims[layer+1]*neuron + weight] << ":" << std::endl;
                }
            }
        }
    }
};

void Network::save() {
    FILE * file = fopen("dimsLength", "wb");
    fwrite(&dimsLength, sizeof(int), 1, file);
    fclose(file);

    file = fopen("dims", "wb");
    fwrite(dims, sizeof(int), dimsLength, file);
    fclose(file);

    file = fopen("biases", "wb");
    fwrite(biases, sizeof(float), totalNNSize, file);
    fclose(file);

    file = fopen("weights", "wb");
    fwrite(weights, sizeof(float), totalWeights, file);
    fclose(file);
};

void Network::read() {
    //fread(data[i], sizeof(data[i][0]), ny, file);
    FILE * file = fopen("dimsLength", "rb");
    fread(&dimsLength, sizeof(int), 1, file);
    fclose(file);

    dims = (int *) malloc(dimsLength * sizeof(int));
    file = fopen("dims", "rb");
    fread(dims, sizeof(int), dimsLength, file);
    fclose(file);

    // This initializes the rest of the Network class so I can use totalNNSize and totalWeights in the next lines.
    initAll();
    
    file = fopen("biases", "rb");
    fread(biases, sizeof(float), totalNNSize, file);
    fclose(file);

    file = fopen("weights", "rb");
    fread(weights, sizeof(float), totalWeights, file);
    fclose(file);
};

int main() {
    int dims[] = {2, 3, 1};
    float * biases = (float *) calloc(6, sizeof(float));
    float * weights = (float *) calloc(9, sizeof(float));
    weights[0] = 0.25;
    weights[1] = 0.5;
    weights[2] = 0.75;
    weights[3] = 0.33;
    weights[4] = 0.66; 
    weights[5] = 1;
    weights[6] = 0.5;
    weights[7] = 0.5; 
    weights[8] = 0.5;
    Network test = Network(3, dims, 1);
    test.biases = biases;
    test.weights = weights;

    // Network test = Network(); // If no arguments are passed, it tries to read from a file.
    float passIn[] = {0.5, 1};
    test.loadNetworkToGPU();
    //test.print();
    //test.randomize(); // This does nothing to the result since the weights and biases were already loaded to the GPU.
    float * result = test.feedForwardCPU(passIn);
    float * result2 = test.feedForwardGPU(passIn);

    //test.unloadNetworkFromGPU(); // (Delete) should call unloadNetworkFromGPU, but I call it anyways.
    float expected[] = {1.5};
    test.getBackPropDeltasCPU(expected, result);
    test.backPropGPU(expected, result2);
    test.print();

    float * deltaWeights = (float *) malloc(test.totalWeights * sizeof(float));
    float * deltaBiases = (float *) malloc(test.totalNNSize * sizeof(float));
    cudaMemcpy(deltaWeights, test.deviceDeltaWeights, test.totalWeights * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(deltaBiases, test.deviceDeltaBiases, test.totalNNSize * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < test.totalWeights; i++) {
        std::cout << "Weights: " << test.deltaWeights[i] << " " << deltaWeights[i] << " " <<  std::endl;
    }
    for(int i = 0; i < test.totalNNSize; i++) {
        std::cout << "Biases: " << test.deltaBiases[i] << " " << deltaBiases[i] << std::endl;
    }

    free(result);
    //test.save();
    std::cout << "Results: " << result[0] << std::endl;
    //std::cout << "Results: " << result2[0] << std::endl;
    // Expected outputs: 0.455, 0.91, 1.375, 1.37
}

/*
float abValue(float num) {
    return num > 0 ? num : -num;
}

int main() {
    int dims[] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    const int TIMESTOTEST = 100;
    const int INPUTLAYERAMOUNT = 1024;
    int dimsLength = sizeof(dims)/sizeof(dims[0]);
    float CPUSolutions[TIMESTOTEST];
    float GPUSolutions[TIMESTOTEST];
    Network test = Network(dimsLength, dims);
    test.randomize();
    test.loadNetworkToGPU();
    float input[TIMESTOTEST][INPUTLAYERAMOUNT];

    srand(23452345);

    for(int i = 0; i < TIMESTOTEST; i++) {
        for(int j = 0; j < INPUTLAYERAMOUNT; j++) {
            input[i][j] = 2*((float) rand() / (float) RAND_MAX) - 1;
        }
    }

    std::cout << "Starting CPU now." << std::endl;
    auto startCPU = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < TIMESTOTEST; i++) {
        float * result = test.feedForwardCPU(input[i]);
        CPUSolutions[i] = result[0];
        free(result);
    }
    auto stopCPU = std::chrono::high_resolution_clock::now();

    std::cout << "Starting GPU now." << std::endl;
    auto startGPU = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < TIMESTOTEST; i++) {
        float * result = test.feedForwardGPU(input[i]);
        GPUSolutions[i] = result[0];
        free(result);
    }
    auto stopGPU = std::chrono::high_resolution_clock::now();

    auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(stopCPU - startCPU);
    auto durationGPU = std::chrono::duration_cast<std::chrono::microseconds>(stopGPU - startGPU);

    std::cout << "The CPU took " << durationCPU.count() << " microseconds." << std::endl;
    std::cout << "The GPU took " << durationGPU.count() << " microseconds." << std::endl;
    int correct = 0;
    for(int i = 0; i < TIMESTOTEST; i++) {
        correct += (abValue(CPUSolutions[i]/GPUSolutions[i]) > 0.95 && abValue(CPUSolutions[i]/GPUSolutions[i]) < 1.05);
        std::cout << GPUSolutions[i] << " " << CPUSolutions[i] << " " << abValue(CPUSolutions[i]/GPUSolutions[i]) << std::endl;
    }
    
    std::cout << "The GPU got " << correct << " correct answer(s)." << std::endl;
};*/
