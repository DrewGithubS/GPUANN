#include<iostream>
#include<cstdlib>

__global__ void feedForwardLayer(float * values, float * biases, float * weights, int * dims, int * sumDims, int * sumWeights, int currentLayer) {
    /* This gets the 'index' of the thread. The thread id is the id of the thread in the thread block
       The blockDim is the amount of threads per block and the block id is the id of the block this thread is in
    */ This basically gives the index of the thread so you can use it to access memory.
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    // If the thread index is less than the size of the array.
    if(index < dims[currentLayer]) {
        float temp = 0;
        for(int subNeuron = sumDims[currentLayer-1]; subNeuron < dims[currentLayer]; subNeuron++) {
            temp += values[sumDims[currentLayer-1] + subNeuron] * weights[sumWeights[currentLayer-1] + dims[currentLayer]*subNeuron + index];
        }
        temp += biases[sumDims[currentLayer] + index];
        values[sumDims[currentLayer] + index] = temp/10; // This division by 10 is arbitrary, I'm using it to keep the numbers small.
    }
}


class Network {
    public:
        int THREADSPERBLOCK = 1024;
        bool loaded = false;
        float * biases;
        float * weights;
        int * dims;
        int * weightDims;
        int * sumDims;
        int * sumWeights;
        int dimsLength;
        int totalNNSize = 0;
        int totalWeights = 0;

        float * deviceBiases;
        float * deviceWeights;
        int * deviceDims;
        int * deviceSumDims;
        int * deviceSumWeights;        

        // Even though this could probably be made to run faster, this is branchless and clear.
        Network(int dimsLen, int * neuronCountList) {
            // Sets dims and dimsLength
            dimsLength = dimsLen;
            dims = (int *) malloc(dimsLength * sizeof(int));
            for(int layer = 0; layer < dimsLength; layer++) {
                dims[layer] = neuronCountList[layer];
            }

            // Gets the total size of the network.
            for(int layer = 0; layer < dimsLength; layer++) {
                totalNNSize += dims[layer];
            }

            // Gets the total amount of weights, and the amount of weights per layer
            weightDims = (int *) malloc((dimsLength) * sizeof(int));
            weightDims[dimsLength-1] = 0;
            for(int layer = 0; layer < dimsLength-1; layer++) {
                weightDims[layer] = dims[layer] * dims[layer+1];
                totalWeights += weightDims[layer];
            }

            // Gets the amount of weights from the 0th weight for each layer
            sumWeights = (int *) malloc((dimsLength) * sizeof(int));
            sumWeights[0] = 0;
            sumWeights[1] = weightDims[0];
            for(int layer = 1; layer < dimsLength-1; layer++) {
                sumWeights[layer+1] = weightDims[layer] + weightDims[layer+1];
            }

            // Gets the amount of neuron from the 0th neuron for each layer
            sumDims = (int *) malloc((dimsLength+1) * sizeof(int));
            sumDims[0] = 0;
            sumDims[1] = dims[0];
            for(int layer = 0; layer < dimsLength; layer++) {
                sumDims[layer+2] = sumDims[layer] + dims[layer+1];
            }

            // Initializes the weights and biases with a value of 0.
            biases = (float *) calloc(totalNNSize, sizeof(float));
            weights = (float *) calloc(totalWeights, sizeof(float));
        };
        void loadNetworkToGPU() {
            cudaMalloc(&deviceBiases, totalNNSize * sizeof(float));
            cudaMalloc(&deviceWeights, totalWeights * sizeof(float));
            cudaMalloc(&deviceDims, dimsLength * sizeof(int));
            cudaMalloc(&deviceSumDims, (dimsLength+1) * sizeof(int));
            cudaMalloc(&deviceSumWeights, dimsLength * sizeof(int));

            cudaMemcpy(deviceBiases, biases, totalNNSize * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(deviceWeights, weights, totalWeights * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(deviceDims, dims, dimsLength * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(deviceSumDims, sumDims, (dimsLength+1) * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(deviceSumWeights, sumWeights, dimsLength * sizeof(int), cudaMemcpyHostToDevice);
            loaded = true;
        }
        float * feedForward(float input[]) {
            // Creates a new, empty value array.
            float * values = (float *) calloc(totalNNSize, sizeof(float));

            // Puts the inputs into the values to be fed forward.
            for(int neuron = 0; neuron < dims[0]; neuron++) {
                values[neuron] = input[neuron];
            }
            
            // Allocates device memory for the values
            float * deviceValues;
            cudaMalloc(&deviceValues, totalNNSize * sizeof(float));
            cudaMemcpy(deviceValues, values, totalNNSize * sizeof(float), cudaMemcpyHostToDevice);

            // For each layer, feed forward using a GPU thread instead of a for loop.
            for(int layer = 1; layer < dimsLength; layer++) {
                /* Serial Method to feedforward
                for(int neuron = 0; neuron < dims[layer]; neuron++) {
                    for(int subNeuron = sumDims[layer-1]; subNeuron < sumDims[layer]; subNeuron++) {
                        values[sumDims[layer] + neuron] += values[sumDims[layer-1] + subNeuron] * weights[sumWeights[layer-1] + dims[layer]*subNeuron + neuron];
                    }
                }
                */
                int dimBlock = THREADSPERBLOCK;
                int dimGrid = (dims[layer] + THREADSPERBLOCK - 1)/THREADSPERBLOCK;
                feedForwardLayer <<< dimGrid, dimBlock >>> (deviceValues, deviceBiases, deviceWeights, deviceDims, deviceSumDims, deviceSumWeights, layer);
            }

            cudaMemcpy(values, deviceValues, totalNNSize * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(deviceValues);
            return values;
        }
        void unloadNetworkFromGPU() {
            cudaFree(deviceBiases);
            cudaFree(deviceWeights);
            cudaFree(deviceDims);
            cudaFree(deviceSumDims);
            cudaFree(deviceSumWeights);
            loaded = false;
        }
        ~Network() {
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
        void print() {
            for(int layer = 0; layer < dimsLength; layer++) {
                std::cout << "Layer " << layer+1 << ":" << std::endl;
                for(int neuron = 0; neuron < dims[layer]; neuron++) {
                    std::cout << "\tNeuron " << neuron+1 << ":" << std::endl;
                    std::cout << "\t\tBias: " << biases[sumDims[layer] + neuron] << ":" << std::endl;
                    if(layer < (dimsLength-1)) {
                        std::cout << "\t\tWeights: " << std::endl;
                        for(int weight = 0; weight < dims[layer+1]; weight++) {
                            std::cout << "\t\t\tWeight " << weight+1 << ": " << weights[sumWeights[layer] + dims[layer+1]*neuron + weight] << ":" << std::endl;
                        }
                    }
                }
            }
        }
};

/* Some sample code:

int main() {
    int dims[] = {2, 3};
    float * biases = (float *) calloc(5, sizeof(float));
    float * weights = (float *) calloc(6, sizeof(float));
    weights[0] = 0.25;
    weights[1] = 0.5;
    weights[2] = 0.75;
    weights[3] = 0.33;
    weights[4] = 0.66; 
    weights[5] = 1;
    Network * test = new Network(2, dims);
    test.biases = biases;
    test.weights = weights;
    float passIn[] = {0.5, 1};
    test.loadNetworkToGPU();
    float * result = test.feedForward(passIn);
    test.unloadNetworkFromGPU();
    delete test; // Delete should call unloadNetworkFromGPU, but I call it anyways.
    test.print();
    std::cout << "Results: " << result[2] << " " << result[3] << " " << result[4] << std::endl;
    std::cout << result[2] << std::endl;
    std::cout << result[3] << std::endl;
    std::cout << result[4] << std::endl;
    // Expected outputs: 0.455, 0.91, 1.375
}
*/
