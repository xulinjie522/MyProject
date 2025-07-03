#include "BaseTRTInfer.hpp"

void BaseTRTInfer::doLoadEngine(){
    loadEngine();
    allocateMemory();
}

void BaseTRTInfer::doInfer(){
    infer(getInputPtr(), getOutputPtr());
}

// Function to load a serialized engine from file
void BaseTRTInfer::loadEngine() {
    std::ifstream engineFile(config.enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Failed to open engine file: " << config.enginePath << std::endl;
        exit(EXIT_FAILURE);
    }

    engineFile.seekg(0, std::ios::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);

    std::vector<char> engineData(fileSize);
    engineFile.read(engineData.data(), fileSize);

    IRuntime* runtime = createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "Failed to create runtime." << std::endl;
        exit(EXIT_FAILURE);
    }

    engine = runtime->deserializeCudaEngine(engineData.data(), fileSize);
    if (!engine) {
        std::cerr << "Failed to create engine from serialized data." << std::endl;
        exit(EXIT_FAILURE);
    }
    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        return;
    }
    
    runtime->destroy();
}

float* BaseTRTInfer::getInputPtr(){
    return hInput.get();
}

float* BaseTRTInfer::getOutputPtr(){
    return hOutput.get();
}

void BaseTRTInfer::infer(float* input, float* output) {

    int h = config.kInputH, w = config.kInputW;

    // Convert input to float and copy to device memory
    CHECK(cudaMemcpyAsync(buffers[inputIndex], hInput.get(), config.InputSize, cudaMemcpyHostToDevice, stream));

    // Execute the inference
    context->enqueueV2((void**)buffers, stream, nullptr);
    
    cudaMemcpyAsync(hOutput.get(), buffers[outputIndex], config.OutputSize, cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);

}

BaseTRTInfer::~BaseTRTInfer(){
    cudaStreamDestroy(stream);
    if(context){
        context->destroy();
        context = nullptr;
    }
    if(engine){
        engine->destroy();
        engine = nullptr;
    }
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
}