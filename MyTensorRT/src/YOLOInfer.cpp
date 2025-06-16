#include "YOLOInfer.hpp"

using namespace nvinfer1;

// Function to load a serialized engine from file
void YOLOInfer::loadEngine() {

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
    
    model_bboxes = engine->getBindingDimensions(1).d[0]; //TO DO?
    std::cout << "Output dims: ";
    for(int i = 0; i < engine->getBindingDimensions(1).nbDims; ++i) std::cout << engine->getBindingDimensions(1).d[i] << " ";

    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        return;
    }

    runtime->destroy();
}

void YOLOInfer::allocateMemory(){
    inputIndex = engine->getBindingIndex("images");
    outputIndex = engine->getBindingIndex("output0");
    // Allocate device memory
    cudaMalloc((void**)&buffers[inputIndex], config.InputSize);
    cudaMalloc((void**)&buffers[outputIndex], config.OutputSize);
    hInput = std::make_unique<float[]>(config.InputSize);
    hOutput = std::make_unique<float[]>(config.OutputSize);
    decode_ptr_host = new float[1 + config.kMaxNumOutputBbox * bbox_element];
    CHECK(cudaMalloc((void**)&deocde_ptr_device, sizeof(float) * (1 + config.kMaxNumOutputBbox * bbox_element)));
    cudaStreamCreate(&stream);
}

