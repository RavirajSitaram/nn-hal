#define LOG_TAG "CpuPreparedModel"

#include "CpuPreparedModel.h"
#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <fstream>
#include <thread>
#include "ValidateHal.h"
#include "utils.h"

using namespace android::nn;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

void CpuPreparedModel::deinitialize() {
    VLOG(L1, "deinitialize");
    delete enginePtr;
    enginePtr = nullptr;
    VLOG(L1, "free engine");
}

bool CpuPreparedModel::initialize(const Model& model) {
    VLOG(L1, "initialize");
    bool success = false;

     // Check operation supoorted or not, user may not call getOpertionSupported()
    for (const auto& operation : mModel.operations) {
        success = isOperationSupported(operation, mModel);
        dumpOperationSupport(operation, success);
        if (!success) {
            VLOG(L1, "get unsupported operation in initialize()");
            return false;
        }
    }

    success = setRunTimePoolInfosFromHidlMemories(&mPoolInfos, mModel.pools);
    if (!success) {
        VLOG(L1, "setRunTimePoolInfosFromHidlMemories failed.");
        return false;
    }

    success = initializeRunTimeOperandInfo();
    if (!success) {
        VLOG(L1, "initializeRunTimeOperandInfo failed.");
        return false;
    }
    VLOG(L1, "initializeRunTimeOperandInfo success.");

    for (const auto& operation : mModel.operations) {
        VLOG(L1, "get operation %d ready to add", operation.type);
        dumpOperation(operation);
        switch (operation.type) {
            case OperationType::ADD: {
                VLOG(L1, "Initializing ADD operation");
                success = add::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = add::updateDataPtr();
            } break;
            case OperationType::MUL: {
                VLOG(L1, "Initializing MUL operation");
                success = mul::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = mul::updateDataPtr();
            } break;
            case OperationType::CONCATENATION: {
                VLOG(L1, "Initializing CONCATENATION operation");
                success = concat::initialize(mTargetDevice.c_str(), operation, model);
                mCreateNgraph->addConcat(concat::getNodeName(), concat::getInputName(), concat::getAxis());
            } break;
            case OperationType::FULLY_CONNECTED: {
                VLOG(L1, "Initializing FULLY_CONNECTED operation");
                success = fullyconnected::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = fullyconnected::updateDataPtr();
            } break;
            case OperationType::AVERAGE_POOL_2D: {
                VLOG(L1, "Initializing AVERAGE_POOL_2D operation");
                success = avgpool::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = avgpool::updateDataPtr();
            } break;
            case OperationType::MAX_POOL_2D: {
                VLOG(L1, "Initializing MAX_POOL_2D operation");
                success = maxpool::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = maxpool::updateDataPtr();
            } break;
            case OperationType::CONV_2D: {
                VLOG(L1, "Initializing CONV_2D operation");
                success = convolution::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = convolution::updateDataPtr();
                GenConvParams gPrms = convolution::getGenConvPrms();
                mCreateNgraph->addConvolution(convolution::getNodeName(), convolution::getInputName(), gPrms);
            } break;
            case OperationType::DEPTHWISE_CONV_2D: {
                VLOG(L1, "Initializing DEPTHWISE_CONV_2D operation");
                success = depthconv::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = depthconv::updateDataPtr();
                GenConvParams gPrms = depthconv::getGenConvPrms();
                mCreateNgraph->addConvolution(depthconv::getNodeName(), depthconv::getInputName(), gPrms);
            } break;
            case OperationType::L2_NORMALIZATION: {
                VLOG(L1, "Initializing L2_NORMALIZATION operation");
                success = l2normalization::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = l2normalization::updateDataPtr();
            } break;
            case OperationType::LOCAL_RESPONSE_NORMALIZATION: {
                VLOG(L1, "Initializing LOCAL_RESPONSE_NORMALIZATION operation");
                success = lrn::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = lrn::updateDataPtr();
            } break;
            case OperationType::RELU: {
                VLOG(L1, "Initializing RELU operation");
                success = relu::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = relu::updateDataPtr();
            } break;
            case OperationType::RELU1: {
                VLOG(L1, "Initializing RELU1 operation");
                success = relu1::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = relu1::updateDataPtr();
            } break;
            case OperationType::RELU6: {
                VLOG(L1, "Initializing RELU6 operation");
                success = relu6::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = relu6::updateDataPtr();
            } break;
            case OperationType::LOGISTIC: {
                VLOG(L1, "Initializing LOGISTIC operation");
                success = logistic::initialize(mTargetDevice.c_str(), operation, model);
            } break;
            case OperationType::SOFTMAX: {
                VLOG(L1, "Initializing SOFTMAX operation");
                success = softmax::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = softmax::updateDataPtr();
            } break;
            case OperationType::TANH: {
                VLOG(L1, "Initializing TANH operation");
                success = tanh::initialize(mTargetDevice.c_str(), operation, model);
            } break;
            case OperationType::RESHAPE: {
                VLOG(L1, "Initializing RESHAPE operation");
                success = reshape::initialize(mTargetDevice.c_str(), operation, model);
                mPorts[operation.outputs[0]] = reshape::updateDataPtr();
                mCreateNgraph->addReshape(reshape::getNodeName(), reshape::getInputName(), reshape::getShape());
            } break;
            default:
                VLOG(L1, "unsupported operation %d", operation.type);
                return false;
        }
        if (success == false) {
            VLOG(L1, "failed to convert operation %d", operation.type);
            return false;
        }
    }

    initializeInput();
    success = finalizeOutput();

    InferenceEngine::CNNNetwork ngraph_net;
    ngraph_net = mCreateNgraph->generate(std::string("/data/vendor/neuralnetworks/ngraph_ir.xml"),
                                         std::string("/data/vendor/neuralnetworks/ngraph_ir.bin"));
    if (success == false) return success;

    mNet.buildNetwork();
    VLOG(L1, "initialize ExecuteNetwork for device %s", mTargetDevice.c_str());
    enginePtr = new ExecuteNetwork(ngraph_net, mNet, mTargetDevice);
    enginePtr->prepareInput();
    enginePtr->loadNetwork(ngraph_net);

    return true;
}

IRBlob::Ptr CpuPreparedModel::GetConstWeightsOperandAsTensor(uint32_t index) {
    dumpOperand(index, mModel);
    const auto op = mModel.operands[index];
    uint32_t len;
    const uint8_t* buf = GetOperandMemory(mModel, index, len);
    VLOG(L1, "CpuPreparedModel:: Operand: index: %d, len: %d, buf: %p", index, len, buf);
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        vec<unsigned int> order;
        Layout layout;
        if (op.dimensions.size() == 4) {
            // order = {0,3,1,2};  //nhwc -> nchw
            order = {3, 0, 1, 2};   // IHWO -> OIHW for depth conv
            layout = Layout::OIHW;  // weights layout
        } else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = Layout::NC;
        } else {
            order = {0};  //(op.dimensions.size() < 2)
            layout = Layout::C;
        }
        auto inputDims = toDims(op.dimensions);
        TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
        if (buf == nullptr) {
            VLOG(L1, "TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            if (inputDims.size() != 4) {
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();

                auto dims_ohwi = inputDims;  // toDims(op.dimensions);
                size_t out_depth = dims_ohwi[0];
                size_t in_depth = dims_ohwi[3];
                size_t height = dims_ohwi[1];
                size_t width = dims_ohwi[2];
                size_t offset = 0;  // blob->size() == o*i*h*w and simlar to nchw memory layout
                const float* inputFilter =
                    reinterpret_cast<const float*>(buf);  // OHWI memory layout

                // convert OHWI -> OIHW

                // for depth conv need reorder as IOHW since for tflite O is always 1 and IE expects
                // reorder to [in_channels, depth_multiplier, filter_height, filter_width]
                for (size_t i = 0; i < in_depth; i++) {
                    for (size_t o = 0; o < out_depth; o++) {
                        for (size_t h = 0; h < height; h++) {
                            for (size_t w = 0; w < width; w++) {
                                size_t offset_ohwi = o * height * width * in_depth +
                                                     h * width * in_depth + w * in_depth +
                                                     i;  // similar to NHWC memory layout
                                blob->buffer().as<float*>()[offset++] = inputFilter[offset_ohwi];
                            }
                        }
                    }
                }

                return blob;
            }
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        VLOG(L1, "check if const tensors of type IN32 supported");
        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
        if (buf == nullptr) {
            VLOG(L1, "TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
            return blob;
        }
    } else {
        VLOG(L1, "not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

IRBlob::Ptr CpuPreparedModel::GetConstOperandAsTensor(int operand_idx, int operation_idx) {
    dumpOperand(operand_idx, mModel);
    const auto op = mModel.operands[operand_idx];
    uint32_t len;

    const uint8_t* buf = GetOperandMemory(mModel, operand_idx, len);
    VLOG(L1, "CpuPreparedModel:: operand_index: %d, operation_index :%d,len: %d, buf: %p",
         operand_idx, operation_idx, len, buf);

    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        vec<unsigned int> order;
        Layout layout;
        if (op.dimensions.size() == 4) {
            order = {0, 3, 1, 2};   // nhwc -> nchw
            layout = Layout::OIHW;  // weights layout
        } else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = Layout::NC;
        } else {
            order = {0};  //(op.dimensions.size() < 2)
            layout = Layout::C;
        }
        auto inputDims = toDims(op.dimensions);
        TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
        if (buf == nullptr) {
            VLOG(L1, "TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            if (inputDims.size() != 4) {
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();

                auto dims_ohwi = inputDims;  // toDims(op.dimensions);
                size_t out_depth = dims_ohwi[0];
                size_t in_depth = dims_ohwi[3];
                size_t height = dims_ohwi[1];
                size_t width = dims_ohwi[2];
                size_t offset = 0;  // blob->size() == o*i*h*w and simlar to nchw memory layout
                const float* inputFilter =
                    reinterpret_cast<const float*>(buf);  // OHWI memory layout

                for (size_t o = 0; o < out_depth; o++) {
                    for (size_t i = 0; i < in_depth; i++) {
                        for (size_t h = 0; h < height; h++) {
                            for (size_t w = 0; w < width; w++) {
                                size_t offset_ohwi = o * height * width * in_depth +
                                                     h * width * in_depth + w * in_depth +
                                                     i;  // similar to NHWC memory layout
                                blob->buffer().as<float*>()[offset++] = inputFilter[offset_ohwi];
                            }
                        }
                    }
                }

                return blob;
            }
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        VLOG(L1, "check if const tensors of type IN32 supported");
        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
        if (buf == nullptr) {
            VLOG(L1, "TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
            return blob;
        }
    } else {
        VLOG(L1, "not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

Blob::Ptr CpuPreparedModel::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                                  uint32_t& len) {
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        if (op.lifetime == OperandLifeTime::MODEL_INPUT) {
            VLOG(L1, "Create input blob !!!!");
            vec<unsigned int> order;
            Layout layout;
            if (op.dimensions.size() == 4) {
                order = {0, 3, 1, 2};  // nhwc -> nchw
                layout = Layout::NCHW;
            } else if (op.dimensions.size() == 2) {
                order = {0, 1};
                layout = Layout::NC;
            } else {
                order = {0};  //(op.dimensions.size() < 2)
                layout = Layout::C;
            }

            auto inputDims = toDims(op.dimensions);
            TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);

            if (buf == nullptr) {
                VLOG(L1, "MODEL_INPUT buf is NULL !!!!!!!!!!!!!!!");
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();
                return blob;
            } else {
                if (inputDims.size() != 4) {
                    InferenceEngine::TBlob<float>::Ptr blob =
                        std::make_shared<InferenceEngine::TBlob<float>>(td, (float*)buf, len);
                    return blob;
                } else {
                    InferenceEngine::TBlob<float>::Ptr blob =
                        std::make_shared<InferenceEngine::TBlob<float>>(td);
                    blob->allocate();

                    auto dims_nhwc = inputDims;  // toDims(op.dimensions);
                    size_t batch = dims_nhwc[0];
                    size_t in_depth = dims_nhwc[3];  // channels
                    size_t height = dims_nhwc[1];
                    size_t width = dims_nhwc[2];
                    size_t offset = 0;  // blob->size() == o*i*h*w and simlar to nchw memory layout
                    const float* input = reinterpret_cast<const float*>(buf);  // OHWI memory layout

                    // convert NHWC -> NCHW

                    for (size_t b = 0; b < batch; b++) {
                        for (size_t i = 0; i < in_depth; i++) {
                            for (size_t h = 0; h < height; h++) {
                                for (size_t w = 0; w < width; w++) {
                                    size_t offset_nhwc = b * height * width * in_depth +
                                                         h * width * in_depth + w * in_depth +
                                                         i;  // similar to NHWC memory layout
                                    blob->buffer().as<float*>()[offset++] = input[offset_nhwc];
                                }
                            }
                        }
                    }

                    return blob;
                }
            }
        } else if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
            VLOG(L1, "Create output blob !!!!");
            vec<unsigned int> order;
            Layout layout;
            if (op.dimensions.size() == 4) {
                // order = {0,3,1,2};  //nhwc -> nchw
                layout = Layout::NHWC;
            } else if (op.dimensions.size() == 2) {
                // order = {0, 1};
                layout = Layout::NC;
            } else if (op.dimensions.size() == 3) {
                // order = {0, 1, 2, 3};  // nhwc -> nchw
                layout = Layout::CHW;
                ALOGI("Anoob : GetInOutOperandAsBlob output already transposed to NHWC");
            } else {
                // order = {0}; //(op.dimensions.size() < 2)
                layout = Layout::C;
            }

            TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), layout);  // nhwc
            if (buf == nullptr) {
                VLOG(L1, "MODEL_OUTPUT buf is NULL !!!!!!!!!!!!!!!");
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                    InferenceEngine::make_shared_blob<float>(td, (float*)buf, len);
                return blob;
            }
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        VLOG(L1, "check if const tensors of type IN32 supported");
        // nnAssert(true);
        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
        return std::make_shared<InferenceEngine::TBlob<int32_t>>(td, (int32_t*)buf, len);
    } else {
        VLOG(L1, "not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
