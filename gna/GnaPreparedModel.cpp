#define LOG_TAG "GnaPreparedModel"

#include "GnaPreparedModel.h"
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

void GnaPreparedModel::deinitialize() {
    VLOG(L1, "deinitialize");
    delete enginePtr;
    enginePtr = nullptr;
    VLOG(L1, "free engine");
}

bool GnaPreparedModel::initialize() {
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

    Model model = mModel;

    for (const auto& operation : mModel.operations) {
        VLOG(L1, "get operation %d ready to add", operation.type);
        dumpOperation(operation);
        switch (operation.type) {

            case OperationType::CONCATENATION:{
                success = concat::initialize(mTargetDevice.c_str(), operation, model);
            } break;
            case OperationType::FULLY_CONNECTED:{
                success = fullyconnected::initialize(mTargetDevice.c_str(), operation, model);
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

IRBlob::Ptr GnaPreparedModel::GetConstWeightsOperandAsTensor(uint32_t index)
{
    dumpOperand(index, mModel);
    const auto op = mModel.operands[index];
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(mModel, index, len);

    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        if (buf == nullptr)
            VLOG(L1, "TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");

        vec<unsigned int> order;
        Layout layout;
        if (op.dimensions.size() == 4) {
            order = {3,0,1,2};  //IHWO -> OIHW for depth conv
            layout = Layout::OIHW; //weights layout
        }
        else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = Layout::NC;
        }
        else {
            order = {0};
            layout = Layout::C;
        }

        TensorDims inputDims;
        if (op.dimensions.size() == 3) {
            auto channel_size = op.dimensions[1] * op.dimensions[2];
            uint32_t op_dimensions_size = 2;
            std::vector<uint32_t> op_dimensions = {op.dimensions[0], channel_size};
            inputDims = toDims(op_dimensions);
        }
        else {
            inputDims = toDims(op.dimensions);
        }

        TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
        if (inputDims.size() != 4) {
            InferenceEngine::TBlob<float>::Ptr blob =
                                std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
            blob->allocate();
            return blob;
        } else {
            InferenceEngine::TBlob<float>::Ptr blob =
                                    std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();

            auto dims_ohwi = inputDims;
            size_t out_depth = dims_ohwi[0];
            size_t in_depth = dims_ohwi[3];
            size_t height = dims_ohwi[1];
            size_t width = dims_ohwi[2];
            size_t offset = 0;
            const float* inputFilter = reinterpret_cast<const float *>(buf);

            //convert OHWI -> OIHW
            //for depth conv need reorder as IOHW since for tflite O is always 1 and IE expects reorder to
            //[in_channels, depth_multiplier, filter_height, filter_width]
            for (size_t i = 0; i < in_depth; i++) {
                for (size_t o = 0; o < out_depth; o++) {
                    for (size_t h = 0; h < height; h++) {
                        for (size_t w = 0; w < width; w++) {
                            //similar to NHWC memory layout
                            size_t offset_ohwi = o*height*width*in_depth +
                                                 h*width*in_depth +
                                                 w*in_depth + i;
                            blob->buffer().as<float*>()[offset++] = inputFilter[offset_ohwi];
                        }
                    }
                }
            }
            return blob;
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        if (buf == nullptr)
            VLOG(L1, "TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");

        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
        return std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
    } else {
        VLOG(L1, "Do not support const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

IRBlob::Ptr GnaPreparedModel::GetConstOperandAsTensor(int operand_index, int operation_idx)
{
    dumpOperand(operand_index, mModel);
    const auto op = mModel.operands[operand_index];
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(mModel, operand_index, len);

    VLOG(L1, "GnaPreparedModel:: Operand: index: %d, len: %d, buf: %p", operand_index, len, buf);
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        vec<unsigned int> order;
        Layout layout;
        if (op.dimensions.size() == 4) {
            order = {0,3,1,2};  //nhwc -> nchw
            layout = Layout::OIHW; //weights layout
        } else if (op.dimensions.size() == 2 || op.dimensions.size() == 3) {
            order = {0, 1};
            layout = Layout::NC;
        } else {
            order = {0};
            layout = Layout::C;
        }

        TensorDims inputDims;
        if (op.dimensions.size() == 3) {
            auto channel_size = op.dimensions[1] * op.dimensions[2];
            uint32_t op_dimensions_size = 2;
            std::vector<uint32_t> op_dimensions = {op.dimensions[0], channel_size};
            inputDims = toDims(op_dimensions);
        } else {
            inputDims = toDims(op.dimensions);
        }

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
                            std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();

                auto dims_ohwi = inputDims;
                size_t out_depth = dims_ohwi[0];
                size_t in_depth = dims_ohwi[3];
                size_t height = dims_ohwi[1];
                size_t width = dims_ohwi[2];
                size_t offset = 0;
                const float* inputFilter = reinterpret_cast<const float *>(buf); //OHWI memory layout

                for (size_t o = 0; o < out_depth; o++) {
                    for (size_t i = 0; i < in_depth; i++) {
                        for (size_t h = 0; h < height; h++) {
                            for (size_t w = 0; w < width; w++) {
                                //similar to NHWC memory layout
                                size_t offset_ohwi = o*height*width*in_depth +
                                                     h*width*in_depth +
                                                     w*in_depth + i;
                                blob->buffer().as<float*>()[offset++] = inputFilter[offset_ohwi];
                            }
                        }
                    }
                }
                return blob;
            }
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        if (buf == nullptr)
            VLOG(L1, "TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");

        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
        return std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
    } else {
        VLOG(L1, "Do not support const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

Blob::Ptr GnaPreparedModel::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len)
{
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        if (op.lifetime == OperandLifeTime::MODEL_INPUT) {
            if (buf == nullptr)
                VLOG(L1, "MODEL_INPUT buf is NULL !!!!!!!!!!!!!!!");

            vec<unsigned int> order;
            Layout layout;
            if (op.dimensions.size() == 4) {
                order = {0,3,1,2};  //nhwc -> nchw
                layout = Layout::NCHW;
            }
            else if (op.dimensions.size() == 2) {
                order = {0, 1};
                layout = Layout::NC;
            }
            else {
                order = {0}; //(op.dimensions.size() < 2)
                layout = Layout::C;
            }

            auto inputDims = toDims(op.dimensions);
            TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
            if (inputDims.size() != 4) {
                //VLOG(L1, "buf data %f", *((float*)buf));
                //VLOG(L1, "buf data %f", *((float*)buf + 1));
                InferenceEngine::TBlob<float>::Ptr blob =
                                std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
                blob->allocate();
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                                            std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();

                auto dims_nhwc = inputDims; //toDims(op.dimensions);
                size_t batch = dims_nhwc[0];
                size_t in_depth = dims_nhwc[3]; //channels
                size_t height = dims_nhwc[1];
                size_t width = dims_nhwc[2];
                size_t offset = 0;
                const float* input = reinterpret_cast<const float *>(buf); //OHWI memory layout

                //convert NHWC -> NCHW
                for (size_t b = 0; b < batch; b++) {
                    for (size_t i = 0; i < in_depth; i++) {
                        for (size_t h = 0; h < height; h++) {
                            for (size_t w = 0; w < width; w++) {
                                //similar to NHWC memory layout
                                size_t offset_nhwc = b*height*width*in_depth +
                                                     h*width*in_depth +
                                                     w*in_depth + i;
                                blob->buffer().as<float*>()[offset++] = input[offset_nhwc];
                            }
                        }
                    }
                }
                return blob;
            }
        } else if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
            if (buf == nullptr)
                VLOG(L1, "MODEL_OUTPUT buf is NULL !!!!!!!!!!!!!!!");

            vec<unsigned int> order;
            Layout layout;
            if (op.dimensions.size() == 4) {
                layout = Layout::NHWC;
            } else if (op.dimensions.size() == 2) {
                layout = Layout::NC;
            } else {
                layout = Layout::C;
            }

            TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), layout); //nhwc
            InferenceEngine::TBlob<float>::Ptr blob =
                            InferenceEngine::make_shared_blob<float>(td, (float *)buf, len);
            return blob;
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
        return std::make_shared<InferenceEngine::TBlob<int32_t>>(td, (int32_t *)buf, len);
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
