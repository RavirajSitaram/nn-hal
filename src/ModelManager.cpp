#include "ModelManager.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

template<typename T>
bool deQuantize(const uint8_t* inputData, const uint32_t& len, const float scale,
                const int32_t zeroPoint, float* outputData) {
      int32_t value;
      const T* inputBuf = reinterpret_cast<const T*>(inputData);
      for (int i = 0; i < len; ++i) {
        value = *(inputBuf + i);
        outputData[i] = static_cast<float>(scale * (value - zeroPoint));
        ALOGD("%s Converting value:%d to %f scale factor:%f zero point:%d", __func__, value, outputData[i], len, scale, zeroPoint);
      }
      return true;
}

bool NnapiModelInfo::initializeRunTimeOperandInfo() {
    // initialize runtime operand info from model.
    const size_t count = mModel.main.operands.size();
    if (!count) {
        VLOG(L1, "NNERR:Operand Count is 0");
        return false;
    }
    mOperands.resize(count);
    mOutputShapes.resize(mModel.main.outputIndexes.size());

    // Start by setting the runtime info to what's in the model.
    for (size_t i = 0; i < count; i++) {
        const Operand& from = mModel.main.operands[i];
        RunTimeOperandInfo& to = mOperands[i];
        to.dimensions.resize(from.dimensions.size());
        for (size_t j = 0; j < from.dimensions.size(); j++) {
            to.dimensions[j] = from.dimensions[j];
        }

        switch (from.type) {
            case OperandType::TENSOR_FLOAT32:
            case OperandType::FLOAT32:
                to.type = OperandType::TENSOR_FLOAT32;
                VLOG(L1, "OperandType = %d\n", from.type);
                break;
            case OperandType::INT32:
            case OperandType::UINT32:
                nnAssert(to.scale == 0);
                FALLTHROUGH_INTENDED;
            case OperandType::TENSOR_INT32:
                to.type = from.type;
                break;
            case OperandType::TENSOR_QUANT8_ASYMM:
            case OperandType::TENSOR_QUANT8_SYMM:
                to.type = from.type;
                break;
            default:
                ALOGE("wrong operand type %d", from.type);
                return false;
        }

        to.scale = from.scale;
        to.length = from.location.length;
        to.lifetime = from.lifetime;
        to.zeroPoint = from.zeroPoint;
        to.ignoreLayout = false;

        ALOGD("Operand index:%d zeroPoint=%d scaleFactor=%f", i, from.zeroPoint, from.scale);
        ALOGD("Operand index:%d zeroPoint=%d scaleFactor=%f", i, to.zeroPoint, to.scale);
        
        switch (from.lifetime) {
            case OperandLifeTime::TEMPORARY_VARIABLE:
                to.buffer = nullptr;
                to.length = sizeOfData(to.type, to.dimensions);
                to.numberOfUsesLeft = from.numberOfConsumers;
                break;
            case OperandLifeTime::CONSTANT_COPY:
                to.buffer = const_cast<uint8_t*>(&mModel.operandValues[from.location.offset]);
                to.numberOfUsesLeft = 0;
                break;
            case OperandLifeTime::CONSTANT_REFERENCE: {
                auto poolIndex = from.location.poolIndex;
                nnAssert(poolIndex < mPoolInfos.size());
                auto& r = mPoolInfos[poolIndex];
                to.buffer = r.buffer + from.location.offset;
                to.numberOfUsesLeft = 0;
                break;
            }
            case OperandLifeTime::SUBGRAPH_OUTPUT:
            case OperandLifeTime::SUBGRAPH_INPUT:
            case OperandLifeTime::NO_VALUE:
                to.buffer = nullptr;
                to.numberOfUsesLeft = 0;
                break;
            default:
                return false;
                break;
        }
    }

    for (uint32_t i = 0; i < mModel.main.outputIndexes.size(); i++) {
        const uint32_t operandIndex = mModel.main.outputIndexes[i];
        const RunTimeOperandInfo& from = mOperands[operandIndex];
        mOutputShapes[i].dimensions = from.dimensions;
        mOutputShapes[i].isSufficient = true;
    }

    return true;
}

// TODO: Move it to Utils class
template <typename T>
T NnapiModelInfo::GetConstFromBuffer(const uint8_t* buf, uint32_t len) {
    VLOG(L1, "buf: %p, len: %d", buf, len);
    if (len != sizeof(T)) {
        VLOG(L1, "fix me: typeid(T).name() should be %d bytes", sizeof(T));
        // fix me if buffer is of type float and if float and OperandLifeTime::CONSTANT_REFERENCE
        nnAssert(false);
    }
    return *(T*)(buf);
}

const uint8_t* NnapiModelInfo::GetOperandMemory(int index, uint32_t& lenOut) {
    ALOGD("%s", __func__);
    const auto op = mModel.main.operands[index];
    lenOut = op.location.length;
    if (op.lifetime == OperandLifeTime::CONSTANT_COPY) {
        ALOGD("CONST_COPY");
        if (op.location.poolIndex != 0) {
            ALOGE("CONSTANT_COPY expects poolIndex to be 0");
            nnAssert(false);
        }
        VLOG(L1, "operand lifetime OperandLifeTime::CONSTANT_COPY");
        return (const_cast<uint8_t*>(&mModel.operandValues[op.location.offset]));
        // to.numberOfUsesLeft = 0;
    } else if (op.lifetime == OperandLifeTime::CONSTANT_REFERENCE) {
        VLOG(L1, "operand lifetime OperandLifeTime::CONSTANT_REFERENCE");
        auto poolIndex = op.location.poolIndex;
        // nnAssert(poolIndex < mPoolInfos.size()); //aks fix me
        auto& r = mPoolInfos[poolIndex];
        return (const_cast<uint8_t*>(r.buffer + op.location.offset));
    } else if (op.lifetime == OperandLifeTime::SUBGRAPH_INPUT ||
            op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT ||
            op.lifetime == OperandLifeTime::NO_VALUE) {
        VLOG(L1, "operand lifetime OperandLifeTime::MODEL_INPUT||MODEL_OUTPUT||NO_VALUE");
        lenOut = sizeOfData(op.type, op.dimensions);
        return nullptr;
    } else if (op.lifetime == OperandLifeTime::TEMPORARY_VARIABLE) {
        VLOG(L1, "operand lifetime OperandLifeTime::TEMPORARY_VARIABLE");
        VLOG(L1, "operand is expected to be const, but lifetime is %d", op.lifetime);
        lenOut = sizeOfData(op.type, op.dimensions);
        // nnAssert(false);
        return nullptr;
    }

    ALOGE("operand is expected to be const, but lifetime is %d", op.lifetime);
    nnAssert(false);  // temp fix since some time const operand set as TEMPORARY_VARIABLE
    return nullptr;
}

Blob::Ptr NnapiModelInfo::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                                uint32_t& len,  bool ignoreLayout) {
    bool isQuantInput = false;
    if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED ||
            op.type == OperandType::TENSOR_QUANT8_ASYMM ||
            op.type == OperandType::TENSOR_QUANT8_SYMM ||
            op.type == OperandType::TENSOR_QUANT16_SYMM ) {
        ALOGD("Is quantized input");
        isQuantInput = true;
    } else {
        ALOGD("Is not quantized input.... %d", op.type);
    }

    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32 || isQuantInput) {
        if (op.lifetime == OperandLifeTime::SUBGRAPH_INPUT) {
            VLOG(L1, "Create input blob !!!!   dimensions size=%d", op.dimensions.size());
            if (buf == nullptr)
                VLOG(L1, "SUBGRAPH_INPUT buf is NULL !!!!!!!!!!!!!!!");

            vec<unsigned int> order;
            InferenceEngine::Layout layout;
            if (op.dimensions.size() == 4) {
                order = {0, 3, 1, 2};  // nhwc -> nchw
                layout = InferenceEngine::Layout::NCHW;
            } else if (op.dimensions.size() == 2) {
                order = {0, 1};
                layout = InferenceEngine::Layout::NC;
            } else {
                order = {0};  //(op.dimensions.size() < 2)
                layout = InferenceEngine::Layout::C;
            }

            auto inputDims = toDims(op.dimensions);
            InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);

                if ( (ignoreLayout && (op.dimensions.size() == 4)) || (inputDims.size() != 4)) {
                    ALOGD("Inside ");
                    InferenceEngine::TBlob<float>::Ptr blob = nullptr;
                    if (isQuantInput || (op.type == OperandType::TENSOR_INT32)) {
                        blob = std::make_shared<InferenceEngine::TBlob<float>>(td);
                        blob->allocate();

                        switch(op.type) {
                            case OperandType::TENSOR_QUANT8_ASYMM_SIGNED:
                                deQuantize<int8_t>(buf, getNumberOfElements(op.dimensions), op.scale, op.zeroPoint, blob->buffer().as<float*>());
                                break;
                            case OperandType::TENSOR_QUANT8_ASYMM:
                                ALOGE("asdasdasdasdasdasdasdasdasdas");
                                deQuantize<uint8_t>(buf, getNumberOfElements(op.dimensions), op.scale, op.zeroPoint, blob->buffer().as<float*>());
                                break;
                            case OperandType::TENSOR_QUANT8_SYMM:
                                deQuantize<int8_t>(buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                                break;
                            case OperandType::TENSOR_QUANT16_SYMM:
                                deQuantize<int16_t>(buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                                break;
                            case OperandType::TENSOR_INT32:
                                deQuantize<int32_t>(buf, getNumberOfElements(op.dimensions), op.scale, op.zeroPoint, blob->buffer().as<float*>());
                                break;
                            default:
                                nnAssert(true);
                                break;
                        }
                    } else {
                        blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
                        blob->allocate();
                    }
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
        } else if (op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
            VLOG(L1, "Create output blob !!!!");
            vec<unsigned int> order;
            InferenceEngine::Layout layout;
            if (op.dimensions.size() == 4) {
                // order = {0,3,1,2};  //nhwc -> nchw
                layout = InferenceEngine::Layout::NHWC;
            } else if (op.dimensions.size() == 2) {
                // order = {0, 1};
                layout = InferenceEngine::Layout::NC;
            } else if (op.dimensions.size() == 3) {
                // order = {0, 1, 2, 3};  // nhwc -> nchw
                layout = InferenceEngine::Layout::CHW;
                ALOGI("GetInOutOperandAsBlob output already transposed to NHWC");
            } else {
                // order = {0}; //(op.dimensions.size() < 2)
                layout = InferenceEngine::Layout::C;
            }

            InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), layout);  // nhwc
            if (buf == nullptr) {
                VLOG(L1, "MODEL_OUTPUT buf is NULL !!!!!!!!!!!!!!!");
                InferenceEngine::TBlob<float>::Ptr blob =
                    std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();
                return blob;
            } else {
                InferenceEngine::TBlob<float>::Ptr blob =
                    InferenceEngine::make_shared_blob<float>(td, (float*)buf);
                return blob;
            }
        }
    } else if (op.type == OperandType::TENSOR_INT32) {
        VLOG(L1, "check if const tensors of type IN32 supported");
        // nnAssert(true);
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), InferenceEngine::Layout::ANY);
        return std::make_shared<InferenceEngine::TBlob<int32_t>>(td, (int32_t*)buf, len);
    } else {
        VLOG(L1, "not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

IRBlob::Ptr NnapiModelInfo::GetConstOperandAsTensor(int operand_idx, int operation_idx) {
    dumpOperand(operand_idx, mModel);
    const auto op = mModel.main.operands[operand_idx];
    uint32_t len;
    bool isQuantInput = false;
    if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED ||
            op.type == OperandType::TENSOR_QUANT8_SYMM ||
            op.type == OperandType::TENSOR_QUANT16_SYMM ) {
        isQuantInput = true;
    }

    const uint8_t* buf = GetOperandMemory(operand_idx, len);
    VLOG(L1, "NnapiModelInfo:: operand_index: %d, operation_index :%d,len: %d, buf: %p",
        operand_idx, operation_idx, len, buf);

    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        vec<unsigned int> order;
        InferenceEngine::Layout layout;
        if (op.dimensions.size() == 4) {
            order = {0, 3, 1, 2};   // nhwc -> nchw
            layout = InferenceEngine::Layout::OIHW;  // weights layout
        } else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = InferenceEngine::Layout::NC;
        } else {
            order = {0};  //(op.dimensions.size() < 2)
            layout = InferenceEngine::Layout::C;
        }
        auto inputDims = toDims(op.dimensions);
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
        if (buf == nullptr) {
            VLOG(L1, "TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            if (inputDims.size() != 4) {
                InferenceEngine::TBlob<float>::Ptr blob = nullptr;
                if (isQuantInput) {
                    blob = std::make_shared<InferenceEngine::TBlob<float>>(td);
                    blob->allocate();

                    switch(op.type) {
                        case OperandType::TENSOR_QUANT8_ASYMM_SIGNED:
                            deQuantize<int8_t>(buf, getNumberOfElements(op.dimensions), op.scale, op.zeroPoint, blob->buffer().as<float*>());
                            break;
                        case OperandType::TENSOR_QUANT8_SYMM:
                            deQuantize<int8_t>(buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                            break;
                        case OperandType::TENSOR_QUANT16_SYMM:
                            deQuantize<int16_t>(buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                            break;
                        case OperandType::TENSOR_INT32:
                            deQuantize<int32_t>(buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                            break;
                        default:
                            nnAssert(true);
                            break;
                    }
                } else {
                    blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
                    blob->allocate();
                }
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
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), InferenceEngine::Layout::ANY);
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

// Redundant.. Remove the code
IRBlob::Ptr NnapiModelInfo::GetConstWeightsOperandAsTensor(uint32_t index) {
    dumpOperand(index, mModel);

    const auto op = mModel.main.operands[index];
    bool isQuantInput = false;
    if (op.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED ||
            op.type == OperandType::TENSOR_QUANT8_SYMM ||
            op.type == OperandType::TENSOR_QUANT16_SYMM ) {
        isQuantInput = true;
    }

    uint32_t len;
    const uint8_t* buf = GetOperandMemory(index, len);
    VLOG(L1, "NnapiModelInfo:: Operand: index: %d, len: %d, buf: %p", index, len, buf);
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32 || isQuantInput) {
        vec<unsigned int> order;
        InferenceEngine::Layout layout;
        if (op.dimensions.size() == 4) {
            // order = {0,3,1,2};  //nhwc -> nchw
            order = {3, 0, 1, 2};   // IHWO -> OIHW for depth conv
            layout = InferenceEngine::Layout::OIHW;  // weights layout
        } else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = InferenceEngine::Layout::NC;
        } else {
            order = {0};  //(op.dimensions.size() < 2)
            layout = InferenceEngine::Layout::C;
        }
        auto inputDims = toDims(op.dimensions);
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
        if (buf == nullptr) {
            VLOG(L1, "TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob =
                std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();
            return blob;
        } else {
            if (inputDims.size() != 4) {
                InferenceEngine::TBlob<float>::Ptr blob = nullptr;
                if (isQuantInput) {
                    blob = std::make_shared<InferenceEngine::TBlob<float>>(td);
                    blob->allocate();

                    switch(static_cast<int>(op.type)) {
                        case static_cast<int>(OperandType::TENSOR_QUANT8_ASYMM_SIGNED):
                            deQuantize<int8_t>(buf, getNumberOfElements(op.dimensions), op.scale, op.zeroPoint, blob->buffer().as<float*>());
                            break;
                        case static_cast<int>(V1_2::OperandType::TENSOR_QUANT8_SYMM):
                            deQuantize<int8_t>(buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>());
                            break;
                        case static_cast<int>(V1_2::OperandType::TENSOR_QUANT16_SYMM):
                            deQuantize<int16_t>(buf, getNumberOfElements(op.dimensions), op.scale, 0, blob->buffer().as<float*>()); // Ugly hack reverting
                            break;
                        default:
                            ALOGE("Failed dequantize for type: %d", op.type);
                            break;
                    }
                }
                else {
                    InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
                    blob->allocate();
                }

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
        InferenceEngine::TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), InferenceEngine::Layout::ANY);
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

bool NnapiModelInfo::setRunTimePoolInfosFromHidlMemories(const hidl_vec<V1_3::Request::MemoryPool>& pools) {
   ALOGD("Number of pools: %d", pools.size());
   mRequestPoolInfos.resize(pools.size());
    for (size_t i = 0; i < pools.size(); i++) {        
         auto& poolInfo = mRequestPoolInfos[i];
         if (!poolInfo.set(pools[i].hidlMemory())) {
            LOG(ERROR) << "Could not map memory pool !!!";
            return false;
        }
    }
    return true;
}

Blob::Ptr NnapiModelInfo::getBlobFromMemoryPoolIn(const V1_3::Request& request, uint32_t index) {
    RunTimeOperandInfo& operand = mOperands[mModel.main.inputIndexes[index]];
    const V1_0::RequestArgument& arg = request.inputs[index];
    auto poolIndex = arg.location.poolIndex;
    nnAssert(poolIndex < mRequestPoolInfos.size());
    auto& r = mRequestPoolInfos[poolIndex];

    if (arg.dimensions.size() > 0) {
            // It's the responsibility of the caller to validate that
            // from.dimensions only modifies the dimensions that were
            // unspecified in the model.main.  That's the case in SampleDriver.cpp
            // with the call to validateRequest().
            operand.dimensions = arg.dimensions;
    }

    operand.buffer = r.buffer + arg.location.offset;
    operand.length = arg.location.length;
    ALOGI("%s Operand length:%d pointer:%p offset:%d pool index: %d ignore layout=%d", __func__,
            operand.length, (r.buffer + arg.location.offset), arg.location.offset, poolIndex, operand.ignoreLayout);

    return GetInOutOperandAsBlob(operand,
                                const_cast<uint8_t*>(r.buffer + arg.location.offset),
                                operand.length,
                                operand.ignoreLayout);
}

void* NnapiModelInfo::getBlobFromMemoryPoolOut(const V1_3::Request& request, uint32_t index) {
    RunTimeOperandInfo& operand = mOperands[mModel.main.outputIndexes[index]];
    const V1_0::RequestArgument& arg = request.outputs[index];
    auto poolIndex = arg.location.poolIndex;
    nnAssert(poolIndex < mRequestPoolInfos.size());
    auto& r = mRequestPoolInfos[poolIndex];

    ALOGD("%s lifetime:%d location offset:%d length:%d pool index:%d", __func__, operand.lifetime, arg.location.offset, arg.location.length, poolIndex);

    if (arg.dimensions.size() > 0) {
            // It's the responsibility of the caller to validate that
            // from.dimensions only modifies the dimensions that were
            // unspecified in the model.main.  That's the case in SampleDriver.cpp
            // with the call to validateRequest().
            operand.dimensions = arg.dimensions;
    }

    operand.buffer = r.buffer + arg.location.offset;
    operand.length = arg.location.length;
    ALOGI("%s Operand length:%d pointer:%p", __func__, operand.length, (r.buffer + arg.location.offset));
    return (r.buffer + arg.location.offset);
}

// template int NnapiModelInfo::ParseOperationInput<int32_t>(
//         android::hardware::neuralnetworks::V1_2::Operation const&, unsigned int);
// template float NnapiModelInfo::ParseOperationInput<float>(
//         android::hardware::neuralnetworks::V1_2::Operation const&, unsigned int);

template int NnapiModelInfo::GetConstOperand<int>(unsigned int);
template unsigned int NnapiModelInfo::GetConstOperand<unsigned int>(unsigned int);
template int NnapiModelInfo::GetConstFromBuffer<int>(unsigned char const*, unsigned int);

// template std::__1::vector<unsigned int, std::__1::allocator<unsigned int> >
//         NnapiModelInfo::GetConstVecOperand<unsigned int>(unsigned int);
// template std::__1::vector<float, std::__1::allocator<float> >
//         NnapiModelInfo::GetConstVecOperand<unsigned float>(unsigned int);
} // namespace nnhal
} // namespace neuralnetworks
} // namespace hardware
} // namespace android
