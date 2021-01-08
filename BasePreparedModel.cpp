/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "BasePreparedModel.h"

#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <thread>
#include "ValidateHal.h"

#ifdef __ANDROID__
#include <cutils/properties.h>
#endif

#define DISABLE_ALL_QUANT
#define LOG_TAG "BasePreparedModel"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using namespace android::nn;

static const Timing kNoTiming = {.timeOnDevice = UINT64_MAX, .timeInDriver = UINT64_MAX};

void BasePreparedModel::deinitialize() {
    VLOG(L1, "deinitialize");
    delete enginePtr;
    enginePtr = nullptr;
    VLOG(L1, "free engine");
}

// TODO: short term, make share memory mapping and updating a utility function.
// TODO: long term, implement mmap_fd as a hidl IMemory service.
bool RunTimePoolInfo::set(const hidl_memory& hidlMemory) {
    this->hidlMemory = hidlMemory;
    auto memType = hidlMemory.name();
    if (memType == "ashmem") {
        memory = mapMemory(hidlMemory);
        if (memory == nullptr) {
            LOG(ERROR) << "Can't map shared memory.";
            return false;
        }
        memory->update();
        buffer = reinterpret_cast<uint8_t*>(static_cast<void*>(memory->getPointer()));
        if (buffer == nullptr) {
            LOG(ERROR) << "Can't access shared memory.";
            return false;
        }
        return true;
    } else if (memType == "mmap_fd") {
        size_t size = hidlMemory.size();
        int fd = hidlMemory.handle()->data[0];
        int prot = hidlMemory.handle()->data[1];
        size_t offset = getSizeFromInts(hidlMemory.handle()->data[2], hidlMemory.handle()->data[3]);
        buffer = static_cast<uint8_t*>(mmap(nullptr, size, prot, MAP_SHARED, fd, offset));
        if (buffer == MAP_FAILED) {
            LOG(ERROR) << "Can't mmap the file descriptor.";
            return false;
        }
        return true;
    } else {
        LOG(ERROR) << "unsupported hidl_memory type";
        return false;
    }
}

// Making sure the output data are correctly updated after execution.
bool RunTimePoolInfo::update() {
    auto memType = hidlMemory.name();
    if (memType == "ashmem") {
        memory->commit();
        return true;
    } else if (memType == "mmap_fd") {
        int prot = hidlMemory.handle()->data[1];
        if (prot & PROT_WRITE) {
            size_t size = hidlMemory.size();
            return msync(buffer, size, MS_SYNC) == 0;
        }
    }
    // No-op for other types of memory.
    return true;
}

bool setRunTimePoolInfosFromHidlMemories(std::vector<RunTimePoolInfo>* poolInfos,
                                         const hidl_vec<hidl_memory>& pools) {
    poolInfos->resize(pools.size());
    for (size_t i = 0; i < pools.size(); i++) {
        auto& poolInfo = (*poolInfos)[i];
        if (!poolInfo.set(pools[i])) {
            LOG(ERROR) << "Could not map pool";
            return false;
        }
    }
    return true;
}

template <typename T>
T getScalarData(const RunTimeOperandInfo& info) {
    // TODO: Check buffer is at least as long as size of data.
    T* data = reinterpret_cast<T*>(info.buffer);
    return data[0];
}

inline int sizeOfData(OperandType type, std::vector<uint32_t> dims) {
    int size;
    switch (type) {
        case OperandType::FLOAT32:
            size = 4;
            break;
        case OperandType::TENSOR_FLOAT32:
            size = 4;
            break;
        case OperandType::TENSOR_INT32:
            size = 4;
            break;
        case OperandType::TENSOR_QUANT8_ASYMM:
        case OperandType::INT32:
            size = 1;
            break;

        default:
            size = 0;
    }
    for (auto d : dims) size *= d;

    return size;
}

// Updates the RunTimeOperandInfo with the newly calculated shape.
// Allocate the buffer if we need to.
static bool setInfoAndAllocateIfNeeded(RunTimeOperandInfo* info, const Shape& shape) {
    // For user-provided model output operands, the parameters must match the Shape
    // calculated from the preparation step.
    if (info->lifetime == OperandLifeTime::MODEL_OUTPUT) {
        if (info->type != shape.type || info->dimensions != shape.dimensions) {
            LOG(ERROR) << "Invalid type or dimensions for model output";
            return false;
        }
        if (info->type == OperandType::TENSOR_QUANT8_ASYMM &&
            (info->scale != shape.scale || info->zeroPoint != shape.offset)) {
            LOG(ERROR) << "Invalid scale or zeroPoint for model output";
            return false;
        }
    }
    info->type = shape.type;
    info->dimensions = shape.dimensions;
    info->scale = shape.scale;
    info->zeroPoint = shape.offset;
    if (info->lifetime == OperandLifeTime::TEMPORARY_VARIABLE && info->buffer == nullptr) {
        uint32_t length = sizeOfData(info->type, info->dimensions);
        info->buffer = new uint8_t[length];
        if (info->buffer == nullptr) {
            return false;
        }
    }
    return true;
}

bool BasePreparedModel::isOperationSupported(const Operation& operation, const Model& model) {
    VLOG(L1, "Check operation %d", operation.type);

#define VLOG_CHECKFAIL(fail) VLOG(L1, "Check failed: %s", fail)

    switch (operation.type) {

        case OperationType::ADD: {
            if(!add::validate(operation, model))
                return false;
        } break;
        case OperationType::AVERAGE_POOL_2D: {
            if(!avgpool::validate(operation, model))
                return false;
        } break;
        case OperationType::FULLY_CONNECTED: {
            if(!avgpool::validate(operation, model))
                return false;
        } break;
        case OperationType::MUL:{
            if(!mul::validate(operation, model))
                return false;
        } break;
        case OperationType::MAX_POOL_2D: {
            if(!maxpool::validate(operation, model))
                return false;
        } break;
        case OperationType::CONCATENATION:{
            if(!concat::validate(operation, model))
                return false;
        } break;
        case OperationType::CONV_2D:{
            if(!convolution::validate(operation, model))
                return false;
        } break;
        case OperationType::DEPTHWISE_CONV_2D:{
            if(!depthconv::validate(operation, model))
                return false;
        } break;
        case OperationType::L2_NORMALIZATION:{
            if(!l2normalization::validate(operation, model))
                return false;
        } break;
        case OperationType::LOCAL_RESPONSE_NORMALIZATION:{
            if(!lrn::validate(operation, model))
                return false;
        } break;
        case OperationType::RELU:{
            if(!relu::validate(operation, model))
                return false;
        } break;
        case OperationType::RELU1:{
            if(!relu1::validate(operation, model))
                return false;
        } break;
        case OperationType::RELU6:{
            if(!relu6::validate(operation, model))
                return false;
        } break;
        case OperationType::LOGISTIC:{
            if(!logistic::validate(operation, model))
                return false;
        } break;
        case OperationType::SOFTMAX:{
            if(!softmax::validate(operation, model))
                return false;
        } break;
        case OperationType::TANH:{
            if(!softmax::validate(operation, model))
                return false;
        } break;
        case OperationType::RESHAPE:{
            if(!reshape::validate(operation, model))
                return false;
        } break;
        default:
            VLOG(L1, "unsupport operation %d", operation.type);
            return false;
    }

#ifdef DISABLE_ALL_QUANT
    for (auto i : operation.inputs) {
        const auto input = model.operands[i];
        if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            VLOG_CHECKFAIL("input quant");
            return false;
        }
    }
    for (auto i : operation.outputs) {
        const auto output = model.operands[i];
        if (output.type == OperandType::TENSOR_QUANT8_ASYMM) {
            VLOG_CHECKFAIL("output quant");
            return false;
        }
    }
#endif

    VLOG(L1, "Operation %d supported by driver", operation.type);

    return true;
}

bool BasePreparedModel::initialize(const Model& model) {
    VLOG(L1, "initialize");
    return true;
}

void BasePreparedModel::initializeInput() {
    VLOG(L1, "initialize Input");
    for (auto i : mModel.inputIndexes) {
        int dims_size = mOperands[i].dimensions.size();

        VLOG(L1, "mPorts[%d] %s dims size %d", i, mPorts[i]->getName().c_str(), dims_size);
        VLOGDIMS(L1, mOperands[i].dimensions, "current operand inpu dims:");
        VLOGDIMS(L1, mPorts[i]->getTensorDesc().getDims(), "Real input dims:");

        auto inputDims = mPorts[i]->getTensorDesc().getDims();

        uint32_t nelem = getNumberOfElements(mOperands[i].dimensions);
        auto inputElem = sizeOfTensor(inputDims);
        if (nelem != inputElem) {
            VLOG(L1, "set operand input dims to real input dims\n");
            for (auto j = 0; j < inputDims.size(); j++)
                mOperands[i].dimensions[j] = static_cast<uint32_t>(inputDims[j]);
            mOperands[i].length = sizeOfData(mOperands[i].type, mOperands[i].dimensions);
        }
    }
}

bool BasePreparedModel::finalizeOutput(/*RunTimeOperandInfo* output */) {
    VLOG(L1, "finalize Output");
    for (auto i : mModel.outputIndexes) {
        int dims_size = mOperands[i].dimensions.size();

        mPorts[i]->setPrecision(InferenceEngine::Precision::FP32);
        mNet.addOutput(mPorts[i]);
        mCreateNgraph->setResultNode(mPorts[i]->getName());

        VLOG(L1, "mPorts[%d] %s dims size %d", i, mPorts[i]->getName().c_str(), dims_size);
        VLOGDIMS(L1, mOperands[i].dimensions, "current operand Output dims:");
        VLOGDIMS(L1, mPorts[i]->getTensorDesc().getDims(), "Real Output dims:");

        auto outputDims = mPorts[i]->getTensorDesc().getDims();

        uint32_t nelem = getNumberOfElements(mOperands[i].dimensions);
        auto outputElem = sizeOfTensor(outputDims);
        if (nelem != outputElem) {
            VLOG(L1, "set correct dims as operand output dims different than real output dims\n");
        }
    }
    return true;
}

template <typename T>
T BasePreparedModel::ParseOperationInput(const Model& model, const Operation& operation,
                                     uint32_t index) {
    uint32_t inputIndex = operation.inputs[index];
    const auto operand = mModel.operands[inputIndex];
    const auto value = GetConstOperand<T>(model, inputIndex);
    VLOG(L1, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
    VLOG(L1, "Operation input index: %d, operand index: %d", index, inputIndex);
    VLOG(L1, "Operation: %s", toString(operation).c_str());
    printHelper<T>::print(value, toString(operand).c_str());
    VLOG(L1, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");

    return value;
}

const uint8_t* BasePreparedModel::GetOperandMemory(const Model& model, uint32_t index,
                                               uint32_t& len_out) {
    const auto op = model.operands[index];
    len_out = op.location.length;
    if (op.lifetime == OperandLifeTime::CONSTANT_COPY) {
        if (op.location.poolIndex != 0) {
            ALOGE("CONSTANT_COPY expects poolIndex to be 0");
            nnAssert(false);
        }
        VLOG(L1, "operand lifetime OperandLifeTime::CONSTANT_COPY");
        return (const_cast<uint8_t*>(&model.operandValues[op.location.offset]));
        // to.numberOfUsesLeft = 0;
    } else if (op.lifetime == OperandLifeTime::CONSTANT_REFERENCE) {
        VLOG(L1, "operand lifetime OperandLifeTime::CONSTANT_REFERENCE");
        auto poolIndex = op.location.poolIndex;
        // nnAssert(poolIndex < mPoolInfos.size()); //aks fix me
        auto& r = mPoolInfos[poolIndex];
        return (const_cast<uint8_t*>(r.buffer + op.location.offset));
    } else if (op.lifetime == OperandLifeTime::MODEL_INPUT ||
               op.lifetime == OperandLifeTime::MODEL_OUTPUT ||
               op.lifetime == OperandLifeTime::NO_VALUE) {
        VLOG(L1, "operand lifetime OperandLifeTime::MODEL_INPUT||MODEL_OUTPUT||NO_VALUE");
        len_out = sizeOfData(op.type, op.dimensions);
        return nullptr;
    } else if (op.lifetime == OperandLifeTime::TEMPORARY_VARIABLE) {
        VLOG(L1, "operand lifetime OperandLifeTime::TEMPORARY_VARIABLE");
        VLOG(L1, "operand is expected to be const, but lifetime is %d", op.lifetime);
        len_out = sizeOfData(op.type, op.dimensions);
        // nnAssert(false);
        return nullptr;
    }

    ALOGE("operand is expected to be const, but lifetime is %d", op.lifetime);
    nnAssert(false);  // temp fix since some time const operand set as TEMPORARY_VARIABLE
    return nullptr;
}

template <typename T>
T BasePreparedModel::GetConstOperand(const Model& model, uint32_t index) {
    dumpOperand(index, model);
    uint32_t len;
    const uint8_t* buf = GetOperandMemory(model, index, len);
    return GetConstFromBuffer<T>(buf, len);
}

template <typename T>
std::vector<T> BasePreparedModel::GetConstVecOperand(const Model& model, uint32_t index) {
    dumpOperand(index, model);
    uint32_t len;
    const uint8_t* buf = GetOperandMemory(model, index, len);
    return GetConstVecFromBuffer<T>(buf, len);
}

template <typename T>
T BasePreparedModel::GetConstFromBuffer(const uint8_t* buf, uint32_t len) {
    VLOG(L1, "buf: %p, len: %d", buf, len);
    if (len != sizeof(T)) {
        VLOG(L1, "fix me: typeid(T).name() should be %d bytes", sizeof(T));
        // fix me if buffer is of type float and if float and OperandLifeTime::CONSTANT_REFERENCE
        nnAssert(false);
    }
    return *(T*)(buf);
}

template <typename T>
std::vector<T> BasePreparedModel::GetConstVecFromBuffer(const uint8_t* buf, uint32_t len) {
    int n = len / sizeof(T);
    if (n * sizeof(T) != len) {
        VLOG(L1, "typeid(T).name() should be  multiples of %d bytes", sizeof(T));
        nnAssert(false);
    }

    std::vector<T> ret;

    for (int i = 0; i < n; i++) {
        ret.push_back(*(T*)buf);
        buf += sizeof(T);
    }

    return ret;
}

bool BasePreparedModel::isConst(int index, const Model& model) {
    VLOG(L1, "---------------------------------------------");
    VLOG(L1, "Operand index: %d", index);
    const auto op = model.operands[index];
    VLOG(L1, " %s", toString(op).c_str());
    bool ret = (op.lifetime == OperandLifeTime::CONSTANT_COPY ||
                op.lifetime == OperandLifeTime::CONSTANT_REFERENCE);
    VLOG(L1, "%s", ret ? "Const" : "Non-Const");
    VLOG(L1, "---------------------------------------------");
    return ret;
}

OutputPort BasePreparedModel::getPort(int index, const Model& model) {
    VLOG(L1, "getPort\n");
    if (isConst(index, model)) {
        VLOG(L1, "index is a const!");
        nnAssert(false);
    }
    const auto op = model.operands[index];
    if (op.lifetime == OperandLifeTime::MODEL_INPUT) {
        VLOG(L1, "Model input operand\n");
        std::ostringstream operandName;
        operandName << "input" << index;

        vec<unsigned int> order;
        if (op.dimensions.size() == 4)
            order = {0, 3, 1, 2};  // nhwc -> nchw
        else if (op.dimensions.size() == 2)
            order = {0, 1};
        else
            order = {0};  //(op.dimensions.size() < 2)

        auto operandInfo = mNet.createInput(
            operandName.str(), permuteDims(toDims(op.dimensions), order));  // NHWC -> NCHW
        mPorts[index] = operandInfo->getInputData();
        mCreateNgraph->addInputParameter(operandName.str(),
                                         mPorts[index]->getTensorDesc().getDims());
        // TODO: workaround 3-D
        int dims_size = op.dimensions.size();

        VLOG(L1, "mPorts[%d] %s dims size %d", index, mPorts[index]->getName().c_str(), dims_size);

        auto dims = permuteDims(toDims(op.dimensions), order);
        // auto dims = toDims(op.dimensions);
        for (auto i = 0; i < dims.size(); i++)
            VLOG(L1, "input dims[%d] = %d & set input dims[%d] = %d ", i, op.dimensions[i], i,
                 dims[i]);

        switch (dims_size) {
            case 2:
                mPorts[index]->setLayout(NC);
                break;
            case 4:
                mPorts[index]->setLayout(NCHW);
                break;
            case 1:
                mPorts[index]->setLayout(C);
                break;
            default:
                VLOG(L1, "unsupported dims size %d", dims_size);
                nnAssert(false);
        }

        return mPorts[index];
    }
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        VLOG(L1, "Model output expected as input, not possible");
        nnAssert(false);
    }
    if (op.lifetime == OperandLifeTime::NO_VALUE) {
        VLOG(L1, "port is expected to be allocated for this as output from other layer");
        nnAssert(false);
    }
    if (op.lifetime == OperandLifeTime::TEMPORARY_VARIABLE) {
        VLOG(L1, "getport OperandLifeTime::TEMPORARY_VARIABLE\n");
        if (!mPorts[index]) nnAssert(false);
        VLOG(L1, "mPorts[%d] already allocated\n", index);
        return mPorts[index];
    }

    return nullptr;
}

bool BasePreparedModel::initializeRunTimeOperandInfo() {
    // initialize runtime operand info from model.
    const size_t count = mModel.operands.size();
    if (!count) {
        VLOG(L1, "NNERR:Operand Count is 0");
        return false;
    }
    mOperands.resize(count);
    mPorts.resize(count);
    // TensorDims dims;

    // Start by setting the runtime info to what's in the model.
    for (size_t i = 0; i < count; i++) {
        const V1_2::Operand& from = mModel.operands[i];
        RunTimeOperandInfo& to = mOperands[i];
        to.dimensions.resize(from.dimensions.size());
        for (size_t j = 0; j < from.dimensions.size(); j++) {
            to.dimensions[j] = from.dimensions[j];
        }

        to.scale = from.scale;
        switch (from.type) {
            case OperandType::TENSOR_FLOAT32:
            case OperandType::FLOAT32:
                to.type = OperandType::TENSOR_FLOAT32;
                VLOG(L1, "OperandType = %d\n", from.type);
                break;
            case OperandType::INT32:
            case OperandType::UINT32:
                nnAssert(to.scale == 0);
            case OperandType::TENSOR_INT32:
                to.type = from.type;
                break;
            case OperandType::TENSOR_QUANT8_ASYMM:
                ALOGE("OperandType::TENSOR_QUANT8_ASYMM is not supported");
                break;
            default:
                ALOGE("wrong operand type %d", from.type);
                return false;
        }

        to.length = from.location.length;
        to.lifetime = from.lifetime;
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
            case OperandLifeTime::MODEL_INPUT:
            case OperandLifeTime::MODEL_OUTPUT:
            case OperandLifeTime::NO_VALUE:
                to.buffer = nullptr;
                to.numberOfUsesLeft = 0;
                break;
            default:
                return false;
                break;
        }
    }
    return true;
}

static Return<void> notify(const sp<V1_0::IExecutionCallback>& callback, const ErrorStatus& status,
                           const hidl_vec<OutputShape>&, Timing) {
    return callback->notify(status);
}

static Return<void> notify(const sp<V1_2::IExecutionCallback>& callback, const ErrorStatus& status,
                           const hidl_vec<OutputShape>& outputShapes, Timing timing) {
    return callback->notify_1_2(status, outputShapes, timing);
}

template <typename T_IExecutionCallback>
Return<ErrorStatus> BasePreparedModel::executeBase(const Request& request, MeasureTiming measure,
                                               const sp<T_IExecutionCallback>& callback) {
    VLOG(L1, "executebase");

    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateRequest(request, mModel)) {
        notify(callback, ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the driver service
    // is expected to live forever.
    std::thread([this, request, measure, driverStart, callback] {
        asyncExecute(request, measure, driverStart, callback);
    }).detach();

    return ErrorStatus::NONE;
}

template <typename T_IExecutionCallback>
void BasePreparedModel::asyncExecute(const Request& request, MeasureTiming measure,
                                 time_point driverStart,
                                 const sp<T_IExecutionCallback>& callback) {
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!setRunTimePoolInfosFromHidlMemories(&requestPoolInfos, request.pools)) {
        notify(callback, ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return;
    }

    time_point driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();

    auto inOutData = [this, &requestPoolInfos](const std::vector<uint32_t>& indexes,
                                               const hidl_vec<RequestArgument>& arguments,
                                               bool inputFromRequest, ExecuteNetwork* enginePtr,
                                               std::vector<OutputPort> mPorts) {
        // do memcpy for input data
        for (size_t i = 0; i < indexes.size(); i++) {
            RunTimeOperandInfo& operand = mOperands[indexes[i]];
            const RequestArgument& arg = arguments[i];
            auto poolIndex = arg.location.poolIndex;
            nnAssert(poolIndex < requestPoolInfos.size());
            auto& r = requestPoolInfos[poolIndex];
            if (arg.dimensions.size() > 0) {
                // It's the responsibility of the caller to validate that
                // from.dimensions only modifies the dimensions that were
                // unspecified in the model.  That's the case in SampleDriver.cpp
                // with the call to validateRequest().
                operand.dimensions = arg.dimensions;
            }
            operand.buffer = r.buffer + arg.location.offset;  // r.getBuffer()
            operand.length = arg.location.length;  // sizeOfData(operand.type, operand.dimensions);

            VLOG(L1, "Copy request input/output to model input/output");
            if (inputFromRequest) {
                auto inputBlob = GetInOutOperandAsBlob(
                    operand, const_cast<uint8_t*>(r.buffer + arg.location.offset),
                    operand.length);  // if not doing memcpy
                VLOG(L1, "Copy inputBlob for mPorts[%d]->name %s", indexes[i],
                     mPorts[indexes[i]]->getName().c_str());

                auto destBlob = enginePtr->getBlob(mPorts[indexes[i]]->getName());
                uint8_t* dest = destBlob->buffer().as<uint8_t*>();
                uint8_t* src = inputBlob->buffer().as<uint8_t*>();
                std::memcpy(dest, src, inputBlob->byteSize());
            } else {
                auto outputBlob = GetInOutOperandAsBlob(
                    operand, const_cast<uint8_t*>(r.buffer + arg.location.offset),
                    operand.length);  // if not doing memcpy
                VLOG(L1, "copyData from IE to Android blob for mPorts[%d]->name %s", indexes[i],
                     mPorts[indexes[i]]->getName().c_str());
                auto srcBlob = enginePtr->getBlob(mPorts[indexes[i]]->getName());
                uint8_t* dest = outputBlob->buffer().as<uint8_t*>();
                uint8_t* src = srcBlob->buffer().as<uint8_t*>();
                std::memcpy(dest, src, outputBlob->byteSize());
            }
        }
    };

    VLOG(L1, "pass request inputs buffer to network/model respectively");

    inOutData(mModel.inputIndexes, request.inputs, true, enginePtr, mPorts);
    VLOG(L1, "Run");

    enginePtr->Infer();

    VLOG(L1, "pass request outputs buffer to network/model respectively");
    inOutData(mModel.outputIndexes, request.outputs, false, enginePtr, mPorts);

    if (measure == MeasureTiming::YES) deviceEnd = now();

    VLOG(L1, "update shared memories");
    for (auto runtimeInfo : requestPoolInfos) {
        runtimeInfo.update();
    }

    InferenceEngine::TBlob<float>::Ptr outBlob =
        enginePtr->getBlob(mPorts[mModel.outputIndexes[0]]->getName());

    InferenceEngine::TBlob<float>::Ptr inBlob =
        enginePtr->getBlob(mPorts[mModel.inputIndexes[0]]->getName());
    hidl_vec<OutputShape> outputShapes;
#ifdef NN_DEBUG
    {
        VLOG(L1, "Model output0 are:");

        auto nelem = (outBlob->size() > 20 ? 20 : outBlob->size());
        for (int i = 0; i < nelem; i++) {
            VLOG(L1, "outBlob elements %d = %f", i, outBlob->readOnly()[i]);
        }

        VLOG(L1, "Model input0 are:");

        nelem = (inBlob->size() > 20 ? 20 : inBlob->size());
        for (int i = 0; i < nelem; i++) {
            VLOG(L1, "inBlob elements %d = %f", i, inBlob->readOnly()[i]);
        }
    }
#endif
    Return<void> returned;
    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        // VLOG(L1, "Driver::asyncExecute timing = %s", toString(timing));
        returned = notify(callback, ErrorStatus::NONE, outputShapes, timing);
    } else {
        returned = notify(callback, ErrorStatus::NONE, outputShapes, kNoTiming);
    }
    if (!returned.isOk()) {
        ALOGE("hidl callback failed to return properly: %s", returned.description().c_str());
    }
}

Return<void> BasePreparedModel::executeSynchronously(const Request& request, MeasureTiming measure,
                                                 executeSynchronously_cb cb) {
    VLOG(L1, "Begin to executeSynchronously");
    time_point driverStart, driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (!validateRequest(request, mModel)) {
        cb(ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return Void();
    }
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!setRunTimePoolInfosFromHidlMemories(&requestPoolInfos, request.pools)) {
        cb(ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return Void();
    }

    auto inOutData = [this, &requestPoolInfos](const std::vector<uint32_t>& indexes,
                                               const hidl_vec<RequestArgument>& arguments,
                                               bool inputFromRequest, ExecuteNetwork* enginePtr,
                                               std::vector<OutputPort> mPorts) {
        // do memcpy for input data
        for (size_t i = 0; i < indexes.size(); i++) {
            RunTimeOperandInfo& operand = mOperands[indexes[i]];
            const RequestArgument& arg = arguments[i];
            auto poolIndex = arg.location.poolIndex;
            nnAssert(poolIndex < requestPoolInfos.size());
            auto& r = requestPoolInfos[poolIndex];
            if (arg.dimensions.size() > 0) {
                // It's the responsibility of the caller to validate that
                // from.dimensions only modifies the dimensions that were
                // unspecified in the model.  That's the case in SampleDriver.cpp
                // with the call to validateRequest().
                operand.dimensions = arg.dimensions;
            }
            operand.buffer = r.buffer + arg.location.offset;  // r.getBuffer()
            operand.length = arg.location.length;  // sizeOfData(operand.type, operand.dimensions);
            VLOG(L1, "Copy request input/output to model input/output");
            if (inputFromRequest) {
                auto inputBlob = GetInOutOperandAsBlob(
                    operand, const_cast<uint8_t*>(r.buffer + arg.location.offset),
                    operand.length);  // if not doing memcpy
                VLOG(L1, "Copy inputBlob for mPorts[%d]->name %s", indexes[i],
                     mCreateNgraph->getNodeName(mPorts[indexes[i]]->getName()).c_str());
                auto destBlob = (mUseNgraph == true)
                                    ? enginePtr->getBlob(
                                          mCreateNgraph->getNodeName(mPorts[indexes[i]]->getName()))
                                    : enginePtr->getBlob(mPorts[indexes[i]]->getName());
                uint8_t* dest = destBlob->buffer().as<uint8_t*>();
                uint8_t* src = inputBlob->buffer().as<uint8_t*>();
                std::memcpy(dest, src, inputBlob->byteSize());
            } else {
                VLOG(L1, "copyData from IE to Android blob for mPorts[%d]->name %s", indexes[i],
                     mCreateNgraph->getNodeName(mPorts[indexes[i]]->getName()).c_str());
                auto srcBlob = (mUseNgraph == true)
                                   ? enginePtr->getBlob(
                                         mCreateNgraph->getNodeName(mPorts[indexes[i]]->getName()))
                                   : enginePtr->getBlob(mPorts[indexes[i]]->getName());
                auto outputBlob = GetInOutOperandAsBlob(
                    operand, const_cast<uint8_t*>(r.buffer + arg.location.offset),
                    operand.length);  // if not doing memcpy
                uint8_t* dest = outputBlob->buffer().as<uint8_t*>();
                uint8_t* src = srcBlob->buffer().as<uint8_t*>();
                std::memcpy(dest, src, outputBlob->byteSize());
            }
        }
    };

    VLOG(L1, "pass request inputs buffer to network/model respectively");

    inOutData(mModel.inputIndexes, request.inputs, true, enginePtr, mPorts);
    VLOG(L1, "Run");

    enginePtr->Infer();

    VLOG(L1, "pass request outputs buffer to network/model respectively");
    inOutData(mModel.outputIndexes, request.outputs, false, enginePtr, mPorts);

    if (measure == MeasureTiming::YES) deviceEnd = now();

    VLOG(L1, "update shared memories");
    for (auto runtimeInfo : requestPoolInfos) {
        runtimeInfo.update();
    }

    InferenceEngine::TBlob<float>::Ptr outBlob =
        (mUseNgraph == true) ? enginePtr->getBlob(mCreateNgraph->getNodeName(
                                   mPorts[mModel.outputIndexes[0]]->getName()))
                             : enginePtr->getBlob(mPorts[mModel.outputIndexes[0]]->getName());

    InferenceEngine::TBlob<float>::Ptr inBlob =
        (mUseNgraph == true) ? enginePtr->getBlob(mCreateNgraph->getNodeName(
                                   mPorts[mModel.inputIndexes[0]]->getName()))
                             : enginePtr->getBlob(mPorts[mModel.inputIndexes[0]]->getName());
    hidl_vec<OutputShape> outputShapes;
#ifdef NN_DEBUG
    {
        VLOG(L1, "Model output0 are:");

        auto nelem = (outBlob->size() > 20 ? 20 : outBlob->size());
        for (int i = 0; i < nelem; i++) {
            VLOG(L1, "outBlob elements %d = %f", i, outBlob->readOnly()[i]);
        }

        VLOG(L1, "Model input0 are:");

        nelem = (inBlob->size() > 20 ? 20 : inBlob->size());
        for (int i = 0; i < nelem; i++) {
            VLOG(L1, "inBlob elements %d = %f", i, inBlob->readOnly()[i]);
        }
    }
#endif
    Return<void> returned;
    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        VLOG(L1, "Driver::executeSynchronously timing = %s", timing);
        cb(ErrorStatus::NONE, outputShapes, timing);
    } else {
        cb(ErrorStatus::NONE, outputShapes, kNoTiming);
    }
    return Void();
}

Return<void> BasePreparedModel::configureExecutionBurst(
    const sp<V1_2::IBurstCallback>& callback,
    const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
    const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel, configureExecutionBurst_cb cb) {
    VLOG(L1, "Driver::configureExecutionBurst");

    cb(ErrorStatus::GENERAL_FAILURE, {});
    return Void();
}

Return<ErrorStatus> BasePreparedModel::execute(const Request& request,
                                           const sp<V1_0::IExecutionCallback>& callback) {
    VLOG(L1, "Begin to execute");
    return executeBase(request, MeasureTiming::NO, callback);
}

Return<ErrorStatus> BasePreparedModel::execute_1_2(const Request& request, MeasureTiming measure,
                                               const sp<V1_2::IExecutionCallback>& callback) {
    VLOG(L1, "Begin to execute_1_2");
    return executeBase(request, measure, callback);
}

IRBlob::Ptr BasePreparedModel::GetConstWeightsOperandAsTensor(uint32_t index, const Model& model) { return nullptr; }

IRBlob::Ptr BasePreparedModel::GetConstOperandAsTensor(int operand_idx, int operation_idx, const Model& model) {
    return nullptr;
}

Blob::Ptr BasePreparedModel::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                               uint32_t& len) {
    return nullptr;
}

template int BasePreparedModel::ParseOperationInput<int>(
        android::hardware::neuralnetworks::V1_2::Model const&,
        android::hardware::neuralnetworks::V1_2::Operation const&, unsigned int);
template float BasePreparedModel::ParseOperationInput<float>(
        android::hardware::neuralnetworks::V1_2::Model const&,
        android::hardware::neuralnetworks::V1_2::Operation const&, unsigned int);

template int BasePreparedModel::GetConstOperand<int>(
        android::hardware::neuralnetworks::V1_2::Model const&, unsigned int);
template int BasePreparedModel::GetConstFromBuffer<int>(unsigned char const*, unsigned int);
template std::__1::vector<unsigned int, std::__1::allocator<unsigned int> >
        BasePreparedModel::GetConstVecOperand<unsigned int>(
                android::hardware::neuralnetworks::V1_2::Model const&, unsigned int);
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
