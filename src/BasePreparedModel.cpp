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
#include "NgraphOpsFactory.h"
#include "DevicePlugin.h"

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

static const V1_2::Timing kNoTiming = {.timeOnDevice = UINT64_MAX, .timeInDriver = UINT64_MAX};

void BasePreparedModel::deinitialize() {
    VLOG(L1, "deinitialize");
}

template <typename T>
T getScalarData(const RunTimeOperandInfo& info) {
    // TODO: Check buffer is at least as long as size of data.
    T* data = reinterpret_cast<T*>(info.buffer);
    return data[0];
}

// Updates the RunTimeOperandInfo with the newly calculated shape.
// Allocate the buffer if we need to.
// static bool setInfoAndAllocateIfNeeded(RunTimeOperandInfo* info, const Shape& shape) {
//     // For user-provided model output operands, the parameters must match the Shape
//     // calculated from the preparation step.
//     if (info->lifetime == OperandLifeTime::MODEL_OUTPUT) {
//         if (info->type != shape.type || info->dimensions != shape.dimensions) {
//             LOG(ERROR) << "Invalid type or dimensions for model output";
//             return false;
//         }
//         if (info->type == OperandType::TENSOR_QUANT8_ASYMM &&
//             (info->scale != shape.scale || info->zeroPoint != shape.offset)) {
//             LOG(ERROR) << "Invalid scale or zeroPoint for model output";
//             return false;
//         }
//     }
//     info->type = shape.type;
//     info->dimensions = shape.dimensions;
//     info->scale = shape.scale;
//     info->zeroPoint = shape.offset;
//     if (info->lifetime == OperandLifeTime::TEMPORARY_VARIABLE && info->buffer == nullptr) {
//         uint32_t length = sizeOfData(info->type, info->dimensions);
//         info->buffer = new uint8_t[length];
//         if (info->buffer == nullptr) {
//             return false;
//         }
//     }
//     return true;
// }

bool BasePreparedModel::initialize() {
    VLOG(L1, "initialize");

    switch (mTargetDevice)
    {
        case IntelDeviceType::CPU:
            mPlugin = new DevicePlugin<IntelDeviceType::CPU>();
            break;
        
        default:
            ALOGE("Plugin not supported yet for this device type!!!");
            return false;
    }    
    
    if (!mModelInfo->initRuntimeInfo()) {
        ALOGE("Failed to initialize Model runtime parameters!!");
        return false;
    }

    mNgraphCreator = std::make_shared<NgraphNetworkCreator>(mModelInfo.get());
    if (!mNgraphCreator->init()) {
        ALOGE("Failed to initialize ngraph creator!!");
        return false;
    }

    auto vecOperations = mModelInfo->getOperations();
    for (auto op: vecOperations) {
        if (!NgraphOpsFactory::isOperationSupported(op, mModelInfo.get()))
            return false;
    }

    mNet = mNgraphCreator->generateIRGraph();
    mPlugin->prepareInput(InferenceEngine::Precision::FP32, Layout::NCHW);
    mPlugin->prepareOutput(InferenceEngine::Precision::FP32, Layout::NCHW);
    mPlugin->loadNetwork();

    return true;
}

Return<ErrorStatus> BasePreparedModel::executeBase(const V1_3::Request& request,
                                                    MeasureTiming measure,
                                                    const sp<V1_3::IExecutionCallback>& callback) {
    VLOG(L1, "executebase");

    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateRequest(request, mModelInfo->getModel())) {
        callback->notify_1_3(ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the driver service
    // is expected to live forever.
    std::thread([this, request, measure, driverStart, callback] {
        asyncExecute(request, measure, driverStart, callback);
    }).detach();

    return ErrorStatus::NONE;
}

void BasePreparedModel::asyncExecute(const V1_3::Request& request, V1_2::MeasureTiming measure,
                                 time_point driverStart,
                                 const sp<V1_3::IExecutionCallback>& callback) {
    time_point driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();

    if (!mModelInfo->setRunTimePoolInfosFromHidlMemories(request.pools)) {
        callback->notify_1_3(ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return;
    }

    hidl_vec<V1_2::OutputShape> outputShapes(request.outputs.size());
    
    auto inLayerMap = mNgraphCreator->getInputLayerMap();
    // Verify all the request inputs are captured during graph construction
    // Can be optimized.
    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto reqIn = mModelInfo->getModelInputIndex(i);
        auto iter2 = std::find_if(inLayerMap.begin(), inLayerMap.end(),
                            [&](const std::pair<uint32_t, LayerInfo>& elem) {
                                return (elem.first == reqIn);
                            });
        if (iter2 == inLayerMap.end()) {
            ALOGE("Did we miss mapping some of the inputs???");
        }
    }

    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto inIndex = mModelInfo->getModelInputIndex(i);
        auto srcBlob = mModelInfo->getBlobFromMemoryPoolIn(request, inIndex);

        // Get name of node from layermap
        // Ignore memory layers for sometime
        if (inLayerMap.find(inIndex) != inLayerMap.end()) {
            auto layerData = inLayerMap[inIndex];
            auto destBlob = mPlugin->getInferRequest().GetBlob(layerData.layerName);
            float* dest = destBlob->buffer().as<float*>();
            float* src = srcBlob->buffer().as<float*>();
            std::memcpy(dest, src, srcBlob->byteSize());
        } else {
            ALOGE("Failed to find layerindex in layername");
        }
    }

    mPlugin->infer();

    auto outLayerMap = mNgraphCreator->getOutputLayerMap();
    for (size_t i = 0; i < request.outputs.size(); i++) {
        auto outIndex = mModelInfo->getModelOutputIndex(i);
        void* destPtr = static_cast<float*>(mModelInfo->getBlobFromMemoryPoolOut(request, outIndex));

        if (outLayerMap.find(outIndex) != outLayerMap.end()) {
            auto layerData = outLayerMap[outIndex];
            auto srcBlob = mPlugin->getInferRequest().GetBlob(layerData.layerName);
            float* src = srcBlob->buffer().as<float*>();
            std::memcpy(destPtr, src, srcBlob->byteSize());
        } else {
            ALOGE("Failed to find layerindex in layername");
        }
    }

    mModelInfo->updateRequestPoolInfos();
    
    Return<void> returned;
    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        // VLOG(L1, "Driver::asyncExecute timing = %s", toString(timing));
        returned = callback->notify_1_3(ErrorStatus::NONE, {}, timing);
    } else {
        returned = callback->notify_1_3(ErrorStatus::NONE, {}, kNoTiming);
    }
    
    if (!returned.isOk()) {
        ALOGE("hidl callback failed to return properly: %s", returned.description().c_str());
    }
}

Return<void> BasePreparedModel::executeSynchronously(const V1_0::Request& request,
                                                        V1_2::MeasureTiming measure,
                                                        executeSynchronously_cb cb) {
    VLOG(L1, "BasePreparedModel::executeSynchronously");
    cb(V1_0::ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
    return Void();
}

Return<void> BasePreparedModel::executeSynchronously_1_3(const V1_3::Request &request,
                                          MeasureTiming measure,
                                          const V1_3::OptionalTimePoint& deadline,
                                          const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                                          V1_3::IPreparedModel::executeSynchronously_1_3_cb cb) {
    VLOG(L1, "Begin to executeSynchronously_1_3");
    time_point driverStart, driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();

    Return<void> returned;
    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        VLOG(L1, "Driver::executeSynchronously timing = %s", timing);
        cb(ErrorStatus::NONE, {}, timing);
    } else {
        cb(ErrorStatus::NONE, {}, kNoTiming);
    }
    return Void();
}

Return<ErrorStatus> BasePreparedModel::execute_1_3(const V1_3::Request& request,
                                          MeasureTiming measure,
                                          const V1_3::OptionalTimePoint&,
                                          const V1_3::OptionalTimeoutDuration&,
                                          const sp<V1_3::IExecutionCallback>& callback) {
    VLOG(L1, "Begin to execute");
    return executeBase(request, MeasureTiming::NO, callback);
}

Return<void> BasePreparedModel::configureExecutionBurst(
    const sp<V1_2::IBurstCallback>& callback,
    const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
    const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel, configureExecutionBurst_cb cb) {
    VLOG(L1, "Driver::configureExecutionBurst");

    cb(V1_0::ErrorStatus::GENERAL_FAILURE, {});
    return Void();
}

Return<V1_0::ErrorStatus> BasePreparedModel::execute(const V1_0::Request& request,
                                           const sp<V1_0::IExecutionCallback>& callback) {
    callback->notify(V1_0::ErrorStatus::GENERAL_FAILURE);
    return V1_0::ErrorStatus::GENERAL_FAILURE;
}

Return<V1_0::ErrorStatus> BasePreparedModel::execute_1_2(const V1_0::Request& request, MeasureTiming measure,
                                               const sp<V1_2::IExecutionCallback>& callback) {
    VLOG(L1, "Begin to execute_1_2");
    callback->notify_1_2(V1_0::ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
    return V1_0::ErrorStatus::GENERAL_FAILURE;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
