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
#include <algorithm>

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

void notify(const sp<V1_0::IPreparedModelCallback>& callback, const ErrorStatus& status,
            const sp<BasePreparedModel>& preparedModel) {
    const auto ret = callback->notify(convertToV1_0(status), preparedModel);
    if (!ret.isOk()) {
        LOG(ERROR) << "Error when calling IPreparedModelCallback::notify: " << ret.description();
    }
}

void notify(const sp<V1_2::IPreparedModelCallback>& callback, const ErrorStatus& status,
            const sp<BasePreparedModel>& preparedModel) {
    const auto ret = callback->notify_1_2(convertToV1_0(status), preparedModel);
    if (!ret.isOk()) {
        LOG(ERROR) << "Error when calling IPreparedModelCallback::notify_1_2: "
                   << ret.description();
    }
}

void notify(const sp<V1_3::IPreparedModelCallback>& callback, const ErrorStatus& status,
            const sp<BasePreparedModel>& preparedModel) {
    const auto ret = callback->notify_1_3(status, preparedModel);
    if (!ret.isOk()) {
        LOG(ERROR) << "Error when calling IPreparedModelCallback::notify_1_3: "
                   << ret.description();
    }
}

void notify(const sp<V1_0::IExecutionCallback>& callback, const ErrorStatus& status,
            const hidl_vec<V1_2::OutputShape>&, Timing) {
    const auto ret = callback->notify(convertToV1_0(status));
    if (!ret.isOk()) {
        LOG(ERROR) << "Error when calling IExecutionCallback::notify: " << ret.description();
    }
}

void notify(const sp<V1_2::IExecutionCallback>& callback, const ErrorStatus& status,
            const hidl_vec<V1_2::OutputShape>& outputShapes, Timing timing) {
    const auto ret = callback->notify_1_2(convertToV1_0(status), outputShapes, timing);
    if (!ret.isOk()) {
        LOG(ERROR) << "Error when calling IExecutionCallback::notify_1_2: " << ret.description();
    }
}

void notify(const sp<V1_3::IExecutionCallback>& callback, const ErrorStatus& status,
            const hidl_vec<V1_2::OutputShape>& outputShapes, Timing timing) {
    const auto ret = callback->notify_1_3(status, outputShapes, timing);
    if (!ret.isOk()) {
        LOG(ERROR) << "Error when calling IExecutionCallback::notify_1_3" << ret.description();
    }
}

static
bool quantizeToQuant8Signed(const float* inputData, int8_t* outputData, const Shape& outputShape) {
    uint32_t size = getNumberOfElements(outputShape.dimensions);
    for (uint32_t i = 0; i < size; ++i) {
        outputData[i] = static_cast<int8_t>(std::max<float>(
                -128.0f,
                std::min<float>(127.0f, outputShape.offset +
                                std::round(inputData[i] / outputShape.scale))));
        ALOGD("%s input: %f output: %d", __func__, inputData[i], outputData[i]);
    }
    return true;
}

static
bool quantizeToQuant8(const float* inputData, uint8_t* outputData, const Shape& outputShape) {
      uint32_t size = getNumberOfElements(outputShape.dimensions);
      for (uint32_t i = 0; i < size; ++i) {
          outputData[i] = static_cast<uint8_t>(std::max<float>(
                  0.0f, std::min<float>(255.0f, outputShape.offset + std::round(inputData[i] /
                                                                                outputShape.scale))));
        ALOGD("%s input: %f output: %d scale: %d offset:%d", __func__, inputData[i], outputData[i],
                                                                        outputShape.scale, outputShape.offset);
      }
      return true;
}

static
bool quantizeToQuant16(const float* inputData, uint16_t* outputData, const Shape& outputShape) {
    uint32_t size = getNumberOfElements(outputShape.dimensions);
    for (uint32_t i = 0; i < size; ++i) {
        outputData[i] = static_cast<uint16_t>(outputShape.offset + (std::round(inputData[i] / outputShape.scale)));
        ALOGD("%s input: %f output: %d", __func__, inputData[i], outputData[i]);
    }      
    return true;
}

// Move out all the code from ModelInfo regarding runtime pool info
// We need to make this function re-entrant
template <typename T_IExecutionCallback>
void asyncExecute(const V1_3::Request& request, V1_2::MeasureTiming measure,
                  BasePreparedModel* preparedModel, time_point driverStart,
                  const V1_3::OptionalTimePoint& halDeadline,
                  const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                  const sp<T_IExecutionCallback>& callback) {
    ALOGD("%s", __func__);

    auto modelInfo = preparedModel->getModelInfo();
    auto plugin = preparedModel->getPlugin();
    auto ngraphNw = preparedModel->getNgraphNwCreator();

    time_point driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();

#if 1
    if (!modelInfo->setRunTimePoolInfosFromHidlMemories(request.pools)) {
        notify(callback, ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return;
    }
#else
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!setRunTimePoolInfosFromHidlMemories(&requestPoolInfos, request.pools)) {
        cb->notify(V1_0_ErrorStatus::GENERAL_FAILURE);
        return;
    }
#endif

    // if (halDeadline.has_value()) {
    //     ALOGE("HAL Deadline has value !!!");
    // }
    
    auto inLayerMap = ngraphNw->getInputLayerMap();
    // Verify all the request inputs are captured during graph construction
    // Can be optimized.
    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto reqIn = modelInfo->getModelInputIndex(i);
        ALOGD("Searching for input index : %d in layermap", reqIn);
        auto iter2 = std::find_if(inLayerMap.begin(), inLayerMap.end(),
                            [&](const std::pair<uint32_t, LayerInfo>& elem) {
                                ALOGD("Index: %d", elem.first);
                                return (elem.first == reqIn);
                            });
        if (iter2 == inLayerMap.end()) {
            ALOGE("Did we miss mapping some of the inputs???");
        }
    }

    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto inIndex = modelInfo->getModelInputIndex(i);
        auto srcBlob = modelInfo->getBlobFromMemoryPoolIn(request, i);

        // Get name of node from layermap
        // Ignore memory layers for sometime
        if (inLayerMap.find(inIndex) != inLayerMap.end()) {
            auto layerData = inLayerMap[inIndex];
            ALOGD("Found input index: %d layername : %s", inIndex, layerData.layerName.c_str());
            auto destBlob = plugin->getInferRequest().GetBlob(layerData.layerName);
            uint8_t* dest = destBlob->buffer().as<uint8_t*>();
            uint8_t* src = srcBlob->buffer().as<uint8_t*>();
            std::memcpy(dest, src, srcBlob->byteSize());
            writeBufferToFile(layerData.layerName, srcBlob->buffer().as<float*>(), srcBlob->size());
        } else {
            ALOGE("Failed to find layerindex in input layers");
        }
    }

    plugin->infer();

    auto outLayerMap = ngraphNw->getOutputLayerMap();
    for (size_t i = 0; i < request.outputs.size(); i++) {
        auto outIndex = modelInfo->getModelOutputIndex(i);
        ALOGI("OutputIndex: %d", outIndex);
        void* destPtr = modelInfo->getBlobFromMemoryPoolOut(request, i);

        if (outLayerMap.find(outIndex) != outLayerMap.end()) {
            auto layerData = outLayerMap[outIndex];
            ALOGD("Found output index: %d layername : %s", outIndex, layerData.layerName.c_str());
            auto srcBlob = plugin->getInferRequest().GetBlob(layerData.layerName);
            auto operandType = modelInfo->getOperandType(outIndex);
            auto operand = modelInfo->getRuntimeOperand(outIndex);
            ALOGD("operandtype: %d", operandType);

            switch(static_cast<int>(operandType)) {
                case static_cast<int>(OperandType::TENSOR_QUANT8_ASYMM):
                    ALOGD("FLOAT32~~~~~~~~~~~~~~~~~~~~~~~~~~ TENSOR_QUANT8_SYMM");
                    quantizeToQuant8(srcBlob->buffer().as<float*>(), (uint8_t*)destPtr, operand.shape());
                    break;
                case static_cast<int>(OperandType::TENSOR_FLOAT32):
                    ALOGD("FLOAT32~~~~~~~~~~~~~~~~~~~~~~~~~~ bytesize=%d", srcBlob->byteSize());
                    std::memcpy((uint8_t*)destPtr, srcBlob->buffer().as<uint8_t*>(), srcBlob->byteSize());
                    break;
                default:
                    ALOGD("FLOAT32~~~~~~~~~~~~~~~~~~~~~~~~~~ default");
                    nnAssert(true);
                    break;
            }
            writeBufferToFile(layerData.layerName, srcBlob->buffer().as<float*>(), srcBlob->size());

            float* a = static_cast<float*>(destPtr);
            ALOGD("########### -- %f", *a);
        } else {
            ALOGE("Failed to find layerindex in output layers");
            notify(callback, ErrorStatus::GENERAL_FAILURE,  {}, kNoTiming);
        }
    }

    if (!modelInfo->updateRequestPoolInfos()) {
        ALOGE("Failed to update the request pool infos");
    }

    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        // VLOG(L1, "Driver::asyncExecute timing = %s", toString(timing));
        notify(callback, ErrorStatus::NONE, modelInfo->getOutputShapes(), timing);
    } else {
        notify(callback, ErrorStatus::NONE,  modelInfo->getOutputShapes(), kNoTiming);
    }
}

void BasePreparedModel::deinitialize() {
    VLOG(L1, "deinitialize");

    mModelInfo->unmapRuntimeMemPools();

    if (mNet)
        delete mNet;

    if (mPlugin)
        delete mPlugin;
}

template <typename T>
T getScalarData(const RunTimeOperandInfo& info) {
    // TODO: Check buffer is at least as long as size of data.
    T* data = reinterpret_cast<T*>(info.buffer);
    return data[0];
}

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
        if (!NgraphOpsFactory::isOperationSupported(op, mModelInfo.get())) {
            ALOGE("Returning false from init() function");
            return false;
        }
    }

    ALOGI("Generating IR Graph");
    mNet = mNgraphCreator->generateIRGraph();
    if (!mNet) {
        ALOGE("Failed to convert to CNNNetwork");
        nnAssert(true);
    }

    //mNet->serialize("/tmp/graph.xml", "/tmp.graph.bin");
    
    switch (mTargetDevice)
    {
        case IntelDeviceType::CPU:
            mPlugin = new DevicePlugin<IntelDeviceType::CPU>(mNet);
            mPlugin->loadNetwork();
            ALOGE("Done calling loadNetwork");
            break;
        
        default:
            ALOGE("Plugin not supported yet for this device type!!!");
            return false;
    }

    ALOGD("Returning from preparedModel init");
    return true;
}

template <typename T_IExecutionCallback>
Return<ErrorStatus> executeBase(const V1_3::Request& request,
                                MeasureTiming measure,
                                BasePreparedModel* preparedModel,
                                const V1_3::OptionalTimePoint& halDeadline,
                                const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                                const sp<T_IExecutionCallback>& callback) {
    VLOG(L1, "executebase");

    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateRequest(request, preparedModel->getModelInfo()->getModel())) {
        notify(callback, ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the driver service
    // is expected to live forever.
    std::thread([preparedModel, request, measure, driverStart, callback, halDeadline,
                    loopTimeoutDuration] {
        asyncExecute(request, measure, preparedModel,
                     driverStart, halDeadline, loopTimeoutDuration, callback);
    }).detach();

    return ErrorStatus::NONE;
}

#if 0
bool setRunTimePoolInfosFromHidlMemories(std::vector<RunTimePoolInfo>* poolInfos,
                                         const hidl_vec<hidl_memory>& pools) {
    poolInfos->resize(pools.size());
    for (size_t i = 0; i < pools.size(); i++) {
        auto& poolInfo = (*poolInfos)[i];
        if (!poolInfo.set(pools[i].hidlMemory())) {
            LOG(ERROR) << "Could not map pool";
            return false;
        }
    }
    return true;
}
#endif

static std::tuple<ErrorStatus, hidl_vec<V1_2::OutputShape>, Timing>
executeSynchronouslyBase(const V1_3::Request& request, V1_2::MeasureTiming measure,
                  BasePreparedModel* preparedModel,
                  const V1_3::OptionalTimePoint& halDeadline,
                  const V1_3::OptionalTimeoutDuration& loopTimeoutDuration) {
    ALOGD("%s", __func__);

    auto modelInfo = preparedModel->getModelInfo();
    auto plugin = preparedModel->getPlugin();
    auto ngraphNw = preparedModel->getNgraphNwCreator();

    time_point driverStart, driverEnd, deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) driverStart = now();

#if 1
    if (!modelInfo->setRunTimePoolInfosFromHidlMemories(request.pools)) {
        ALOGE("Failed to set runtime pool info from HIDL memories");
        return {ErrorStatus::GENERAL_FAILURE, {}, kNoTiming};
    }
#else
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!setRunTimePoolInfosFromHidlMemories(&requestPoolInfos, request.pools)) {
        cb->notify(V1_0_ErrorStatus::GENERAL_FAILURE);
        return;
    }
#endif

    // if (halDeadline.has_value()) {
    //     ALOGE("HAL Deadline has value !!!");
    // }
    
    auto inLayerMap = ngraphNw->getInputLayerMap();
    // Verify all the request inputs are captured during graph construction
    // Can be optimized.
    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto reqIn = modelInfo->getModelInputIndex(i);
        ALOGD("Searching for input index : %d in layermap", reqIn);
        auto iter2 = std::find_if(inLayerMap.begin(), inLayerMap.end(),
                            [&](const std::pair<uint32_t, LayerInfo>& elem) {
                                ALOGD("Index: %d", elem.first);
                                return (elem.first == reqIn);
                            });
        if (iter2 == inLayerMap.end()) {
            ALOGE("Did we miss mapping some of the inputs???");
        }
    }

    for (size_t i = 0; i < request.inputs.size(); i++) {
        auto inIndex = modelInfo->getModelInputIndex(i);
        auto srcBlob = modelInfo->getBlobFromMemoryPoolIn(request, i);

        // Get name of node from layermap
        // Ignore memory layers for sometime
        if (inLayerMap.find(inIndex) != inLayerMap.end()) {
            auto layerData = inLayerMap[inIndex];
            ALOGD("Found input index: %d layername : %s", inIndex, layerData.layerName.c_str());
            auto destBlob = plugin->getInferRequest().GetBlob(layerData.layerName);
            uint8_t* dest = destBlob->buffer().as<uint8_t*>();
            uint8_t* src = srcBlob->buffer().as<uint8_t*>();
            std::memcpy(dest, src, srcBlob->byteSize());
            writeBufferToFile(layerData.layerName, srcBlob->buffer().as<float*>(), srcBlob->size());
        } else {
            ALOGE("Failed to find layerindex in input layers");
        }
    }

    if (measure == MeasureTiming::YES) deviceStart = now();
    plugin->infer();
    if (measure == MeasureTiming::YES) deviceEnd = now();

    auto outLayerMap = ngraphNw->getOutputLayerMap();
    for (size_t i = 0; i < request.outputs.size(); i++) {
        auto outIndex = modelInfo->getModelOutputIndex(i);
        ALOGI("OutputIndex: %d", outIndex);
        void* destPtr = modelInfo->getBlobFromMemoryPoolOut(request, i);

        if (outLayerMap.find(outIndex) != outLayerMap.end()) {
            auto layerData = outLayerMap[outIndex];
            ALOGD("Found output index: %d layername : %s", outIndex, layerData.layerName.c_str());
            auto srcBlob = plugin->getInferRequest().GetBlob(layerData.layerName);
            auto operandType = modelInfo->getOperandType(outIndex);
            auto operand = modelInfo->getRuntimeOperand(outIndex);
            ALOGD("operandtype: %d", operandType);

            switch(static_cast<int>(operandType)) {
                case static_cast<int>(OperandType::TENSOR_QUANT8_ASYMM):
                    ALOGD("O/P Quantizing FLOAT to TENSOR_QUANT8_SYMM");
                    quantizeToQuant8(srcBlob->buffer().as<float*>(), (uint8_t*)destPtr, operand.shape());
                    break;
                case static_cast<int>(OperandType::TENSOR_FLOAT32):
                    ALOGD("O/P copied to destination");
                    std::memcpy((uint8_t*)destPtr, srcBlob->buffer().as<uint8_t*>(), srcBlob->byteSize());
                    break;
                default:
                    ALOGE("O/P no known conversion to float");
                    nnAssert(true);
                    break;
            }
            writeBufferToFile(layerData.layerName, srcBlob->buffer().as<float*>(), srcBlob->size());

            float* a = static_cast<float*>(destPtr);
            ALOGD("########### -- %f", *a);
        } else {
            ALOGE("Failed to find layerindex in output layers");
            return {ErrorStatus::GENERAL_FAILURE, {}, kNoTiming};
        }
    }
    if (!modelInfo->updateRequestPoolInfos()) {
        ALOGE("Failed to update the request pool infos");
        return {ErrorStatus::GENERAL_FAILURE, {}, kNoTiming};
    }
    
    //Return<void> returned;
    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        return {ErrorStatus::NONE, modelInfo->getOutputShapes(), timing};
        // VLOG(L1, "Driver::asyncExecute timing = %s", toString(timing));
        //returned = callback->notify_1_3(ErrorStatus::NONE, modelInfo->getOutputShapes(), timing);
    }
    
    //returned = callback->notify_1_3(ErrorStatus::NONE,  modelInfo->getOutputShapes(), kNoTiming);
    return {ErrorStatus::NONE, modelInfo->getOutputShapes(), kNoTiming};
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
    auto [status, outputShapes, timing] = executeSynchronouslyBase(request, measure, this, deadline, loopTimeoutDuration);
    cb(status, std::move(outputShapes), timing);
    return Void();
}

Return<ErrorStatus> BasePreparedModel::execute_1_3(const V1_3::Request& request,
                                          MeasureTiming measure,
                                          const V1_3::OptionalTimePoint& deadline,
                                          const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                                          const sp<V1_3::IExecutionCallback>& callback) {
    VLOG(L1, "Begin to execute_1_3");
    return executeBase(request, measure, this, deadline, loopTimeoutDuration, callback);
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
    const ErrorStatus status = executeBase(android::nn::convertToV1_3(request), MeasureTiming::NO, this, {}, {}, callback);
    return android::nn::convertToV1_0(status);
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
