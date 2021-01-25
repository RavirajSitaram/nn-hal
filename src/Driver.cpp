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

#define LOG_TAG "Driver"

#include "Driver.h"

#include <android-base/logging.h>
#include <thread>
#include "BasePreparedModel.h"
#include "ValidateHal.h"
#include "NgraphOpsFactory.h"
#include "ModelManager.h"

// sample gRPC client connection headers
#ifdef __ANDROID__
#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <string>
#include "GreeterClient.h"
#include "helloworld.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using helloworld::Greeter;
using helloworld::HelloReply;
using helloworld::HelloRequest;
#endif

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using namespace android::nn;
using Capabilities = V1_3::Capabilities;

static sp<BasePreparedModel> ModelFactory(IntelDeviceType device) {
    sp<BasePreparedModel> driverPreparedModel = nullptr;
    driverPreparedModel = new BasePreparedModel(device);
    return driverPreparedModel;
}

static sp<BasePreparedModel> ModelFactory(IntelDeviceType device, const V1_3::Model& model) {
    sp<BasePreparedModel> driverPreparedModel = nullptr;
    driverPreparedModel = new BasePreparedModel(device, model);
    return driverPreparedModel;
}

// For HAL-1.2 version
hidl_vec<Capabilities::OperandPerformance> nonExtensionOperandPerformance(V1_0::PerformanceInfo perf) {
    using OpPerf = Capabilities::OperandPerformance;

    // Note: range presents enumerators in declaration order, not in numerical order.
    static constexpr ::android::hardware::hidl_enum_range<OperandType> kOperandTypeRange;

    hidl_vec<OpPerf> ret(kOperandTypeRange.end() - kOperandTypeRange.begin());

    std::transform(kOperandTypeRange.begin(), kOperandTypeRange.end(), ret.begin(),
                   [perf](OperandType type) {
                       return V1_3::Capabilities::OperandPerformance{type, perf};
                   });
    std::sort(ret.begin(), ret.end(),
              [](const OpPerf& a, const OpPerf& b) { return a.type < b.type; });

    return ret;
}

Return<void> Driver::getCapabilities_1_3(getCapabilities_1_3_cb cb) {
    ALOGI("Entering %s", __func__);
    switch(mDeviceType) {
        case IntelDeviceType::CPU:
            {
                ALOGI("CPU driver getCapabilities()");
                // Setting operandPerformance value to base value for all operand types
                Capabilities capabilities = {
                    .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.9f, .powerUsage = 0.9f},
                    .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.9f, .powerUsage = 0.9f},
                    .operandPerformance = nonExtensionOperandPerformance({0.9f, 0.9f})};

                ALOGI("CPU MKLDNN driver Capabilities .execTime = 0.9f, .powerUsage = 0.9f");
                cb(ErrorStatus::NONE, capabilities);
                break;
            }
        case IntelDeviceType::GNA:
            {
                ALOGI("GPU driver getCapabilities()");
                Capabilities capabilities = {
                    .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 0.95f, .powerUsage = 0.85f},
                    .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 0.95f, .powerUsage = 0.85f},
                    .operandPerformance = nonExtensionOperandPerformance({0.95f, 0.95f})};

                ALOGI("GPU clDNN driver Capabilities .execTime = 0.95f, .powerUsage = 0.85f");
                cb(ErrorStatus::NONE, capabilities);
                break;
            }
        case IntelDeviceType::VPU:
            {
                ALOGI("Myriad driver getCapabilities()");
                Capabilities capabilities = {
                    .relaxedFloat32toFloat16PerformanceScalar = {.execTime = 1.1f, .powerUsage = 1.1f},
                    .relaxedFloat32toFloat16PerformanceTensor = {.execTime = 1.1f, .powerUsage = 1.1f},
                    .operandPerformance = nonExtensionOperandPerformance({1.1f, 1.1f})};

                ALOGI("Myriad driver Capabilities .execTime = 1.1f, .powerUsage = 1.1f");
                cb(ErrorStatus::NONE, capabilities);
                break;
            }
        default:
            Capabilities capabilities;
            cb(ErrorStatus::DEVICE_UNAVAILABLE, capabilities);
            break;
    }

    return Void();
}

Return<void> Driver::getSupportedOperations_1_3(const Model& model,
                                                getSupportedOperations_1_3_cb cb) {
    ALOGI("Entering %s", __func__);
    std::unique_ptr<NnapiModelInfo> modelInfo = std::make_unique<NnapiModelInfo>(model);

    int count = model.main.operations.size();
    std::vector<bool> resultVec(count, true);

    if (!validateModel(model)) {
        ALOGI("NNERR: %s failed at line no: %d\n", __func__, __LINE__);
        cb(ErrorStatus::INVALID_ARGUMENT, resultVec);
        return Void();
    }

    for (int i = 0; i < count; i++) {
        const auto& operation = model.main.operations[i];
        resultVec[i] = NgraphOpsFactory::isOperationSupported(operation, modelInfo.get());
    }
    cb(ErrorStatus::NONE, resultVec);
    return Void();
}

Return<V1_3::ErrorStatus> Driver::prepareModel_1_3(const V1_3::Model& model,
                                               V1_1::ExecutionPreference preference,
                                               V1_3::Priority priority,
                                               const V1_3::OptionalTimePoint&,
                                               const android::hardware::hidl_vec<android::hardware::hidl_handle>&,
                                               const android::hardware::hidl_vec<android::hardware::hidl_handle>&,
                                               const HidlToken&,
                                               const android::sp<V1_3::IPreparedModelCallback>& cb) {
    if (cb.get() == nullptr) {
        ALOGI("invalid callback passed to prepareModel");
        return V1_3::ErrorStatus::INVALID_ARGUMENT;
    }

    if (!validateModel(model) || !validateExecutionPreference(preference)) {
        cb->notify(V1_0::ErrorStatus::INVALID_ARGUMENT, nullptr);
        return V1_3::ErrorStatus::INVALID_ARGUMENT;
    }

    sp<BasePreparedModel> driverPreparedModel = new BasePreparedModel(mDeviceType, model);
    if (!driverPreparedModel->initialize()) {
        ALOGI("Failed to initialize prepared model");
        cb->notify_1_3(V1_3::ErrorStatus::INVALID_ARGUMENT, nullptr);
    }

    cb->notify_1_3(V1_3::ErrorStatus::NONE, driverPreparedModel);
    return V1_3::ErrorStatus::NONE;
}

Return<ErrorStatus> Driver::prepareModelFromCache_1_3(
        const V1_3::OptionalTimePoint&,
        const android::hardware::hidl_vec<android::hardware::hidl_handle>&,
        const android::hardware::hidl_vec<android::hardware::hidl_handle>&,
        const HidlToken&,
        const sp<V1_3::IPreparedModelCallback>& callback)
{
    ALOGV("V1_3::Driver::prepareModelFromCache_1_3()");

#ifdef __ANDROID__
    // sample gRPC client connection code
    GreeterClient greeter(
        grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));
    auto reply = greeter.SayHello("world");
    ALOGI("***********GRPC server response************* %s", reply.c_str());
#endif

    if (callback.get() == nullptr) {
        ALOGI("invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    
    // if (!validateModel(model) || !validateExecutionPreference(preference)) {
    //     callback->notify(V1_0::ErrorStatus::INVALID_ARGUMENT, nullptr);
    //     return ErrorStatus::INVALID_ARGUMENT;
    // }

    // TODO: make asynchronous later
    sp<BasePreparedModel> driverPreparedModel = ModelFactory(mDeviceType);
    // if (driverPreparedModel == NULL) {
    //     ALOGI("failed to create preparedmodel");
    //     return ErrorStatus::INVALID_ARGUMENT;
    // }

    // if (!driverPreparedModel->initialize(model)) {
    //     ALOGI("failed to initialize preparedmodel");
    //     callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
    //     return ErrorStatus::NONE;
    // }

    callback->notify(V1_0::ErrorStatus::NONE, driverPreparedModel);
    return ErrorStatus::NONE;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android