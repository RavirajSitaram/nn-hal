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

#ifndef ANDROID_ML_NN_DRIVER_H
#define ANDROID_ML_NN_DRIVER_H
#include <android/hardware/neuralnetworks/1.0/IDevice.h>
#include <android/hardware/neuralnetworks/1.0/IExecutionCallback.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModelCallback.h>
#include <android/hardware/neuralnetworks/1.0/types.h>
#include <android/hardware/neuralnetworks/1.1/IDevice.h>
#include <android/hardware/neuralnetworks/1.1/types.h>
#include <android/hardware/neuralnetworks/1.2/IDevice.h>
#include <android/hardware/neuralnetworks/1.2/IExecutionCallback.h>
#include <android/hardware/neuralnetworks/1.2/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.2/IPreparedModelCallback.h>
#include <android/hardware/neuralnetworks/1.2/types.h>
#include <android/hardware/neuralnetworks/1.3/IDevice.h>
#include <android/hardware/neuralnetworks/1.3/types.h>
#include <android/hardware/neuralnetworks/1.3/IPreparedModel.h>

#include <string>
#include <android/hidl/memory/1.0/IMemory.h>
#include "cutils/log.h"

namespace V1_3 = ::android::hardware::neuralnetworks::V1_3;
namespace V1_2 = ::android::hardware::neuralnetworks::V1_2;
namespace V1_1 = ::android::hardware::neuralnetworks::V1_1;
namespace V1_0 = ::android::hardware::neuralnetworks::V1_0;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

enum class IntelDeviceType {
    CPU,
    GNA,
    VPU,
    OTHER
};

using Model                     = V1_3::Model;
using Operand                   = V1_3::Operand;
using OperandLifeTime           = V1_3::OperandLifeTime;
using OperandType               = V1_3::OperandType;
using Operation                 = V1_3::Operation;
using OperationType             = V1_3::OperationType;
using ExecutionCallback         = V1_3::IExecutionCallback;
using getSupportedOperations_cb = V1_3::IDevice::getSupportedOperations_1_3_cb;
using ErrorStatus               = V1_3::ErrorStatus;
using DeviceStatus              = V1_0::DeviceStatus;
using IMemory                   = ::android::hidl::memory::V1_0::IMemory;
using MeasureTiming             = V1_2::MeasureTiming;
using Timing                    = V1_2::Timing;

// using namespace V1_3 = ::android::hardware::neuralnetworks::V1_3;

using ::android::hardware::MQDescriptorSync;
using HidlToken = android::hardware::hidl_array<uint8_t, 32>;

// Base class used to create vpu drivers for the NN HAL.  This class
// provides some implementation of the more common functions.
//
// Since these drivers simulate hardware, they must run the computations
// on the CPU.  An actual driver would not do that.
class Driver : public V1_3::IDevice {
public:
    Driver() {}
    Driver(IntelDeviceType name) : mDeviceType(name) {}

    ~Driver() override {}

    // For HAL-1.0 version
    Return<void> getCapabilities(getCapabilities_cb cb) {
        ALOGI("Entering %s", __func__);

        return Void();
    }

    Return<void> getSupportedOperations(const V1_0::Model& model, getSupportedOperations_cb cb) {
        ALOGI("Entering %s", __func__);

        return Void();
    }

    Return<V1_0::ErrorStatus> prepareModel(const V1_0::Model& model,
                                            const sp<V1_0::IPreparedModelCallback>& callback);

    // For HAL-1.1 version
    Return<void> getCapabilities_1_1(getCapabilities_1_1_cb cb) {
        ALOGI("Entering %s", __func__);

        return Void();
    }

    Return<void> getSupportedOperations_1_1(const V1_1::Model& model,
                                                    getSupportedOperations_1_1_cb cb) {
        ALOGI("Entering %s", __func__);

        return Void();
    }

    Return<V1_0::ErrorStatus> prepareModel_1_1(const V1_1::Model& model,
                                                V1_1::ExecutionPreference preference,
                                                const sp<V1_0::IPreparedModelCallback>& callback) {
        ALOGI("Entering %s", __func__);

        return V1_0::ErrorStatus::NONE;
    }

    // For HAL-1.2 version
    Return<void> getCapabilities_1_2(getCapabilities_1_2_cb cb) {
        ALOGI("Entering %s", __func__);

        return Void();
    }

    Return<void> getSupportedOperations_1_2(const V1_2::Model& model,
                                            V1_2::IDevice::getSupportedOperations_1_2_cb cb) {
        ALOGI("Entering %s", __func__);

        return Void();
    }
    
    Return<V1_0::ErrorStatus> prepareModel_1_2(const V1_2::Model& model, V1_1::ExecutionPreference preference,
                                         const hidl_vec<hidl_handle>& modelCache,
                                         const hidl_vec<hidl_handle>& dataCache,
                                         const HidlToken& token,
                                         const sp<V1_2::IPreparedModelCallback>& callback) {
        ALOGI("Entering %s", __func__);

        return V1_0::ErrorStatus::NONE;
    }

    // For HAL-1.3 version
    Return<void> getCapabilities_1_3(getCapabilities_1_3_cb cb) override;
    Return<void> getSupportedOperations_1_3(const V1_3::Model& model,
                                            getSupportedOperations_1_3_cb cb);
    Return<V1_3::ErrorStatus> prepareModel_1_3(const V1_3::Model& model,
                                               V1_1::ExecutionPreference preference,
                                               V1_3::Priority priority,
                                               const V1_3::OptionalTimePoint&,
                                               const android::hardware::hidl_vec<android::hardware::hidl_handle>&,
                                               const android::hardware::hidl_vec<android::hardware::hidl_handle>&,
                                               const HidlToken&,
                                               const android::sp<V1_3::IPreparedModelCallback>& cb);

    Return<V1_0::ErrorStatus> prepareModelFromCache(
        const hidl_vec<hidl_handle>&, const hidl_vec<hidl_handle>&, const HidlToken&,
        const sp<V1_2::IPreparedModelCallback>& callback) {
        ALOGI("Entering %s", __func__);
        callback->notify_1_2(V1_0::ErrorStatus::GENERAL_FAILURE, nullptr);
        return V1_0::ErrorStatus::GENERAL_FAILURE;
    }

    Return<ErrorStatus> prepareModelFromCache_1_3(
        const V1_3::OptionalTimePoint&,
        const android::hardware::hidl_vec<android::hardware::hidl_handle>&,
        const android::hardware::hidl_vec<android::hardware::hidl_handle>&,
        const HidlToken&,
        const sp<V1_3::IPreparedModelCallback>& callback);

    Return<DeviceStatus> getStatus() {
        ALOGI("DeviceStatus::AVAILABLE");
        return DeviceStatus::AVAILABLE;
    }

    Return<void> getVersionString(getVersionString_cb cb) {
        ALOGI("Entering %s", __func__);
        cb(V1_0::ErrorStatus::NONE, "intel_nn_hal");
        return Void();
    }

    Return<void> getType(getType_cb cb) {
        ALOGI("Entering %s", __func__);
        cb(V1_0::ErrorStatus::NONE, V1_2::DeviceType::CPU);
        return Void();
    }

    Return<void> getSupportedExtensions(getSupportedExtensions_cb cb) {
        ALOGI("Entering %s", __func__);
        cb(V1_0::ErrorStatus::NONE, {/* No extensions. */});
        return Void();
    }

    Return<void> getNumberOfCacheFilesNeeded(getNumberOfCacheFilesNeeded_cb cb) {
        ALOGI("Entering %s", __func__);
        cb(V1_0::ErrorStatus::NONE, 0, 0);
        return Void();
    }

    Return<void> allocate(const V1_3::BufferDesc& /*desc*/,
                          const hidl_vec<sp<V1_3::IPreparedModel>>& /*preparedModels*/,
                          const hidl_vec<V1_3::BufferRole>& /*inputRoles*/,
                          const hidl_vec<V1_3::BufferRole>& /*outputRoles*/,
                          allocate_cb cb) {
        ALOGV("hal_1_3::ArmnnDriver::allocate()");
        cb(ErrorStatus::GENERAL_FAILURE, nullptr, 0);
        return Void();
    }

protected:
    IntelDeviceType mDeviceType;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_VPU_DRIVER_H
