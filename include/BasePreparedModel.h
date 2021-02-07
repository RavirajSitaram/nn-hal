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

#ifndef ANDROID_ML_NN_BASEPREPAREDMODEL_H
#define ANDROID_ML_NN_BASEPREPAREDMODEL_H

#include <android/hardware/neuralnetworks/1.2/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.2/types.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
#include <sys/mman.h>
#include <fstream>
#include <string>
#include <frameworks/ml/nn/runtime/include/NeuralNetworks.h>
#include <frameworks/ml/nn/common/include/Utils.h>

#include "Driver.h"
#include "Utils.h"
#include "ModelManager.h"
#include "NgraphNwManager.h"
#include "DevicePlugin.h"
#include <ie_cnn_network.h>


// #define PARAM_I32(i) ParseOperationInput<int32_t>(mModel, operation, i)
// #define PARAM_FP(i) ParseOperationInput<float>(mModel, operation, i)

#define PARAM_I32(mModel, operation, i) ParseOperationInput<int32_t>(mModel, operation, i)
#define PARAM_FP(i) ParseOperationInput<float>(mModel, operation, i)

using ::android::hardware::MQDescriptorSync;
using ::android::hidl::memory::V1_0::IMemory;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace {

using time_point = std::chrono::steady_clock::time_point;

auto now() { return std::chrono::steady_clock::now(); };

auto microsecondsDuration(decltype(now()) end, decltype(now()) start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
};

}

template <class T>
using vec = std::vector<T>;
typedef uint8_t* memory;

using executeFenced_cb = std::function<void(::android::hardware::neuralnetworks::V1_3::ErrorStatus status,
    const ::android::hardware::hidl_handle& syncFence,
    const ::android::sp<::android::hardware::neuralnetworks::V1_3::IFencedExecutionCallback>& callback)>;

class BasePreparedModel : public V1_3::IPreparedModel{
    public:
        BasePreparedModel(IntelDeviceType device)
        : mTargetDevice(device),
          mNet(nullptr) {
#ifdef __ANDROID__
            mUseNgraph =
                isNgraphPropSet();  // TODO:Should additionally check if all the ops are supported
#endif
            mModelInfo = nullptr;
        }

        BasePreparedModel(IntelDeviceType device, const Model& model)
        : mTargetDevice(device),
          mNet(nullptr) {
#ifdef __ANDROID__
            mUseNgraph = isNgraphPropSet();
#endif
            mModelInfo = std::make_shared<NnapiModelInfo>(model);
        }

        ~BasePreparedModel() { deinitialize(); }

        Return<V1_0::ErrorStatus> execute(const V1_0::Request& request,
                                const sp<V1_0::IExecutionCallback>& callback) override;
        Return<V1_0::ErrorStatus> execute_1_2(const V1_0::Request& request, V1_2::MeasureTiming measure,
                                        const sp<V1_2::IExecutionCallback>& callback) override;
        Return<ErrorStatus> execute_1_3(const V1_3::Request& request,
                                          V1_2::MeasureTiming measure,
                                          const V1_3::OptionalTimePoint&,
                                          const V1_3::OptionalTimeoutDuration&,
                                          const sp<V1_3::IExecutionCallback>& callback) override;

        Return<void> executeSynchronously(const V1_0::Request& request,
                                            V1_2::MeasureTiming measure,
                                            executeSynchronously_cb cb) override;
        Return<void> executeSynchronously_1_3(const V1_3::Request &request,
                                          V1_2::MeasureTiming measure,
                                          const V1_3::OptionalTimePoint& deadline,
                                          const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                                          V1_3::IPreparedModel::executeSynchronously_1_3_cb cb) override;

        Return<void> executeFenced(const V1_3::Request& request,
                               const android::hardware::hidl_vec<android::hardware::hidl_handle>& fenceWaitFor,
                               V1_2::MeasureTiming measure,
                               const V1_3::OptionalTimePoint& deadline,
                               const V1_3::OptionalTimeoutDuration& loopTimeoutDuration,
                               const V1_3::OptionalTimeoutDuration& duration,
                               executeFenced_cb callback) {
            callback(ErrorStatus::GENERAL_FAILURE, hidl_handle(nullptr), nullptr);
            return Void();
        }


        Return<void> configureExecutionBurst(
            const sp<V1_2::IBurstCallback>& callback,
            const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
            const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel,
            configureExecutionBurst_cb cb) override;

        bool initialize();

        std::shared_ptr<NnapiModelInfo> getModelInfo() {
            return mModelInfo;
        }

        std::shared_ptr<NgraphNetworkCreator> getNgraphNwCreator() {
            return mNgraphCreator;
        }

        IDevicePlugin* getPlugin() {
            return mPlugin;
        }
        
protected:
        void deinitialize();
        //bool initializeRunTimeOperandInfo();
        //void initializeInput();
        //bool finalizeOutput();

        IntelDeviceType mTargetDevice;
        std::shared_ptr<NnapiModelInfo> mModelInfo;
        std::shared_ptr<NgraphNetworkCreator> mNgraphCreator;
        CNNNetwork* mNet;
        IDevicePlugin* mPlugin;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_PREPAREDMODEL_H
