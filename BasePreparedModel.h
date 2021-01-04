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
#include <hardware/hardware.h>
#include <hidlmemory/mapping.h>
#include <sys/mman.h>
#include <fstream>
#include <string>

#include "Driver.h"
#include "utils.h"
#include "create_ngraph.hpp"
#include "IENetwork.h"


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

// The type and dimensions of an operand.
struct Shape {
    OperandType type;
    std::vector<uint32_t> dimensions;
    float scale;
    int32_t offset;
};

// Information we maintain about each operand during execution that
// may change during execution.
struct RunTimeOperandInfo {
    OperandType type;
    std::vector<uint32_t> dimensions;
    float scale;
    int32_t zeroPoint;
    uint8_t* buffer;
    uint32_t length;
    OperandLifeTime lifetime;
    uint32_t numberOfUsesLeft;
    Operand::ExtraParams extraParams;
    Shape shape() const {
        return {
                .type = type,
                .dimensions = dimensions,
                .scale = scale,
                .offset = zeroPoint,
        };
    }
};

// Used to keep a pointer to each of the memory pools.
struct RunTimePoolInfo {
    sp<IMemory> memory;
    hidl_memory hidlMemory;
    uint8_t* buffer;

    bool set(const hidl_memory& hidlMemory);
    bool update();
};

bool setRunTimePoolInfosFromHidlMemories(std::vector<RunTimePoolInfo>* poolInfos,
                                         const hidl_vec<hidl_memory>& pools);

// template <typename T_IExecutionCallback>;
class BasePreparedModel : public V1_2::IPreparedModel{
    public:
        BasePreparedModel(const Model& model)
        : mTargetDevice("CPU"),
          mModel(model),
          mNet("nnNet"),
          enginePtr(nullptr){
            g_layer_precision = InferenceEngine::Precision::FP16;
            mUseNgraph =
                isNgraphPropSet();  // TODO:Should additionally check if all the ops are supported
            mCreateNgraph = std::make_shared<CreateNgraph>();
        }
        BasePreparedModel(const std::string device, const Model& model)
        : mTargetDevice(device),
          mModel(model),
          mNet("nnNet"),
          enginePtr(nullptr) {
            if (mTargetDevice == "CPU" || mTargetDevice == "GPU")
                g_layer_precision = InferenceEngine::Precision::FP32;
            else if (mTargetDevice == "MYRIAD")
                g_layer_precision = InferenceEngine::Precision::FP16;
            else
                g_layer_precision = InferenceEngine::Precision::UNSPECIFIED;
            mUseNgraph = isNgraphPropSet();
            mCreateNgraph = std::make_shared<CreateNgraph>();
        }

        virtual ~BasePreparedModel() { deinitialize(); }

        Return<ErrorStatus> execute(const Request& request,
                                const sp<V1_0::IExecutionCallback>& callback) override;
        Return<ErrorStatus> execute_1_2(const Request& request, MeasureTiming measure,
                                        const sp<V1_2::IExecutionCallback>& callback) override;
        Return<void> executeSynchronously(const Request& request, MeasureTiming measure,
                                        executeSynchronously_cb cb) override;
        Return<void> configureExecutionBurst(
            const sp<V1_2::IBurstCallback>& callback,
            const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
            const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel,
            configureExecutionBurst_cb cb) override;

        static bool isOperationSupported(const Operation& operation, const Model& model);
        virtual bool initialize(const Model& model);
        bool isConst(int index, const Model& model);
        OutputPort getPort(int index, const Model& model);
        template <typename T>
        T ParseOperationInput(const Model& model, const Operation& operation, uint32_t index);
        template <typename T>
        std::vector<T> GetConstVecOperand(const Model& model, uint32_t index); //for reshape

        IRDocument mNet; //public for add operation
        std::shared_ptr<CreateNgraph> mCreateNgraph; //for operations
protected:
        virtual void deinitialize();
        bool initializeRunTimeOperandInfo();
        void initializeInput();
        bool finalizeOutput();
        template <typename T>
        T GetConstOperand(const Model& model, uint32_t index);

        template <typename T>
        T GetConstFromBuffer(const uint8_t* buf, uint32_t len);
        template <typename T>
        std::vector<T> GetConstVecFromBuffer(const uint8_t* buf, uint32_t len);
        const uint8_t* GetOperandMemory(const Model& model, uint32_t index, uint32_t& len_out);
        virtual Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index, const Model& model);
        virtual Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx, const Model& model);
        virtual Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                            uint32_t& len);
        template <typename T_IExecutionCallback>
        Return<ErrorStatus> executeBase(const Request& request, MeasureTiming measure,
                                               const sp<T_IExecutionCallback>& callback);
        template <typename T_IExecutionCallback>
        void asyncExecute(const Request& request, MeasureTiming measure,
                                 time_point driverStart,
                                 const sp<T_IExecutionCallback>& callback);
        // OutputPort getPort(int index);

        std::string mTargetDevice;
        Model mModel;
        std::vector<RunTimeOperandInfo> mOperands;
        std::vector<RunTimePoolInfo> mPoolInfos;
        std::vector<OutputPort> mPorts;  // typedef std::shared_ptr<Data> DataPtr;
        ExecuteNetwork* enginePtr;

        bool mUseNgraph = true;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_PREPAREDMODEL_H
