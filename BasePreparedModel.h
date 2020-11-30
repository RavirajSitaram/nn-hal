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

#ifndef ANDROID_ML_NN_PREPAREDMODEL_H
#define ANDROID_ML_NN_PREPAREDMODEL_H

#include <android/hardware/neuralnetworks/1.2/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.2/types.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hardware/hardware.h>
#include <hidlmemory/mapping.h>
#include <sys/mman.h>
#include <fstream>
#include <string>

#include "Driver.h"

using ::android::hardware::MQDescriptorSync;
using ::android::hidl::memory::V1_0::IMemory;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace {

class BasePreparedModel : public V1_2::IPreparedModel{
    public:
        BasePreparedModel(const Model& model)
        : mTargetDevice("CPU"),
          mModel(model),
          mNet("nnNet"),
          enginePtr(nullptr),
          mPadreq(EXPL_PAD) {
            g_layer_precision = InferenceEngine::Precision::FP16;
            mUseNgraph =
                isNgraphPropSet();  // TODO:Should additionally check if all the ops are supported
            mCreateNgraph = std::make_shared<CreateNgraph>();
        }
        BasePreparedModel(const std::string device, const Model& model)
        : mTargetDevice(device),
          mModel(model),
          mNet("nnNet"),
          enginePtr(nullptr),
          mPadreq(EXPL_PAD) {
            if (mTargetDevice == "CPU" || mTargetDevice == "GPU")
                g_layer_precision = InferenceEngine::Precision::FP32;
            else if (mTargetDevice == "MYRIAD")
                g_layer_precision = InferenceEngine::Precision::FP16;
            else
                g_layer_precision = InferenceEngine::Precision::UNSPECIFIED;
            mUseNgraph = isNgraphPropSet();
            mCreateNgraph = std::make_shared<CreateNgraph>();
        }
        ~BasePreparedModel() override { deinitialize(); }
        static bool isOperationSupported(const Operation& operation, const Model& model);
    protected:
        void initializeInput();
        bool finalizeOutput();
        
        std::string mTargetDevice;
        Model mModel;
        std::vector<RunTimeOperandInfo> mOperands;
        std::vector<RunTimePoolInfo> mPoolInfos;
        IRDocument mNet;
        std::vector<OutputPort> mPorts;  // typedef std::shared_ptr<Data> DataPtr;
        ExecuteNetwork* enginePtr;
        uint32_t mPadreq;
        std::shared_ptr<CreateNgraph> mCreateNgraph;
        bool mUseNgraph;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_PREPAREDMODEL_H