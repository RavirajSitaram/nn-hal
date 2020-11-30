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
#include <fstream>
#include <thread>
#include "ValidateHal.h"

#include <cutils/properties.h>

#define DISABLE_ALL_QUANT
#define LOG_TAG "BasePreparedModel"

using namespace android::nn;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

void BasePreparedModel::deinitialize() {
    VLOG(L1, "deinitialize");
    delete enginePtr;
    enginePtr = nullptr;
    VLOG(L1, "free engine");
}

bool BasePreparedModel::isOperationSupported(const Operation& operation, const Model& model) {
    VLOG(L1, "Check operation %d", operation.type);

#define VLOG_CHECKFAIL(fail) VLOG(L1, "Check failed: %s", fail)

    switch (operation.type) {
        
        case OperationType::FULLY_CONNECTED: {
            if(!fullyconnected::validate()) {
                return false;
            }
        } break;
        case OperationType::CONCATENATION:{
            if(!concat::validate()) {
                return false;
            }
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

void BasePreparedModel::initializeInput() {
    VLOG(L1, "initialize Input");
    for (auto i : mModel.inputIndexes) {
        int dims_size = mOperands[i].dimensions.size();

        VLOG(L1, "mPorts[%d] %s dims size %d", i, mPorts[i]->getName().c_str(), dims_size);
        VLOGDIMS(L1, mOperands[i].dimensions, "current operand inpu dims:");
        VLOGDIMS(L1, mPorts[i]->getTensorDesc().getDims(), "Real input dims:");

        auto inputDims = mPorts[i]->getTensorDesc().getDims();

        uint32_t nelem = getNumberOfElements(mOperands[i].dimensions);
        auto inputElem = sizeOf(inputDims);
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
#ifdef USE_NGRAPH
        mCreateNgraph->setResultNode(mPorts[i]->getName());
#endif

        VLOG(L1, "mPorts[%d] %s dims size %d", i, mPorts[i]->getName().c_str(), dims_size);
        VLOGDIMS(L1, mOperands[i].dimensions, "current operand Output dims:");
        VLOGDIMS(L1, mPorts[i]->getTensorDesc().getDims(), "Real Output dims:");

        auto outputDims = mPorts[i]->getTensorDesc().getDims();

        uint32_t nelem = getNumberOfElements(mOperands[i].dimensions);
        auto outputElem = sizeOf(outputDims);
        if (nelem != outputElem) {
            VLOG(L1, "set correct dims as operand output dims different than real output dims\n");
        }
    }
    return true;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
