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

#include "Softmax.h"
#include "CpuPreparedModel.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace softmax{

OutputPort softmaxDataPtr;

inline static OutputPort Softmax(const OutputPort &src) {
    auto inputDims = src->getTensorDesc().getDims();

    std::string name = "Softmax-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto l = std::make_shared<InferenceEngine::SoftMaxLayer>(prm);
    l->type = "SoftMax";
    src >> l;
    // addOutput(l, src->getTensorDesc().getDims());
    addOutput(l, inputDims);

    return output(l);
}

bool validate(const Operation& operation, const Model& model){
    VLOG(L1, "Validating SOFTMAX operation params");
    const auto& input0 = model.operands[operation.inputs[OP_INPUT_IDX_CONV]];
    const auto& input1 = model.operands[operation.inputs[1]];
    float beta = getOperandConstVal<float>(model, input1);

    const auto& output = model.operands[operation.outputs[0]];

    // Check Input/Filter  Operand type
    if (input0.type != OperandType::TENSOR_FLOAT32 || input1.type != OperandType::FLOAT32) {
        VLOG(L1, "NNERR: input0/input1 invalid operand types");
        return false;
    }

    if (output.type != OperandType::TENSOR_FLOAT32) {
        VLOG(L1, "NNERR:output operand types invalid,aborting!!");
        return false;
    }
    if (beta != 1) {
        VLOG(L1,"NNERR:beta equal to 1 only supported");
        return false;
    }
    return true;
}

bool initialize(const char* device, const Operation& operation, const Model& model){
    if (strcmp(device, "CPU") == 0){
        VLOG(L1, "OperationType::SOFTMAX");
        sp<CpuPreparedModel> PreparedModelObj;

        auto input = PreparedModelObj->getPort(operation.inputs[0], model);

        softmaxDataPtr = Softmax(input);
        float beta /*scale*/ = PreparedModelObj->ParseOperationInput<float>(model, operation, 1);

        VLOG(L1, "Softmax beta = %f ", beta);

        if (beta <= 0.0f) {
            ALOGE("beta must be positive for softmax");
            nnAssert(false);
        }

        return true;
    } else if (strcmp(device, "GNA") == 0){
        return false;
    } else {
        return false;
    }
}

OutputPort updateDataPtr() {
    return softmaxDataPtr;
}


}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android