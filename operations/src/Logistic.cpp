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

#include "Logistic.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace logistic{

bool validate(const Operation& operation, const Model& model){
    const auto& input0 = model.operands[operation.inputs[0]];
    const auto& output = model.operands[operation.outputs[0]];

    if (input0.dimensions.size() > 4 || input0.type != OperandType::TENSOR_FLOAT32 ||
        operation.outputs.size() > 1 || operation.inputs.size() > 1 ||
        output.type != OperandType::TENSOR_FLOAT32) {
        VLOG(L1, "NNERR: input/output  params invalid for Relu/Logit, aborting!!");
        return false;
    }

    if (input0.dimensions[0] > 1) {
        VLOG(L1, "NNERR:batch size more than 1 not supported for relu/logit");
        return false;
    }
    return true;
}

bool initialize(const char* device, const Operation& operation, const Model& model){
    if (strcmp(device, "CPU") == 0){
        return false;
    } else if (strcmp(device, "GNA") == 0){
        return false;
    } else {
        return false;
    }
}


}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android