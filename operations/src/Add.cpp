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

#include "Add.h"
#include "CpuPreparedModel.h"
#include "fusion.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace add{

OutputPort addDataPtr;

bool validate(const Operation& operation, const Model& model){
    const auto& input0 = model.operands[operation.inputs[OP_INPUT0_IDX_ADD]];
    const auto& input1 = model.operands[operation.inputs[OP_INPUT1_IDX_ADD]];
    const auto& input2 = model.operands[operation.inputs[OP_ACTV_IDX_ADD]];
    if (input0.dimensions != input1.dimensions) {
        VLOG(L1, "NNERR:dims not match");
        return false;
    }

    if (input0.type != input1.type) {
        VLOG(L1, "NNERR:input0 and input1 type not equal,aborting!!");
        return false;
    }

    if (input2.type != OperandType::INT32) {
        VLOG(L1, "NNERR:input2 type invalid,aborting!!");
        return false;
    }

    const auto& output = model.operands[operation.outputs[0]];

    if (output.type != input0.type) {
        VLOG(L1, "NNERR: output type not equalt to input0 type ,aborting!!");
        return false;
    }
    return true;
}

bool initialize(const std::string& device, const Operation& operation, const Model& model){
    if (device.compare("CPU")){

    VLOG(L1, "OperationType::ADD");
    OutputPort out;
    sp<CpuPreparedModel> PreparedModelObj;
    bool isIn0Const = PreparedModelObj->isConst(operation.inputs[OP_INPUT_IDX_CONV]);
    bool isIn1Const = PreparedModelObj->isConst(operation.inputs[OP_FILTER_IDX_CONV]);
    VLOG(L1, "isIn0Const = %d isIn1Const = %d \n", isIn0Const, isIn1Const);
    if (isIn0Const || isIn1Const) {
        if (isIn0Const && isIn1Const) {
            VLOG(L1, "adding 2 constants, we can do it now and put const as output");
            nnAssert(true);
        }
        // this will use ScaleShift
        if (isIn0Const)  // if op.inputs[OP_FILTER_IDX] is a Model input
            out = AddConst(
                PreparedModelObj->mNet, PreparedModelObj->getPort(operation.inputs[OP_FILTER_IDX_CONV]),
                PreparedModelObj->GetConstOperandAsTensor(operation.inputs[OP_INPUT_IDX_CONV], OP_INPUT_IDX_CONV));
        else  // isIn1Const is const //op.inputs[OP_INPUT_IDX_CONV] is a Model input
            out = AddConst(
                PreparedModelObj->mNet, PreparedModelObj->getPort(operation.inputs[OP_INPUT_IDX_CONV]),
                PreparedModelObj->GetConstOperandAsTensor(operation.inputs[OP_FILTER_IDX_CONV], OP_FILTER_IDX_CONV));
    } else {  // both inputs[OP_INPUT_IDX_CONV] & inputs[OP_FILTER_IDX_CONV] aremodel inputs
        out = PreparedModelObj->getPort(operation.inputs[OP_INPUT_IDX_CONV]) +
              PreparedModelObj->getPort(operation.inputs[OP_FILTER_IDX_CONV]);
    }
    // check fusion
    VLOG(L1, "check fusion parameter = %d\n",PreparedModelObj->ParseOperationInput<int32_t>(model, operation, 2));

    addDataPtr = handleFusion(out, PreparedModelObj->ParseOperationInput<int32_t>(model, operation, 2));

    return true;
    
    } else if (device.compare("GNA")){
        return false;
    } else {
        return false;
    }
}

OutputPort updateDataPtr() {
    return addDataPtr;
}

}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android