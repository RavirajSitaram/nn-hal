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

#include "common.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace add{

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

bool initialize(const std::string& device){
    if (device.compare("CPU")){

    VLOG(L1, "OperationType::ADD");
    OutputPort out;
    bool isIn0Const = isConst(operation.inputs[OP_INPUT_IDX_CONV]);
    bool isIn1Const = isConst(operation.inputs[OP_FILTER_IDX_CONV]);
    VLOG(L1, "isIn0Const = %d isIn1Const = %d \n", isIn0Const, isIn1Const);
    if (isIn0Const || isIn1Const) {
        if (isIn0Const && isIn1Const) {
            VLOG(L1, "adding 2 constants, we can do it now and put const as output");
            nnAssert(true);
        }
        // this will use ScaleShift
        if (isIn0Const)  // if op.inputs[OP_FILTER_IDX] is a Model input
            out = AddConst(
                mNet, getPort(operation.inputs[OP_FILTER_IDX_CONV]),
                GetConstOperandAsTensor(operation.inputs[OP_INPUT_IDX_CONV], OP_INPUT_IDX_CONV));
        else  // isIn1Const is const //op.inputs[OP_INPUT_IDX_CONV] is a Model input
            out = AddConst(
                mNet, getPort(operation.inputs[OP_INPUT_IDX_CONV]),
                GetConstOperandAsTensor(operation.inputs[OP_FILTER_IDX_CONV], OP_FILTER_IDX_CONV));
    } else {  // both inputs[OP_INPUT_IDX_CONV] & inputs[OP_FILTER_IDX_CONV] aremodel inputs
        out = getPort(operation.inputs[OP_INPUT_IDX_CONV]) +
              getPort(operation.inputs[OP_FILTER_IDX_CONV]);
    }
    // check fusion
    VLOG(L1, "check fusion parameter = %d\n", PARAM_I32(2));

    mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(2));

    VLOG(L1, "add mPorts[%d]->name %s + mPorts[%d]->name %s  = mPorts[%d]->name %s \n",
         operation.inputs[0],
         isIn0Const ? "isIn0Const" : mPorts[operation.inputs[0]]->getName().c_str(),
         operation.inputs[1],
         isIn1Const ? "isIn1Const" : mPorts[operation.inputs[1]]->getName().c_str(),
         operation.outputs[0], mPorts[operation.outputs[0]]->getName().c_str());

    return true;
    
    } else if (device.compare("GNA")){
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