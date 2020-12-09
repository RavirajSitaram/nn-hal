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

#include "Mul.h"
#include "CpuPreparedModel.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace mul{

OutputPort mulDataPtr;

inline static IRLayer create(const OutputPort &src1, const OutputPort &src2) {
    std::string name = "Mul-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto mul = std::make_shared<InferenceEngine::EltwiseLayer>(prm);
    mul->type = "Mul";
    mul->_operation = InferenceEngine::EltwiseLayer::Prod;
    src1 >> mul;
    src2 >> mul;
    if (src1->getTensorDesc().getDims() != src2->getTensorDesc().getDims())
        THROW_IE_EXCEPTION << "input sizes for Element wise Mul do not match";
    addOutput(mul, src1->getTensorDesc().getDims());
    return mul;
}

inline OutputPort operator*(const OutputPort &a, const OutputPort &b) {
    return output(create(a, b));
}

bool validate(const Operation& operation, const Model& model){
    return true;
}

bool initialize(const std::string& device, const Operation& operation, const Model& model){
    if (device.compare("CPU")){
        VLOG(L1, "OperationType::MUL");
        sp<CpuPreparedModel> PreparedModelObj;
        mulDataPtr =
            handleFusion(PreparedModelObj->getPort(operation.inputs[0]) * PreparedModelObj->getPort(operation.inputs[1]), PreparedModelObj->ParseOperationInput<int32_t>(model, operation, 2));
        return true;
    } else if (device.compare("GNA")){
        return false;
    } else {
        return false;
    }
}

OutputPort updateDataPtr() {
    return mulDataPtr;
}

}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android