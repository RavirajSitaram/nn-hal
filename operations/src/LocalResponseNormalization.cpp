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

#include "LocalResponseNormalization.h"
#include "CpuPreparedModel.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace lrn{

OutputPort lrnDataPtr;

inline OutputPort LRN(const OutputPort &src, float alpha, float beta, int local_size,
                      bool isAcross = true, float k = 1) {
    auto inp = src;
    std::string name = "Norm-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto l = std::make_shared<InferenceEngine::NormLayer>(prm);
    l->type = "Norm";

    src >> l;
    l->_alpha = alpha;
    l->_beta = beta;
    l->_isAcrossMaps = isAcross;
    l->_size = local_size;
    l->_k = (unsigned int)k;
    return addOutput(l, inp->getTensorDesc().getDims());
}

bool validate(const Operation& operation, const Model& model){
    return true;
}

bool initialize(const std::string& device, const Operation& operation, const Model& model){
    if (device.compare("CPU")){

        VLOG(L1, "OperationType::LOCAL_RESPONSE_NORMALIZATION");
        sp<CpuPreparedModel> PreparedModelObj;

        float alpha = PreparedModelObj->ParseOperationInput<float>(model, operation, 3);
        float beta = PreparedModelObj->ParseOperationInput<float>(model, operation, 4);
        int size = PreparedModelObj->ParseOperationInput<int32_t>(model, operation, 1);
        float k = PreparedModelObj->ParseOperationInput<float>(model, operation, 2);
        lrnDataPtr = LRN(PreparedModelObj->getPort(operation.inputs[0]), alpha, beta, size, false, k);
        return true;
    } else if (device.compare("GNA")){
        return false;
    } else {
        return false;
    }
}

OutputPort updateDataPtr() {
    return lrnDataPtr;
}

}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android