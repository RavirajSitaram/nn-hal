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
#ifndef FUSION_H
#define FUSION_H

#include "common.h"
#include "Relu.h"
#include "CpuPreparedModel.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace {

inline OutputPort Clamp(const OutputPort &src, float min, float max) {
    std::string name = "Clamp-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prms;
    prms.precision = g_layer_precision;
    prms.name = name;
    auto layer = std::make_shared<InferenceEngine::ClampLayer>(prms);
    layer->type = "Clamp";
    layer->min_value = min;
    layer->max_value = max;
    layer->params["max"] = std::to_string(layer->max_value);
    layer->params["min"] = std::to_string(layer->min_value);
    src >> layer;
    addOutput(layer, src->getTensorDesc().getDims());
    return output(layer);
}

inline OutputPort handleFusion(const OutputPort& out, int32_t fusedOp) {
    VLOG(L1, "fusedOp: %d", fusedOp);
    OutputPort ret = out;
    sp<CpuPreparedModel> PreparedModelObj;
    if (fusedOp == (int32_t)FusedActivationFunc::RELU) {
        VLOG(L1, "fusedOp is RELU");
        ret = relu::ReLU(out);
        PreparedModelObj->mCreateNgraph->addRelu(ret->getName(), out->getName());
    } else if (fusedOp == (int32_t)FusedActivationFunc::RELU1) {
        VLOG(L1, "fusedOp is RELU1");
        ret = Clamp(out, -1, 1);
        PreparedModelObj->mCreateNgraph->addClamp(ret->getName(), out->getName(), -1, 1);
    } else if (fusedOp == (int32_t)FusedActivationFunc::RELU6) {
        VLOG(L1, "fusedOp is RELU6");
        ret = Clamp(out, 0, 6);
        PreparedModelObj->mCreateNgraph->addClamp(ret->getName(), out->getName(), 0, 6);
    }

    VLOG(L1, "No ActivationFunc");
    return ret;
}


}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // FUSION_H