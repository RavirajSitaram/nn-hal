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
#include "Relu.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace relu{

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

bool initialize(const std::string& device){
    if (device.compare("CPU")){
        VLOG(L1, "OperationType::RELU");
        mPorts[operation.outputs[0]] = ReLU(getPort(operation.inputs[0]));
    } else if (device.compare("GNA")){
        return false;
    } else {
        return false;
    }
}

// extern const std::string ReLU = "ReLU";

static IRLayer create(const OutputPort &src, const std::string &type) {
    std::string name = type + "-";  // todo: make it unique
    name = name << layer_name_count++;
    IRLayer layer;
    if ((strncasecmp(type.c_str(), "relu", type.size()) == 0)) {
        InferenceEngine::LayerParams prm;
        prm.precision = g_layer_precision;
        prm.name = name;
        layer = std::make_shared<InferenceEngine::ReLULayer>(prm);
        layer->type = "ReLU";
    } else {
        InferenceEngine::LayerParams prm;
        prm.precision = g_layer_precision;
        prm.name = name;
        layer = std::make_shared<InferenceEngine::CNNLayer>(prm);
        layer->type = "Activation";
        addAttr(layer, "type", type);
    }

    src >> layer;

    std::vector<size_t> dims = src->getTensorDesc().getDims();
#ifdef NNLOG
    for (int i = 0; i < dims.size(); i++) {
        ALOGI("Activation function output dims[%d] = %lu ", i, dims[i]);
    }
#endif

    addOutput(layer, src->getTensorDesc().getDims());
    return layer;
}

static IRLayer create(const IRLayer &src, const std::string &type) {
    return create(output(src), type);
}

template <typename T>
OutputPort ReLU(const T &src) {
    return output(create(src, ReLU));
}

}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android