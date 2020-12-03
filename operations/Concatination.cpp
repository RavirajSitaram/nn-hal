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
namespace concat{
    bool validate(const Operation& operation, const Model& model){
        return true;
    }

    bool initialize(const std::string& device){
        if (device.compare("CPU")){
            VLOG(L1, "OperationType::CONCATENATION");
            uint32_t axis;
            auto n = operation.inputs.size() - 1;
            std::vector<OutputPort> inputs;
            const auto op = mModel.operands[operation.inputs[0]];
            auto input = getPort(operation.inputs[0]);
            auto inDims = input->getTensorDesc().getDims();
            if (op.dimensions.size() == 4) {
                std::vector<uint32_t> axisMap = {2, 3, 1};  // NCHW = axisMap[NHWC]
                axis = axisMap[PARAM_I32(n)];
            } else if (op.dimensions.size() == 3) {
                std::vector<uint32_t> axisMap = {2, 3, 1};  // NCHW = axisMap[HWC]
                axis = axisMap[PARAM_I32(n)];
            }
            VLOG(L1, "shape of output tensor axis %d inDims size %d, op_dimensionsize %d", axis,
                inDims.size(), op.dimensions.size());

            for (int i = 0; i < n; i++) inputs.push_back(getPort(operation.inputs[i]));
            auto out = Concat(inputs, axis);
            std::vector<std::string> inputNames;
            for (int i = 0; i < inputs.size(); ++i) {
                inputNames.push_back(inputs[i]->getName());
            }
            mCreateNgraph->addConcat(out->getName(), inputNames, axis);

            return true;
        } else if (device.compare("GNA")){
            return false;
        } else {
            return false;
        }
    }

    inline OutputPort createGraph(const std::vector<OutputPort> inputs, int axis = 1){
        std::string name = "Concat-";  // todo: make it unique
        name = name << layer_name_count++;
        InferenceEngine::LayerParams prm;
        prm.precision = g_layer_precision;
        prm.name = name;
        auto ret = std::make_shared<InferenceEngine::ConcatLayer>(prm);
        ret->type = "Concat";
        addAttr(ret, "axis", axis);
        inputs[0] >> ret;
        auto outDim = inputs[0]->getTensorDesc().getDims();
        // it was fixed, should be backward compatiobale though...
        // axis = static_cast<int>(outDim.size()) - axis - 1; // todo: we are all in reverse here :-(
        auto axisSize = outDim[axis];
        for (int i = 1; i < inputs.size(); ++i) {
            inputs[i] >> ret;
            axisSize += inputs[i]->getTensorDesc().getDims()[axis];
        }
        outDim[axis] = axisSize;
        return addOutput(ret, outDim);
    }
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android