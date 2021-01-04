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

#include "Reshape.h"
#include "CpuPreparedModel.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace reshape{

OutputPort reshapeDataPtr;
std::string nodeName;
std::string inputName;
std::vector<size_t> shapePrm;

OutputPort Reshape(const TensorDims &newDims, const OutputPort &src) {
    if (sizeOfTensor(src->getTensorDesc().getDims()) != sizeOfTensor(newDims))
        THROW("Cannot reorder different volumes");

    /*//first implementation
        if(src->creatorLayer.lock()->type == "Reshape") // fuse reshapes
        {
            src->setDims(newDims);
            return src;
        }

        auto op = output(Generic("Reshape", src));
        op->setDims(newDims);

        return op;
    */
    // end of first implementation

    /*
     //FIX ME fuse reshape
        //if(src->creatorLayer.lock()->type == "Reshape") // fuse reshapes
        if(src->getCreatorLayer().lock()->type == "Reshape") // fuse reshapes
        {
            src->setDims(newDims);
            return src;
        }
    */

    std::string name = "Reshape-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prms;
    prms.precision = g_layer_precision;
    prms.name = name;
    auto layer = std::make_shared<InferenceEngine::ReshapeLayer>(prms);
    layer->type = "Reshape";
    src >> layer;

    layer->params["axis"] = std::to_string(layer->axis);
    layer->params["num_axes"] = std::to_string(layer->num_axes);

    /*  //check if mandatory to provide shape
        for (int i = 0; i < newDims.size(); i++)
        layer->shape[i] = static_cast<int>(newDims[i]);
        // VectorToStringI(layer->shape)
        std::string result;
        const char sep = ',';
        for (auto it : layer->shape) {
            result += std::to_string(it) + sep;
        }
        if (!result.empty()) {
            result = result.substr(0, result.size() - 2);
        }

       layer->params["dim"] = result;
    */
    addOutput(layer, newDims);
    auto op = output(layer);
    // op->setDims(newDims);

    /*
        //FIX ME : HACK for [VPU] Unsupported 1D dimensions
        if (op->getTensorDesc().getDims().size() == 1) {
        TensorDims dims = {1, newDims[0]};
        op->setDims(dims);
        #ifdef NNLOG
        ALOGI("Reshape oputput data set dims size = %lu ", op->getTensorDesc().getDims().size());
        #endif
        }
    */
    return op;
}

bool validate(const Operation& operation, const Model& model){
    return true;
}

bool initialize(const char* device, const Operation& operation, const Model& model){
    if (strcmp(device, "CPU") == 0){

        VLOG(L1, "OperationType::RESHAPE");
        sp<CpuPreparedModel> PreparedModelObj;

        auto input = PreparedModelObj->getPort(operation.inputs[0], model);
        auto inDims = input->getTensorDesc().getDims();

        auto outDims = toDims(PreparedModelObj->GetConstVecOperand<uint32_t>(model, operation.inputs[1]));
        auto numInputElements = sizeOfTensor(inDims);  // getNumberOfElements

    int strechDim = -1;
    auto numOutputElements = 1;  // shape
    VLOG(L1, "mModel outDims size[%d] ", outDims.size());
    if (outDims.size() == 3) 
        outDims.insert(outDims.begin(), 1);
    for (auto i = 0; i < outDims.size(); i++) {
        VLOG(L1, "operand1: shape of output tensor outDims[%d] = %d ", i, outDims[i]);
        if ((int)outDims[i] < 0) {
            strechDim = i;  // strechdim
            VLOG(L1, "strechDim = %d", i);
            continue;
        }
        numOutputElements *= outDims[i];  // shape
    }
    if (strechDim >= 0) {
        auto strechValue = numInputElements / numOutputElements;
        outDims[strechDim] = (uint32_t)strechValue;
        numOutputElements *= strechValue;

        VLOG(L1, "numInputElements or size = %d, index = %d, outDims[index] = %d", numInputElements,
             strechDim, outDims[strechDim]);
    }

    for (auto i = 0; i < outDims.size(); i++)
        VLOG(L1, "operand1: shape of output tensor outDims[%d] = %d ", i, outDims[i]);
    if (numInputElements != numOutputElements) {
        VLOG(L1, "numInputElements is not equal to numOutputElements", numInputElements,
             numOutputElements);
        nnAssert(false);
    }
    // Note: " error [VPU] Unsupported 1 D dimensions" for reshape output and fix me
    reshapeDataPtr = Reshape(outDims, input);
    nodeName = reshapeDataPtr->getName();
    inputName = input->getName();
    shapePrm = outDims;
    // PreparedModelObj->mCreateNgraph->addReshape(reshapeDataPtr->getName(), input->getName(), outDims);

    return true;
    
    } else if (strcmp(device, "GNA") == 0){
    
    } else {
        return false;
    }
    return false;
}

OutputPort updateDataPtr() {
    return reshapeDataPtr;
}
std::string getNodeName() {
    return nodeName;
}
std::string getInputName() {
    return inputName;
}
std::vector<size_t> getShape() {
    return shapePrm;
}

}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android