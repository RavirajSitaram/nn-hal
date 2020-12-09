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

#include "FullyConnected.h"
#include "CpuPreparedModel.h"
#include "Reshape.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace fullyconnected{

    OutputPort fcDataPtr;
inline static IRLayer create(const IRBlob::Ptr &weights, const OutputPort &src) {
#ifdef NNLOG
    ALOGI("Create FC layer");
#endif
    std::string name = "FC-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;

    auto inDims = src->getTensorDesc().getDims();  // (batch, IFM)

    auto wDim = weights->getTensorDesc().getDims();

    IR_ASSERT(inDims.size() == 2);

    unsigned int ofm = 0;
    if (wDim.size() == 2) {
#ifdef NNLOG
        ALOGI("inDims[0] = %d inDims[1] = %d", inDims[0], inDims[1]);
        ALOGI("wDim[0] = %d wDim[1] = %d", wDim[0], wDim[1]);
#endif

        IR_ASSERT(inDims[1] == wDim[1]);           // Weights: (Out,In)
        ofm = static_cast<unsigned int>(wDim[0]);  // Out
    } else if (wDim.size() == 1)                   // linear, just a blob, line in IR
    {
        ofm = static_cast<unsigned int>(weights->size() / inDims[1]);
        IR_ASSERT(inDims[1] * ofm == weights->size());  // should be divided properly
    } else
        THROW_IE_EXCEPTION << "expecting weights for FC only as 1 dim (blob) or 2 dim (Matrix)";

    auto fc = std::make_shared<InferenceEngine::FullyConnectedLayer>(prm);
    fc->type = "FullyConnected";

    fc->_out_num = ofm;
    addAttr(fc, "out-size ", ofm);  // aks added
    // todo: assert that input should be cols
    addOutput(fc, {inDims[0], static_cast<uint32_t>(fc->_out_num)});
    src >> fc;
    fc->_weights = weights;
    fc->blobs["weights"] = weights;  // todo: have setter for those layers...
    return fc;
}



inline InferenceEngine::CNNLayer::Ptr operator*(const IRBlob::Ptr &weights, const IRLayer &b) {
    std::cout << "FCLayer::create operator*(const IRBlob::Ptr &weights, const IRLayer &b)"
              << std::endl;
    return create(weights, output(b));
}

inline OutputPort operator*(const IRBlob::Ptr &weights, const OutputPort &op) {
    std::cout << "FCLayer::create operator*(const IRBlob::Ptr &weights, const OutputPort &op)"
              << std::endl;
    return output(create(weights, op));
}

    bool validate(const Operation& operation, const Model& model){
        const auto& input0 = model.operands[operation.inputs[OP_INPUT_IDX_FC]];

            const auto& input1 = model.operands[operation.inputs[OP_WGHT_IDX_FC]];
            const auto& input2 = model.operands[operation.inputs[OP_BIAS_IDX_FC]];

            if (input0.type != OperandType::TENSOR_FLOAT32 ||
                input1.type != OperandType::TENSOR_FLOAT32 ||
                input2.type != OperandType::TENSOR_FLOAT32) {
                VLOG(L1, "NNERR: input0/input1/input2 invalid operand types");
                return false;
            }

            if (input0.lifetime == input1.lifetime) {
                VLOG(L1, "NNERR: Filter (index %d) as model_input not supported,aborting!!",
                     operation.inputs[OP_FILTER_IDX_CONV]);
                return false;
            }

            if (input0.dimensions.size() < 2 || input1.dimensions.size() < 2 ||
                input2.dimensions.size() < 1) {
                VLOG(L1, "NNERR: input 0-2 dimensions size invalid, aborting!!");
                return false;
            }
            if (input0.dimensions[1] != input1.dimensions[1]) {
                VLOG(L1,
                     "NNERR: input0 and input1(weight) with unequal input-size "
                     "value, aborting!!");
                return false;
            }

            const auto& input3 = model.operands[operation.inputs[OP_ACTV_IDX_FC]];

            if (input3.type != OperandType::INT32) {
                VLOG(L1, "NNERR: input3  invalid operand types");
                return false;
            }

            const auto& output = model.operands[operation.outputs[0]];

            if (output.type != OperandType::TENSOR_FLOAT32) {
                VLOG(L1, "NNERR:invalid output operand types for FC ,aborting!!");
                return false;
            } 
        return true;
    }

    bool initialize(const std::string& device, const Operation& operation,  const Model& model){
        if (device.compare("CPU")){
            VLOG(L1, "OperationType::FULLY_CONNECTED");
            dumpOperationParam(operation);
            sp<CpuPreparedModel> PreparedModelObj;

            auto input = PreparedModelObj->getPort(operation.inputs[OP_INPUT_IDX_CONV]);
            auto weights =
                PreparedModelObj->GetConstOperandAsTensor(operation.inputs[OP_FILTER_IDX_CONV], OP_FILTER_IDX_CONV);
            auto bias = PreparedModelObj->GetConstOperandAsTensor(operation.inputs[OP_BIAS_IDX_CONV], OP_BIAS_IDX_CONV);

            auto inputDims = input->getTensorDesc().getDims();
            for (auto i = 0; i < inputDims.size(); i++) VLOG(L1, "input dims[%d] = %d ", i, inputDims[i]);

            auto weightsDims = weights->getTensorDesc().getDims();
            for (auto i = 0; i < weightsDims.size(); i++)
                VLOG(L1, "weights dims[%d] = %d ", i, weightsDims[i]);

            auto biasDims = bias->getTensorDesc().getDims();

            nnAssert(inputDims.size() >= 2);
            nnAssert(weightsDims.size() == 2);
            uint32_t numInputElements = sizeOfTensor(inputDims);
            uint32_t num_units = weightsDims[0];
            uint32_t input_size = weightsDims[1];
            uint32_t batch_size = numInputElements / input_size;
            nnAssert(biasDims[0] == num_units);
            nnAssert(input_size * batch_size == numInputElements);

            if (inputDims.size() > 2) {
                // todo: could be we need to rotate the input weights to reflect the different layout of
                // input tensor when it is not 2D: NHWC vs NCHW in IE
                // Reshape
                // input = Reshape({inputDims[0], product(inputDims)/inputDims[0]}, input);

                TensorDims outDims = {
                    (uint32_t)-1,
                    numInputElements / batch_size};  // fix me: find correct outDims and if -1 is fine

                int strechDim = -1;
                auto numOutputElements = 1;  // shape
                for (auto i = 0; i < outDims.size(); i++) {
                    VLOG(L1, "shape of output tensor outDims[%d] = %d ", i, outDims[i]);
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

                    VLOG(L1, "numInputElements = %d, index = %d, outDims[index] = %d", numInputElements,
                        strechDim, outDims[strechDim]);
                }

                input = reshape::Reshape(outDims, input);
            }

            const auto newInputDims = input->getTensorDesc().getDims();

            auto out = weights * input + bias;

            // mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(3));
            fcDataPtr = handleFusion(out, PreparedModelObj->ParseOperationInput<int32_t>(model, operation, 3));

            VLOG(L1, "----------------------------------------------");
            VLOGDIMS(L1, inputDims, "inputs dims");
            VLOGDIMS(L1, newInputDims, "newInput dims");
            VLOGDIMS(L1, weightsDims, "weights dims");
            VLOG(L1, "----------------------------------------------");

            return true;

        } else if (device.compare("GNA")){
            VLOG(L1, "OperationType::FULLY_CONNECTED");

            return true;

        } else {
            return false;
        }
    }

    OutputPort updateDataPtr() {
    return fcDataPtr;
}

    bool createGraph(){
        return true;
    }
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android