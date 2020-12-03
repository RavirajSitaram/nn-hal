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
namespace fullyconnected{

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

    bool initialize(const std::string& device){
        if (device.compare("CPU")){
            VLOG(L1, "OperationType::FULLY_CONNECTED");
            dumpOperationParam(operation);

            auto input = getPort(operation.inputs[OP_INPUT_IDX_CONV]);
            auto weights =
                GetConstOperandAsTensor(operation.inputs[OP_FILTER_IDX_CONV], OP_FILTER_IDX_CONV);
            auto bias = GetConstOperandAsTensor(operation.inputs[OP_BIAS_IDX_CONV], OP_BIAS_IDX_CONV);

            auto inputDims = input->getTensorDesc().getDims();
            for (auto i = 0; i < inputDims.size(); i++) VLOG(L1, "input dims[%d] = %d ", i, inputDims[i]);

            auto weightsDims = weights->getTensorDesc().getDims();
            for (auto i = 0; i < weightsDims.size(); i++)
                VLOG(L1, "weights dims[%d] = %d ", i, weightsDims[i]);

            auto biasDims = bias->getTensorDesc().getDims();

            nnAssert(inputDims.size() >= 2);
            nnAssert(weightsDims.size() == 2);
            uint32_t numInputElements = sizeOf(inputDims);
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

                input = Reshape(outDims, input);
            }

            const auto newInputDims = input->getTensorDesc().getDims();

            auto out = weights * input + bias;

            mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(3));

            VLOG(L1, "----------------------------------------------");
            VLOGDIMS(L1, inputDims, "inputs dims");
            VLOGDIMS(L1, newInputDims, "newInput dims");
            VLOGDIMS(L1, weightsDims, "weights dims");
            VLOG(L1, "----------------------------------------------");

            return true;

        } else if (device.compare("GNA")){
            VLOG(L1, "OperationType::FULLY_CONNECTED");

            auto getOperandLifeTime = [&](uint32_t idx) {
                const auto op = mModel.operands[idx];
                return (int)op.lifetime;
            };

            auto getIRBlobFromOperand = [&](uint32_t idx, uint32_t offset) {
                const auto op = mModel.operands[idx];

                auto blob = GetConstOperandAsTensor(idx, offset);
                if (op.lifetime == OperandLifeTime::MODEL_INPUT)
                {
                    mOpIndex2BlobMap[idx] = blob;
                    VLOG(L1, "blob idx=%d (model_input) ptr=%p", idx, blob.get());
                }

                return blob;
            };

            IRBuilder::BuilderFCLayer::FCParams params;
            auto input = getIRBlobFromOperand(operation.inputs[0], 0);

            params.weights.data = getIRBlobFromOperand(operation.inputs[1], 1);
            params.weights.lifeTime = getOperandLifeTime(operation.inputs[1]);

            params.bias.data = getIRBlobFromOperand(operation.inputs[2], 2);
            params.bias.lifeTime = getOperandLifeTime(operation.inputs[2]);

            auto inputDims = input->getTensorDesc().getDims();
            for (auto i = 0; i < inputDims.size(); i++) VLOG(L1, "input dims[%d] = %d ", i, inputDims[i]);

            auto weightsDims = params.weights.data->getTensorDesc().getDims();
            for (auto i = 0; i < weightsDims.size(); i++)
                VLOG(L1, "weights dims[%d] = %d ", i, weightsDims[i]);

            auto biasDims = params.bias.data->getTensorDesc().getDims();

            nnAssert(inputDims.size() >= 2);
            nnAssert(weightsDims.size() == 2);
            uint32_t numInputElements = sizeOfTensor(inputDims);
            uint32_t num_units = weightsDims[0];
            uint32_t input_size = weightsDims[1];
            uint32_t batch_size = numInputElements / input_size;
            nnAssert(biasDims[0] == num_units);
            nnAssert(input_size * batch_size == numInputElements);

            const auto newInputDims = input->getTensorDesc().getDims();


            if (mBuilderModel == nullptr) {
                VLOG(L1, "mBuilder = nullptr !!!");
                // ASSERT
            }

            mPorts[operation.outputs[0]] = mBuilderModel->createFC(params, input);

            VLOG(L1, "----------------------------------------------");
            VLOGDIMS(L1, inputDims, "inputs dims");
            VLOGDIMS(L1, newInputDims, "newInput dims");
            VLOGDIMS(L1, weightsDims, "weights dims");
            VLOG(L1, "----------------------------------------------");

            return true;

        } else {
            return false;
        }
    }

    bool createGraph(){
        return true;
    }
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android