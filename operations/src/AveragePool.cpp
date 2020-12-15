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

#include "AveragePool.h"
#include "CpuPreparedModel.h"
#include "fusion.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace avgpool{

OutputPort avgpoolDataPtr;

inline OutputPort Pooling(const OutputPort &inp, const Point2D &kernel, const Point2D &stride,
                          const Point2D &pad_start, const Point2D &pad_end, std::string padType,
                          InferenceEngine::PoolingLayer::PoolType type) {
    auto src = inp;
    std::string name = "Pooling-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto ret = std::make_shared<InferenceEngine::PoolingLayer>(prm);
    ret->type = "Pooling";

    ret->_kernel.clear();
    ret->_kernel.insert(InferenceEngine::X_AXIS, kernel.x);
    ret->_kernel.insert(InferenceEngine::Y_AXIS, kernel.y);
    ret->_stride.clear();
    ret->_stride.insert(InferenceEngine::X_AXIS, stride.x);
    ret->_stride.insert(InferenceEngine::Y_AXIS, stride.y);
    ret->_padding.clear();
    ret->_padding.insert(InferenceEngine::X_AXIS, pad_start.x);
    ret->_padding.insert(InferenceEngine::Y_AXIS, pad_start.y);
    ret->_pads_end.clear();
    ret->_pads_end.insert(InferenceEngine::X_AXIS, pad_end.x);
    ret->_pads_end.insert(InferenceEngine::Y_AXIS, pad_end.y);
    ret->_type = type;
    ret->_exclude_pad = true;

#ifdef NNLOG
    //        ALOGI("Pooling  kernel.x= %d kernel.y= %d stride.x= %d stride.y= %d pad_start.x= %d pad_start.y= %d \
//        pad_end.x= %d pad_end.y= %d ", kernel.x, kernel.y, stride.x, stride.y, pad_start.x, pad_start.y, pad_end.x, pad_end.y);
#endif
    //<data exclude-pad="true" kernel-x="4" kernel-y="4" pad-x="0" pad-y="0" pool-method="avg"
    // stride="1,1,2,2" stride-x="2" stride-y="2"/>
    ret->params["auto_pad"] = padType;
    ret->params["_exclude_pad"] = std::to_string(ret->_exclude_pad);
    ret->params["kernel-x"] = std::to_string(ret->_kernel.at(InferenceEngine::X_AXIS));
    ret->params["kernel-y"] = std::to_string(ret->_kernel.at(InferenceEngine::Y_AXIS));
    ret->params["pad-begin-x"] = std::to_string(ret->_padding.at(InferenceEngine::X_AXIS));
    ret->params["pad-begin-y"] = std::to_string(ret->_padding.at(InferenceEngine::Y_AXIS));
    ret->params["pad-end-x"] = std::to_string(ret->_pads_end.at(InferenceEngine::X_AXIS));
    ret->params["pad-end-y"] = std::to_string(ret->_pads_end.at(InferenceEngine::Y_AXIS));
    std::string poolingType =
        ret->_type == InferenceEngine::PoolingLayer::PoolType::AVG ? "avg" : "max";
    ret->params["pool-method"] = poolingType;  // std::to_string(poolingType);
    ret->params["stride-x"] = std::to_string(ret->_stride.at(InferenceEngine::X_AXIS));
    ret->params["stride-y"] = std::to_string(ret->_stride.at(InferenceEngine::Y_AXIS));

    src >> ret;

    auto inDims = src->getTensorDesc().getDims();

    if (padType == "explicit") {
        Point2D in_size = {static_cast<int>(inDims[3]), static_cast<int>(inDims[2])};
        // todo: handle uneven padding
        Point2D out_size = (in_size - kernel + pad_start + pad_end + stride) /
                           stride;  // add stride-1 to round ceiling

        addOutput(ret, {inDims[0], inDims[1], (size_t)out_size.y, (size_t)out_size.x});
    } else {
        // Calculate output height and width for uneven padding
        float OHTemp = 1.f, OWTemp = 1.f;
        size_t inputN = inDims[0];
        size_t IC = inDims[1];
        size_t IH = inDims[2];
        size_t IW = inDims[3];
        size_t KH = ret->_kernel[InferenceEngine::Y_AXIS];
        size_t KW = ret->_kernel[InferenceEngine::X_AXIS];
        size_t SH = ret->_stride[InferenceEngine::Y_AXIS];
        size_t SW = ret->_stride[InferenceEngine::X_AXIS];

        if (padType == "valid") {
            OHTemp = std::ceil((IH - KH + 1.f) / SH);
            OWTemp = std::ceil((IW - KW + 1.f) / SW);
        } else if (padType == "same_upper") {
            OHTemp = std::ceil(1.f * IH / SH);
            OWTemp = std::ceil(1.f * IW / SW);
        } else if (padType == "same_lower") {
            OHTemp = std::floor(1.f * IH / SH);
            OWTemp = std::floor(1.f * IW / SW);
        }

        size_t OH = static_cast<size_t>(OHTemp);
        size_t OW = static_cast<size_t>(OWTemp);
        addOutput(ret, {inputN, IC, OH, OW});
    }

    return output(ret);
}

bool validate(const Operation& operation, const Model& model){
    int oper_size = operation.inputs.size();
    const auto& input0 = model.operands[operation.inputs[OP_INPUT_IDX_POOL]];
    VLOG(L1, "Validating AVG_POOL_2D params");

    if (input0.type != OperandType::TENSOR_FLOAT32) {
        VLOG(L1, "NNERR: input0 invalid operand types");
        return false;
    }

    // Check Input Dimension size
    if (input0.dimensions.size() != NHWC_DIM_NUM) {
        VLOG(L1, "NNERR: input-0 dim-size %d invalid,aborting!!", input0.dimensions.size());
        return false;
    }

    // Check all other Input operand types for implicit/explicit Padding

    if (oper_size == IMPL_PAD_PARAMS_POOL) {
        const auto& input1 = model.operands[operation.inputs[OP_PADSCHEME_IDX_POOL]];
        const auto& input2 = model.operands[operation.inputs[OP_STRD_WD_IDX_IMPL_POOL]];
        const auto& input3 = model.operands[operation.inputs[OP_STRD_HT_IDX_IMPL_POOL]];
        const auto& input4 = model.operands[operation.inputs[OP_FLT_WD_IDX_IMPL_POOL]];
        const auto& input5 = model.operands[operation.inputs[OP_FLT_HT_IDX_IMPL_POOL]];
        const auto& input6 = model.operands[operation.inputs[OP_ACTV_FUNC_IDX_IMPL_POOL]];

        if (input1.type != OperandType::INT32 || input2.type != OperandType::INT32 ||
            input3.type != OperandType::INT32 || input4.type != OperandType::INT32 ||
            input5.type != OperandType::INT32 || input6.type != OperandType::INT32) {
            VLOG(L1, "NNERR: inputs 1-6 invalid operand types");
            return false;
        }
    } else if (oper_size == EXPL_PAD_PARAMS_POOL) {
        const auto& input1 = model.operands[operation.inputs[OP_PADL_IDX_POOL]];
        const auto& input2 = model.operands[operation.inputs[OP_PADR_IDX_POOL]];
        const auto& input3 = model.operands[operation.inputs[OP_PADH_IDX_POOL]];
        const auto& input4 = model.operands[operation.inputs[OP_PADW_IDX_POOL]];
        const auto& input5 = model.operands[operation.inputs[OP_STRD_WD_IDX_EXPL_POOL]];
        const auto& input6 = model.operands[operation.inputs[OP_STRD_HT_IDX_EXPL_POOL]];
        const auto& input7 = model.operands[operation.inputs[OP_FLT_WD_IDX_EXPL_POOL]];
        const auto& input8 = model.operands[operation.inputs[OP_FLT_HT_IDX_EXPL_POOL]];
        const auto& input9 = model.operands[operation.inputs[OP_ACTV_FUNC_IDX_EXPL_POOL]];

        if (input1.type != OperandType::INT32 || input2.type != OperandType::INT32 ||
            input3.type != OperandType::INT32 || input4.type != OperandType::INT32 ||
            input5.type != OperandType::INT32 || input6.type != OperandType::INT32 ||
            input7.type != OperandType::INT32 || input8.type != OperandType::INT32 ||
            input9.type != OperandType::INT32) {
            VLOG(L1, "NNERR:inputs 1-9 as invalid operand types");
            return false;
        }
    }

    const auto& output = model.operands[operation.outputs[0]];

    if (output.type != OperandType::TENSOR_FLOAT32) {
        VLOG(L1, "NNERR:output operand types invalid,aborting!!");
        return false;
    }
    return true;
}

bool initialize(const std::string& device, const Operation& operation, const Model& model){
    if (device.compare("CPU")){

         VLOG(L1, "OperationType::AVERAGE_POOL_2D");
         sp<CpuPreparedModel> PreparedModelObj;
    
    auto input = PreparedModelObj->getPort(operation.inputs[0]);
    const auto indims = input->getTensorDesc().getDims();

    Point2D pad_start = {0, 0};
    Point2D pad_end = {0, 0};
    Point2D stride = {0, 0};
    Point2D kernel = {0, 0};
    std::string padType;
    int fusion_index = -1;

    if (operation.inputs.size() == 10) {
        padType = "explicit";
        pad_start = {PreparedModelObj->ParseOperationInput<int32_t>(model, operation,1), PreparedModelObj->ParseOperationInput<int32_t>(model, operation,3)};
        pad_end = {PreparedModelObj->ParseOperationInput<int32_t>(model, operation,2), PreparedModelObj->ParseOperationInput<int32_t>(model, operation,4)};
        stride = {PreparedModelObj->ParseOperationInput<int32_t>(model, operation,5), PreparedModelObj->ParseOperationInput<int32_t>(model, operation,6)};
        kernel = {PreparedModelObj->ParseOperationInput<int32_t>(model, operation,7), PreparedModelObj->ParseOperationInput<int32_t>(model, operation,8)};
        fusion_index = 9;
    } else if (operation.inputs.size() == 7) {  // implicit padding
        const auto pad_type = PreparedModelObj->ParseOperationInput<int32_t>(model, operation,1);
        int stride_width = PreparedModelObj->ParseOperationInput<int32_t>(model, operation,2);
        int stride_height = PreparedModelObj->ParseOperationInput<int32_t>(model, operation,3);
        int filter_width = PreparedModelObj->ParseOperationInput<int32_t>(model, operation,4);
        int filter_height = PreparedModelObj->ParseOperationInput<int32_t>(model, operation,5);
        fusion_index = 6;
        stride = {stride_width, stride_height};
        kernel = {filter_width, filter_height};

        int input_width = indims[3];
        int input_height = indims[2];

        int padding_left, padding_right;
        int padding_top, padding_bottom;

        if (pad_type == kPaddingSame) {

            calculateExplicitPadding(input_width, stride_width, filter_width,
                                     pad_type /*padding_implicit*/, &padding_left, &padding_right);
            calculateExplicitPadding(input_height, stride_height, filter_height,
                                     pad_type /*padding_implicit*/, &padding_top, &padding_bottom);

            pad_start = {padding_left, padding_top};
            pad_end = {padding_right, padding_bottom};
            padType = "same_upper";

        } else if (pad_type == kPaddingValid) {
            /**
             * VALID padding.
             * No padding. When the input size is not evenly divisible by
             * the filter size, the input at the end that could not fill
             * the whole filter tile will simply be ignored.
             */
            pad_start = {0, 0};
            pad_end = {0, 0};
            padType = "valid";
        }
    }

    auto out = Pooling(input, kernel, stride, pad_start, pad_end, padType,
                       InferenceEngine::PoolingLayer::PoolType::AVG);
    avgpoolDataPtr = handleFusion(out, PreparedModelObj->ParseOperationInput<int32_t>(model, operation,fusion_index));

    return true;
    
    } else if (device.compare("GNA")){
        return false;
    } else {
        return false;
    }
}

OutputPort updateDataPtr() {
    return avgpoolDataPtr;
}

}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android