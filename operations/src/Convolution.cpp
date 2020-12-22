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

#include "Convolution.h"
#include "CpuPreparedModel.h"
#include "fusion.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace convolution{

OutputPort convDataPtr;
GenConvParams genConvPrms;
std::string nodeName;
std::string inputName;

inline static IRLayer create(const OutputPort &src) {
    std::string name = "Conv-";  // todo: make it unique
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    name = name << layer_name_count++;
    prm.name = name;
    auto conv_layer = std::make_shared<InferenceEngine::ConvolutionLayer>(prm);
    conv_layer->type = "Convolution";
    src >> conv_layer;
    return conv_layer;
}

inline void ConvolutionParamsToGenConvParams(ConvolutionParams &cPrms, GenConvParams &gPrms,
                                             IRBlob::Ptr weights, IRBlob::Ptr biases) {
    gPrms.groups = cPrms.groups;
    float *buffer = weights->buffer().as<float *>();
    gPrms.weightsBuf = {buffer, buffer + weights->size()};
    gPrms.weightsDims = weights->getTensorDesc().getDims();
    buffer = biases->buffer().as<float *>();
    gPrms.biasesBuf = {buffer, buffer + biases->size()};
    gPrms.biasesDims = biases->getTensorDesc().getDims();
    gPrms.strides = {(size_t)cPrms.stride.x, (size_t)cPrms.stride.y};
    gPrms.pads_begin = {cPrms.pad_start.x, cPrms.pad_start.y};
    gPrms.pads_end = {cPrms.pad_end.x, cPrms.pad_end.y};
    gPrms.dilations = {1, 1};
    gPrms.pad_type = cPrms.padType.c_str();
}

inline OutputPort Convolution(const OutputPort &src, const ConvolutionParams &prms) {
    auto ret = As<InferenceEngine::ConvolutionLayer>(create(src));
    auto inDims = src->getTensorDesc().getDims();
    IR_ASSERT(inDims.size() == 4);
    IR_ASSERT((prms.kernel.size() * in_ch(src) * prms.num_output_planes) / prms.groups ==
              prms.weights->size());

    ret->_weights = prms.weights;
    ret->blobs["weights"] = prms.weights;

    ret->_biases = prms.biases;
    ret->blobs["biases"] = prms.biases;

    ret->_kernel.clear();
    ret->_kernel.insert(InferenceEngine::X_AXIS, prms.kernel.x);
    ret->_kernel.insert(InferenceEngine::Y_AXIS, prms.kernel.y);
    ret->_stride.clear();
    ret->_stride.insert(InferenceEngine::X_AXIS, prms.stride.x);
    ret->_stride.insert(InferenceEngine::Y_AXIS, prms.stride.y);
    ret->_padding.clear();
    ret->_padding.insert(InferenceEngine::X_AXIS, prms.pad_start.x);
    ret->_padding.insert(InferenceEngine::Y_AXIS, prms.pad_start.y);
    ret->_pads_end.clear();
    ret->_pads_end.insert(InferenceEngine::X_AXIS, prms.pad_end.x);
    ret->_pads_end.insert(InferenceEngine::Y_AXIS, prms.pad_end.y);

    ret->_dilation.clear();
    ret->_dilation.insert(InferenceEngine::X_AXIS, 1);
    ret->_dilation.insert(InferenceEngine::Y_AXIS, 1);

    ret->_group = prms.groups;
    ret->_out_depth = prms.num_output_planes;

    //<data dilation-x="1" dilation-y="1" group="1" kernel-x="3" kernel-y="3" output="8" pad-x="0"
    // pad-y="0" stride="1,1,2,2" stride-x="2" stride-y="2"/>

    ret->params["auto_pad"] = prms.padType;
    ret->params["dilation-x"] = std::to_string(ret->_dilation.at(InferenceEngine::X_AXIS));
    ret->params["dilation-y"] = std::to_string(ret->_dilation.at(InferenceEngine::Y_AXIS));
    ret->params["group"] = std::to_string(ret->_group);

    ret->params["kernel-x"] = std::to_string(ret->_kernel.at(InferenceEngine::X_AXIS));
    ret->params["kernel-y"] = std::to_string(ret->_kernel.at(InferenceEngine::Y_AXIS));
    ret->params["output"] = std::to_string(ret->_out_depth);
    ret->params["pad-begin-x"] = std::to_string(ret->_padding.at(InferenceEngine::X_AXIS));
    ret->params["pad-begin-y"] = std::to_string(ret->_padding.at(InferenceEngine::Y_AXIS));
    ret->params["pad-end-x"] = std::to_string(ret->_pads_end.at(InferenceEngine::X_AXIS));
    ret->params["pad-end-y"] = std::to_string(ret->_pads_end.at(InferenceEngine::Y_AXIS));
    ret->params["stride-x"] = std::to_string(ret->_stride.at(InferenceEngine::X_AXIS));
    ret->params["stride-y"] = std::to_string(ret->_stride.at(InferenceEngine::Y_AXIS));

#ifdef NNLOG
    ALOGI(
        "Convolution  prms.groups = %d kernel.x= %d kernel.y= %d stride.x= %d stride.y= %d pad_start.x= %d pad_start.y= %d \
        pad_end.x= %d pad_end.y= %d ",
        prms.groups, ret->_kernel.at(InferenceEngine::X_AXIS),
        ret->_kernel.at(InferenceEngine::Y_AXIS), ret->_stride.at(InferenceEngine::X_AXIS),
        ret->_stride.at(InferenceEngine::Y_AXIS), ret->_padding.at(InferenceEngine::X_AXIS),
        ret->_padding.at(InferenceEngine::Y_AXIS), ret->_pads_end.at(InferenceEngine::X_AXIS),
        ret->_pads_end.at(InferenceEngine::Y_AXIS));
#endif

    if (prms.padType == "explicit") {
        Point2D in_size = {static_cast<int>(inDims[3]), static_cast<int>(inDims[2])};
        // Point2D out_size = (in_size + prms.pad_start + prms.pad_end - prms.kernel + prms.stride)
        // / prms.stride + 1;
        Point2D out_size =
            (in_size - prms.kernel + prms.stride + prms.pad_start + prms.pad_end) / prms.stride;

        addOutput(ret, {inDims[0], (size_t)prms.num_output_planes, (size_t)out_size.y,
                        (size_t)out_size.x});  // nchw
        // addOutput(ret, {inDims[0], (size_t) out_size.y, (size_t) out_size.x, (size_t)
        // prms.num_output_planes}); //nhwc
    } else {
        // Calculate output height and width for uneven padding
        size_t inputN = inDims[0];
        size_t IH = inDims[2];
        size_t IW = inDims[3];
        size_t KH = 0, KW = 0;
        float OH_temp = 1.f, OW_temp = 1.f;

        if (ret->_dilation[InferenceEngine::Y_AXIS])
            KH = (ret->_kernel[InferenceEngine::Y_AXIS] - 1) *
                     ret->_dilation[InferenceEngine::Y_AXIS] +
                 1;
        else
            KH = ret->_kernel[InferenceEngine::Y_AXIS];
        if (ret->_dilation[InferenceEngine::X_AXIS])
            KW = (ret->_kernel[InferenceEngine::X_AXIS] - 1) *
                     ret->_dilation[InferenceEngine::X_AXIS] +
                 1;
        else
            KW = ret->_kernel[InferenceEngine::X_AXIS];

        size_t SH = ret->_stride[InferenceEngine::Y_AXIS];
        size_t SW = ret->_stride[InferenceEngine::X_AXIS];
        size_t OC = ret->_out_depth;

        if (prms.padType == "valid") {
            OH_temp = std::ceil((IH - KH + 1.f) / SH);
            OW_temp = std::ceil((IW - KW + 1.f) / SW);
        } else if (prms.padType == "same_upper") {
            OH_temp = std::ceil(1.f * IH / SH);
            OW_temp = std::ceil(1.f * IW / SW);
        } else if (prms.padType == "same_lower") {
            OH_temp = std::floor(1.f * IH / SH);
            OW_temp = std::floor(1.f * IW / SW);
        }

        size_t OH = static_cast<size_t>(OH_temp);
        size_t OW = static_cast<size_t>(OW_temp);
        addOutput(ret, {inputN, OC, OH, OW});
    }

    return output(ret);
}

bool validate(const Operation& operation, const Model& model){
    int oper_size = operation.inputs.size();
    const auto& input0 = model.operands[operation.inputs[OP_INPUT_IDX_CONV]];
    const auto& input1 = model.operands[operation.inputs[OP_FILTER_IDX_CONV]];
    const auto& input2 = model.operands[operation.inputs[OP_BIAS_IDX_CONV]];

    VLOG(L1, "Validating CONV2D params");
    // filter in == channel
    // Check Input/Filter  Operand type

    if (input0.type != OperandType::TENSOR_FLOAT32 ||
        input1.type != OperandType::TENSOR_FLOAT32 ||
        input2.type != OperandType::TENSOR_FLOAT32) {
        VLOG(L1, "NNERR: input0/input1/input2 invalid operand types");
        return false;
    }

    if (input0.lifetime == input1.lifetime) {
        VLOG(L1,
                "NNERR: Filter (index %d) as model_input (index %d) not "
                "supported,aborting!!",
                operation.inputs[OP_FILTER_IDX_CONV], operation.inputs[OP_INPUT_IDX_CONV]);
        return false;
    }

    // Check Input Dimension size
    if (input0.dimensions.size() != NHWC_DIM_NUM ||
        input1.dimensions.size() != NHWC_DIM_NUM) {
        VLOG(L1,
                "NNERR: input-0 dim-size %d  or input1 dim-size %d "
                "invalid,aborting!!",
                input0.dimensions.size(), input1.dimensions.size());
        return false;
    }

    // Check Channel parameter for Input and filter/kernel
    if (input0.dimensions[NHWC_CH_IDX] != input1.dimensions[NHWC_CH_IDX]) {
        VLOG(L1,
                "NNERR: input-0 ch-size %d  and input-1 ch-size %d not "
                "equal,aborting!!",
                input0.dimensions.size(), input1.dimensions.size());
        return false;
    }

    if (input1.dimensions[NHWC_HT_IDX] != input1.dimensions[NHWC_WD_IDX]) {
        VLOG(L1, "NNERR: non-square Filter size(H:%d,W:%d) not supported,warning!!",
                input1.dimensions[NHWC_HT_IDX], input1.dimensions[NHWC_WD_IDX]);
        return false;
    }

    // Check all other Input operand types for implicit/explicit Padding

    if (oper_size == IMPL_PAD_PARAMS_CONV) {
        const auto& input3 = model.operands[operation.inputs[OP_PADSCHEME_IDX_CONV]];
        const auto& input4 = model.operands[operation.inputs[OP_STRD_WD_IDX_IMPL_CONV]];
        const auto& input5 = model.operands[operation.inputs[OP_STRD_HT_IDX_IMPL_CONV]];
        const auto& input6 = model.operands[operation.inputs[OP_ACTV_FUNC_IDX_IMPL_CONV]];

        if (input3.type != OperandType::INT32 || input4.type != OperandType::INT32 ||
            input5.type != OperandType::INT32 || input6.type != OperandType::INT32) {
            VLOG(L1, "NNERR: inputs 3-6 invalid operand types");
            return false;
        }
    } else if (oper_size == EXPL_PAD_PARAMS_CONV) {
        const auto& input3 = model.operands[operation.inputs[OP_PADL_IDX_CONV]];
        const auto& input4 = model.operands[operation.inputs[OP_PADR_IDX_CONV]];
        const auto& input5 = model.operands[operation.inputs[OP_PADH_IDX_CONV]];
        const auto& input6 = model.operands[operation.inputs[OP_PADW_IDX_CONV]];
        const auto& input7 = model.operands[operation.inputs[OP_STRD_WD_IDX_EXPL_CONV]];
        const auto& input8 = model.operands[operation.inputs[OP_STRD_HT_IDX_EXPL_CONV]];
        const auto& input9 = model.operands[operation.inputs[OP_ACTV_FUNC_IDX_EXPL_CONV]];

        if (input3.type != OperandType::INT32 || input4.type != OperandType::INT32 ||
            input5.type != OperandType::INT32 || input6.type != OperandType::INT32 ||
            input7.type != OperandType::INT32 || input8.type != OperandType::INT32 ||
            input9.type != OperandType::INT32) {
            VLOG(L1, "NNERR:inputs 3-9 invalid operand types");
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

bool initialize(const char* device, const Operation& operation, const Model model){
    VLOG(L1, "Initializing OperationType::CONV_2D");
    if (strcmp(device, "CPU") == 0){
        VLOG(L1, "Initializing OperationType::CONV_2D, inside cpu");
        dumpOperationParam(operation);
        sp<CpuPreparedModel> PreparedModelObj;
        uint32_t mPadreq;

        auto input = PreparedModelObj->getPort(operation.inputs[OP_INPUT_IDX_CONV]);
        auto filter = PreparedModelObj->GetConstOperandAsTensor(operation.inputs[OP_FILTER_IDX_CONV],
                                            OP_FILTER_IDX_CONV);  // OIHW
        auto bias = PreparedModelObj->GetConstOperandAsTensor(operation.inputs[OP_BIAS_IDX_CONV], OP_BIAS_IDX_CONV);
        if (bias == nullptr) {
            VLOG(L1, "NNERR:bias blob is NULL");
            return false;
        }

        if (operation.outputs.size() > 1) {
            VLOG(L1, "NNERR:More than one output for Conv2d,Aborting!!");
            return false;
        }

        const auto inputDims = input->getTensorDesc().getDims();
        const auto filterDims = filter->getTensorDesc().getDims();

        ConvolutionParams prms;

        int in_channels = (int)inputDims[1];
        int input_height = (int)inputDims[2];
        int input_width = (int)inputDims[3];

        int filter_in = (int)filterDims[1];
        int filter_out = (int)filterDims[0];
        int filter_height = (int)filterDims[2];
        int filter_width = (int)filterDims[3];

        int32_t fusion_index = -1;

        if (operation.inputs.size() == EXPL_PAD_PARAMS_CONV) {
            VLOG(L1, "Explicit padding requested");
            mPadreq = EXPL_PAD;
            prms.padType = "explicit";
            prms.pad_start.x = PreparedModelObj->ParseOperationInput<int32_t>(model, operation, OP_PADL_IDX_CONV);
            prms.pad_start.y = PreparedModelObj->ParseOperationInput<int32_t>(model, operation, OP_PADH_IDX_CONV);
            CHECK_OPERAND_2D(prms.pad_start, OP_PADL_IDX_CONV, OP_PADH_IDX_CONV);
            prms.pad_end.x = PreparedModelObj->ParseOperationInput<int32_t>(model, operation, OP_PADR_IDX_CONV);
            prms.pad_end.y = PreparedModelObj->ParseOperationInput<int32_t>(model, operation, OP_PADW_IDX_CONV);
            CHECK_OPERAND_2D(prms.pad_end, OP_PADR_IDX_CONV, OP_PADW_IDX_CONV);
            prms.stride.x = PreparedModelObj->ParseOperationInput<int32_t>(model, operation, OP_STRD_WD_IDX_EXPL_CONV);
            prms.stride.y = PreparedModelObj->ParseOperationInput<int32_t>(model, operation, OP_STRD_HT_IDX_EXPL_CONV);
            CHECK_OPERAND_2D(prms.stride, OP_STRD_WD_IDX_EXPL_CONV, OP_STRD_HT_IDX_EXPL_CONV);
            prms.kernel = {filter_width, filter_height};
            prms.num_output_planes = filter_out;  // depth out
            fusion_index = OP_ACTV_FUNC_IDX_EXPL_CONV;
        } else if (operation.inputs.size() == IMPL_PAD_PARAMS_CONV) {  // PAD SAME
            VLOG(L1, "Implicit padding requested");
            mPadreq = IMPL_PAD;
            const auto pad_type = PreparedModelObj->ParseOperationInput<int32_t>(model, operation, 3);  // padding_implicit
            int stride_width = PreparedModelObj->ParseOperationInput<int32_t>(model, operation, 4);
            int stride_height = PreparedModelObj->ParseOperationInput<int32_t>(model, operation, 5);
            int padding_left, padding_right;
            int padding_top, padding_bottom;
            if (pad_type == kPaddingSame) {
                calculateExplicitPadding(input_width, stride_width, filter_width,
                                        pad_type /*padding_implicit*/, &padding_left, &padding_right);
                calculateExplicitPadding(input_height, stride_height, filter_height,
                                        pad_type /*padding_implicit*/, &padding_top, &padding_bottom);

                prms.pad_start = {padding_left, padding_top};
                prms.pad_end = {padding_right, padding_bottom};
                prms.padType = "same_upper";

            } else if (pad_type == kPaddingValid) {
                /**
                 * VALID padding.
                 * No padding. When the input size is not evenly divisible by
                 * the filter size, the input at the end that could not fill
                 * the whole filter tile will simply be ignored.
                 */
                prms.pad_start = {0, 0};
                prms.pad_end = {0, 0};
                prms.padType = "valid";
            }
            prms.stride = {stride_width, stride_height};
            prms.kernel = {filter_width, filter_height};
            prms.num_output_planes = filter_out;  // depth out
            fusion_index = OP_ACTV_FUNC_IDX_IMPL_CONV;
        }

        if (bias && bias->size() != prms.num_output_planes) {
            VLOG(L1, "NNERR:biases size (%d)mismatch output planes (%d),warning", bias->size(),
                prms.num_output_planes);
            // return false;
            // nnAssert(false);
        }

        // input_size (validate)
        if (filter_in != in_channels) {
            VLOG(L1, "NNERR:filter depth_in size (%d) mismatch input depth (%d),warning!!", filter_in,
                in_channels);
            // return false;
            // nnAssert(false);
        }

        prms.weights = static_cast<IRBlob::Ptr>(
            filter);  // layout [filter_in, filter_out, filter_height, filter_width]
        const auto weightsDims = prms.weights->getTensorDesc().getDims();

        prms.biases = static_cast<IRBlob::Ptr>(bias);
        auto out = Convolution(input, prms);
        
        ConvolutionParamsToGenConvParams(prms, genConvPrms, filter, bias);
        nodeName = out->getName();
        inputName = input->getName();
        //PreparedModelObj->mCreateNgraph->addConvolution(out->getName(), input->getName(), genConvPrms);

        if (fusion_index < 0) {
            VLOG(L1, "invalid fusion index");
            nnAssert(false);
        }
        auto acv_func = PreparedModelObj->ParseOperationInput<int32_t>(model, operation, fusion_index);
        if (acv_func < 0) {
            VLOG(L1, "Invalid Activation function passed,aborting!!");
            return false;
        }
        // now here the out is next layer's input , and next layer is an activation
        // layer..relu/sigmoid etc...
        convDataPtr = handleFusion(out, acv_func);

        VLOG(L1, "----------------------------------------------");
        VLOGDIMS(L1, inputDims, "inputs dims");
        VLOGDIMS(L1, filterDims, "filter dims");
        VLOGDIMS(L1, weightsDims, "weights dims");
        VLOG(L1, "----------------------------------------------");

        return true;
    
    } else if (strcmp(device, "GNA") == 0){
        VLOG(L1, "Initializing OperationType::CONV_2D, inside gna");
        return false;
    } else {
        VLOG(L1, "Initializing OperationType::CONV_2D, inside default");
        return false;
    }
}

OutputPort updateDataPtr() {
    return convDataPtr;
}
std::string getNodeName(){
    return nodeName;
}
std::string getInputName(){
    return inputName;
}
GenConvParams getGenConvPrms(){
    return genConvPrms;
}

}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android