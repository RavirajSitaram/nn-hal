/*
 * INTEL CONFIDENTIAL
 * Copyright 2017 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */

#pragma once
#include <cassert>
#include "IRDocument.h"
#include "IRLayer.h"
#include "file_utils.h"
#include "ie_common.h"
#include "ie_layers_property.hpp"

//#define LOG_TAG "graphAPI"

#ifdef NNLOG
#include <android/log.h>
#include <log/log.h>
#endif

#define USE_NGRAPH

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

struct GenConvParams {
    int groups = 1;
    std::vector<float> weightsBuf;
    std::vector<size_t> weightsDims;
    std::vector<float> biasesBuf;
    std::vector<size_t> biasesDims;
    size_t weightsSize;
    std::vector<size_t> strides;
    std::vector<std::ptrdiff_t> pads_begin;
    std::vector<std::ptrdiff_t> pads_end;
    std::vector<size_t> dilations;
    const char *pad_type;
};

extern int layer_name_count;
extern InferenceEngine::Precision g_layer_precision;

inline OutputPort addOutput(const IRLayer &layer, const InferenceEngine::SizeVector &dims) {
    std::string d_name = layer->name;
    if (!layer->outData.empty()) {
        std::stringstream oss;
        oss << d_name << ":" << layer->outData.size();
        d_name = oss.str();
    }
    OutputPort data;
    if (dims.size() == 2) {
        std::cout << "addOutput dims size 2" << std::endl;
        InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::NC);
        data = std::make_shared<InferenceEngine::Data>(d_name, td);

    } else if (dims.size() == 4) {
        std::cout << "addOutput dims size " << dims.size() << std::endl;
        // InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::ANY);
        // InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::NCHW);
        InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::NHWC);
        data = std::make_shared<InferenceEngine::Data>(d_name, td);

    } else if (dims.size() == 3) {
#ifdef NNLOG
        ALOGI("addOutput data dims %d ", dims.size());
#endif
        InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::CHW);
        data = std::make_shared<InferenceEngine::Data>(d_name, td);
    } else {
        std::cout << "addOutput dims size " << dims.size() << std::endl;
        // InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::ANY);
        InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::C);
        data = std::make_shared<InferenceEngine::Data>(d_name, td);
    }

    layer->outData.push_back(data);
    data->getCreatorLayer() = layer;

#ifdef NNLOG
    std::vector<size_t> outdims = data->getTensorDesc().getDims();
    for (int i = 0; i < outdims.size(); i++) {
        ALOGI("addOutput data dims[%d] = %lu ", i, outdims[i]);
    }
#endif

    return data;
}

template <typename T>
void addAttr(IRLayer layer, const std::string &a_name, T val) {
    std::stringstream oss;
    oss << val;
    layer->params[a_name] = oss.str();
};
template <typename T, typename S>
std::shared_ptr<T> As(const std::shared_ptr<S> &src) {
    return /*std::dynamic_pointer_cast<T>(src)*/ std::static_pointer_cast<T>(src);
}  // aks

/*
 * @brief Creates a generic layer with one input and one output
 */

inline IRLayer Generic(const std::string &type) {
    std::string name = type + "-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prms;
    prms.precision = g_layer_precision;
    prms.name = name;
    prms.type = type;
    return std::make_shared<InferenceEngine::CNNLayer>(prms);
}

inline IRLayer Generic(const std::string &type, const OutputPort &src) {
    std::string name = type + "-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prms;
    prms.precision = g_layer_precision;
    prms.name = name;
    auto layer = std::make_shared<InferenceEngine::CNNLayer>(prms);
    layer->type = type;
    src >> layer;
    addOutput(layer, src->getTensorDesc().getDims());
    return layer;
}

inline OutputPort output(const IRLayer &src, int index = 0) { return src->outData[index]; }

inline IRLayer Generic(const std::string &type, const IRLayer &src) {
    return Generic(type, output(src));
}

inline IRLayer LayerOf(const OutputPort &src) { return src->getCreatorLayer().lock(); }

template <typename T, typename A>
std::string dumpVec(std::vector<T, A> const &vec) {
    if (vec.empty()) return "[]";
    std::stringstream oss;
    oss << "[" << vec[0];
    for (size_t i = 1; i < vec.size(); i++) oss << "," << vec[i];
    oss << "]";
    return oss.str();
}


static OutputPort ScaleShiftNode(const OutputPort &src, const IRBlob::Ptr &scale,
                                 const IRBlob::Ptr &bias) {
    std::cout << "ScaleShiftNode" << std::endl;
    std::string name = "ConstMul-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    prm.type = "ScaleShift";
    auto l = std::make_shared<InferenceEngine::ScaleShiftLayer>(prm);

    src >> l;
    l->_weights = scale;
    l->_broadcast = false;
    l->_biases = bias;
    l->blobs["biases"] = bias;
    return addOutput(l, src->getTensorDesc().getDims());
}

inline OutputPort AddTryConst(const OutputPort &src, const IRBlob::Ptr &biases) {
    auto fc = As<InferenceEngine::WeightableLayer>(LayerOf(src));
    if (fc) {
        // todo: check if biases was not lready being set
        std::cout << "AddTryConst" << std::endl;
#ifdef NNLOG
        ALOGI("AddTryConst for biases");
#endif

        fc->_biases = biases;
        fc->blobs["biases"] = biases;
        return src;  // it was fused with prev layer
    } else {
        // need to create an add with Const here using ScaleShift with no weights...
        // there are two options, scale shift with no weights, or cosnt with an Add
        return ScaleShiftNode(src, nullptr, biases);
    }
}

inline OutputPort operator+(const OutputPort &src, const IRBlob::Ptr &biases) {
    return AddTryConst(src, biases);
}

struct Point2D {
    int x, y;

    inline int size() const { return x * y; }
};
inline Point2D operator+(const Point2D &a, const Point2D &b) { return {a.x + b.x, a.y + b.y}; }
inline Point2D operator-(const Point2D &a, const Point2D &b) { return {a.x - b.x, a.y - b.y}; }
inline Point2D operator*(const Point2D &a, const Point2D &b) { return {a.x * b.x, a.y * b.y}; }
inline Point2D operator/(const Point2D &a, const Point2D &b) { return {a.x / b.x, a.y / b.y}; }
inline Point2D operator+(const Point2D &a, const int &rhs) { return {a.x + rhs, a.y + rhs}; }

struct ConvolutionParams {
    int groups = 1;
    Point2D kernel, stride = {1}, pad_start = {0}, pad_end = {0};
    int num_output_planes;
    IRBlob::Ptr weights;
    IRBlob::Ptr biases;
    std::string padType;
};

#ifdef USE_NGRAPH

#endif

inline size_t in_ch(const OutputPort &src) {
    auto dims = src->getTensorDesc().getDims();
    return dims.size() == 4 ? dims[1] : dims[2];
}

struct BatchNormParams {
    float epsilon;
    IRBlob::Ptr weights;
    IRBlob::Ptr bias;
};

inline IRLayer BatchNormalization(const OutputPort &src, BatchNormParams &prms) {
    auto inp = src;
    std::string name = "BatchNormalization-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto l = std::make_shared<InferenceEngine::BatchNormalizationLayer>(prm);
    l->type = "BatchNormalization";
    src >> l;
    l->epsilon = prms.epsilon;
    l->_weights = prms.weights;
    l->_biases = prms.bias;
    addOutput(l, inp->getTensorDesc().getDims());
    return l;
}



inline OutputPort Crop(const OutputPort &src, const std::vector<int> &axis,
                       const std::vector<int> &dim, const std::vector<int> &offset) {
    auto inp = src;
    std::string name = "Crop-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto l = std::make_shared<InferenceEngine::CropLayer>(prm);
    l->type = "Crop";
    src >> l;
    l->axis = axis;
    l->dim = dim;
    l->offset = offset;
    InferenceEngine::SizeVector sv(dim.begin(), dim.end());
    return addOutput(l, sv);
}

namespace SumLayer {
static IRLayer create(const OutputPort &src1, const OutputPort &src2) {
    std::string name = "Sum-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto sum = std::make_shared<InferenceEngine::EltwiseLayer>(prm);
    sum->type = "Eltwise";
    src1 >> sum;
    src2 >> sum;
    if (src1->getTensorDesc().getDims() != src2->getTensorDesc().getDims())
        THROW_IE_EXCEPTION << "input sizes for Element wise Sum do not match";
    addOutput(sum, src1->getTensorDesc().getDims());
    return sum;
}
};  // namespace SumLayer

namespace ScaleShift {

static OutputPort Diagnoal(const Vector &weights, const OutputPort &src) {
    std::string name = "ConstMul-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto l = std::make_shared<InferenceEngine::ScaleShiftLayer>(prm);
    l->type = "ConstMul";
    src >> l;
    addOutput(l, src->getTensorDesc().getDims());
    l->_weights = weights.data;
    if (weights.length == 1)
        l->_broadcast = 0;
    else if (weights.length == src->getTensorDesc().getDims()[1]) {
        l->_broadcast = 1;
    }

    return output(l);
}
static InferenceEngine::CNNLayer::Ptr create(OutputPort src, IRBlob::Ptr scale, IRBlob::Ptr bias) {
    std::string name = "ConstMul-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto l = std::make_shared<InferenceEngine::ScaleShiftLayer>(prm);
    l->type = "ScaleShift";
    src >> l;
    l->_weights = scale;
    l->_broadcast = false;
    addOutput(l, src->getTensorDesc().getDims());
    // AddConst(l, bias);
    return l;
}
};  // namespace ScaleShift

inline OutputPort operator*(const Vector &weights, const IRLayer &b) {
    return (ScaleShift::Diagnoal(weights, output(b)));
}

inline OutputPort operator*(const Vector &weights, const OutputPort &op) {
    return (ScaleShift::Diagnoal(weights, op));
}


namespace SplitUtil {

static IRLayer create(int size, const OutputPort &src, int axis = 1) {
    std::string name = "Split-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto me = std::make_shared<InferenceEngine::SplitLayer>(prm);
    me->type = "Split";
    addAttr(me, "axis", axis);
    src >> me;
    auto out_dim = src->getTensorDesc().getDims();
    // axis = static_cast<int>(out_dim.size()) - axis - 1; // todo: we are all in reverse here :-(
    out_dim[axis] = out_dim[axis] / size;
    IR_ASSERT(out_dim[axis] * size == src->getTensorDesc().getDims()[axis]);

    for (int i = 0; i < size; i++) {
        addOutput(me, out_dim);
    }
    return me;
}
};  // namespace SplitUtil

inline std::vector<OutputPort> Split(const OutputPort &src, int splitElements, int axis = 1) {
    return SplitUtil::create(splitElements, src, axis)->outData;
}

inline std::vector<OutputPort> Split(const IRLayer &src, int splitElements, int axis = 1) {
    return Split(output(src), splitElements, axis);
}


inline OutputPort Gather(const std::vector<OutputPort> inputs, int axis = 1) {
    std::string name = "Gather-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto ret = std::make_shared<InferenceEngine::GenericLayer>(prm);
    ret->type = "Gather";
    addAttr(ret, "axis", axis);
    inputs[0] >> ret;
    inputs[1] >> ret;
    auto outDim = inputs[0]->getTensorDesc().getDims();
    // axis = static_cast<int>(outDim.size()) - axis - 1; // todo: we are all in reverse here :-(
    outDim[0] = inputs[1]->getTensorDesc().getDims()[1];
    addOutput(ret, outDim);
    return output(ret);
}

inline OutputPort operator+(const OutputPort &a, const OutputPort &b) {
    return output(SumLayer::create(a, b));
}

inline OutputPort AddConst(IRDocument &doc, const OutputPort &src, const IRBlob::Ptr &biases) {
    // use const layer with elment wise add
    auto constNode = Generic("Const");
    doc.add(constNode);
    constNode->blobs["custom"] = biases;
    const auto constOut = addOutput(constNode, src->getTensorDesc().getDims());
    return src + constOut;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
