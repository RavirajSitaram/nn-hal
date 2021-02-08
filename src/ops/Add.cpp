#include <Add.h>
#include "Utils.h"
#include "NgraphNwManager.h"

#define LOG_TAG "AddOperation"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using FusedActivationFunc = V1_0::FusedActivationFunc;

bool Add::validate(const Operation& op, NnapiModelInfo* modelInfo) { 
	const auto& input0 = modelInfo->getOperand(op.inputs[0]);
    const auto& input1 = modelInfo->getOperand(op.inputs[1]);

    if (input0.type != input1.type) {
        ALOGE("NNERR:input0 and input1 type not equal,aborting!!");
        return false;
    }

    if ((input0.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) ||
		(static_cast<int>(input0.type) == static_cast<int>(V1_2::OperandType::TENSOR_QUANT16_ASYMM)) ||
        (static_cast<int>(input0.type) == static_cast<int>(V1_2::OperandType::TENSOR_QUANT16_SYMM))) {
        ALOGE("Unsupported data type format.. TENSOR_QUANT8_ASYMM Or TENSOR_QUANT8_ASYMM_SIGNED");
        return false;
    }

    const auto& output = modelInfo->getOperand(op.outputs[0]);
    if (output.type != input0.type) {
        ALOGE("NNERR: output type not equalt to input0 type ,aborting!!");
        return false;
    }

    ALOGD("type: %d type: %d", input0.type, input1.type);

    ALOGD("Add::Validate succeeded");
    return true;
}

// Make it std::unique pointer..
bool Add::createNode(const Operation& nnApiOp) {
    ALOGD("%s", __func__);
    
	std::shared_ptr<ngraph::Node> inNode0 = nullptr,inNode1 = nullptr,activation = nullptr;

	auto createNode = [&](Operation op, uint32_t index) -> std::shared_ptr<ngraph::Node> {
        auto inputIndex = op.inputs[index];
        ngraph::Shape inShape;
		auto nnOperand = mModelInfo->getOperand(inputIndex);
        
        ALOGD("Input index: %d type: %d", inputIndex, nnOperand.type);
        if (nnOperand.lifetime == OperandLifeTime::SUBGRAPH_INPUT) {
                std::string name = "Add-"+ std::to_string(mNwCreator->getNumber());
                ALOGD("Input is of type subgraph input %s  type=%d", name.c_str(), nnOperand.type);
                auto in = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, toNgraphShape(nnOperand.dimensions));
                in->set_friendly_name(name);

                ALOGD("Setting graph input layer name: %s", name.c_str());
                mNwCreator->addInputNode(inputIndex, in);

                ALOGD("Adding layer metadata");
                mNwCreator->addLayerMetadata(inputIndex, LayerInfo(name, false), true);

                ALOGD("Done ...........");
                return in;
        } else if ((nnOperand.lifetime == OperandLifeTime::CONSTANT_COPY) || (nnOperand.lifetime == OperandLifeTime::CONSTANT_REFERENCE)) {
                ALOGD("Input is of type : const copy / reference %d", nnOperand.dimensions.size());
                std::vector<float> vals;
                if (static_cast<int>(nnOperand.type) == static_cast<int>(OperandType::TENSOR_QUANT8_ASYMM)) {
					vals = mModelInfo->GetConstVecOperand<float, uint8_t>(inputIndex);
				} else {
					vals = mModelInfo->GetConstVecOperand<float, float>(inputIndex);
				}

                for (auto val : vals) {
                    ALOGD("Dumping vals: %f", val);
                }

                //auto vals = std::vector<float>(1);
                auto in = std::make_shared<ngraph::opset3::Constant>(ngraph::element::f32,
                                                                     ngraph::Shape(toNgraphShape(nnOperand.dimensions)),
                                                                     vals);
                return in;
        } else {
            ALOGD("Input is of type temporary variable or unsupported");
            return nullptr;
        }
	};

    ALOGD("========> Creating Node 0");
    inNode0 = createNode(nnApiOp, 0);

    if (mModelInfo->getOperand(nnApiOp.inputs[0]).dimensions.size() > 2) {
        RunTimeOperandInfo& runtimeOp = mModelInfo->getRuntimeOperand(nnApiOp.inputs[0]);
		runtimeOp.ignoreLayout = true;
    }

    ALOGD("========> Creating Node 1");
    inNode1 = createNode(nnApiOp, 1);

    auto getNode = [&](uint32_t index) {
        std::shared_ptr<ngraph::Node> node;
        uint32_t outIndex;
        std::tie(node, outIndex) = mNwCreator->getIntermediateNodeOutput(index);
        return node->outputs()[outIndex];
    };

	auto addOp = std::make_shared<ngraph::opset3::Add>((inNode0!=nullptr)?inNode0:getNode(nnApiOp.inputs[0]),
                                                        (inNode1!=nullptr)?inNode1:getNode(nnApiOp.inputs[1]),
                                                        ngraph::op::AutoBroadcastType::NUMPY);
    mNwCreator->appendNodeToMap(addOp);
	
	uint32_t activationFn = 0;
    std::string activationFnName;
	activationFn = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 2);

    if (activationFn) {
        //Special case .. Need to add generic template to handle activation functions
        switch(activationFn) {
            case (int32_t)FusedActivationFunc::RELU:
                ALOGD("Adding relu");
                activation = std::make_shared<ngraph::opset3::Relu>(addOp);
                activationFnName = "relu";
                break;
            case (int32_t)FusedActivationFunc::RELU6:
                ALOGD("Adding relu6");
                activation = std::make_shared<ngraph::opset3::Clamp>(addOp, -1, 1);
                activationFnName = "relu6";
                break;
            case (int32_t)FusedActivationFunc::RELU1:
                ALOGD("Adding relu1");
                activation = std::make_shared<ngraph::opset3::Clamp>(addOp, 0, 6);
                activationFnName = "relu1";
                break;
            default:
                ALOGD("UNKNOWN ACTIVATION FUNCTION !!!!!");
                break;
        }
        activationFnName += std::to_string(mNwCreator->getNumber());
        activation->set_friendly_name(activationFnName);
        mNwCreator->appendNodeToMap(activation);
    }

    auto outputName = activationFn?activation->outputs()[0].get_node()->get_friendly_name():addOp->outputs()[0].get_node()->get_friendly_name();
    ALOGD("Output data type: %d", mModelInfo->getOperand(nnApiOp.outputs[0]).type);
    // Check if the output is output node or intermediate node in the graph
    switch(mModelInfo->getOperandLifetime(nnApiOp.outputs[0])) {
        case OperandLifeTime::TEMPORARY_VARIABLE:
            ALOGD("Output lifetime TEMPORARY_VARIABLE");
            if (activationFn) {
                mNwCreator->addIntermediateNode(nnApiOp.outputs[0], activation->outputs()[0]);
                mNwCreator->mapIntermediateNodeOutput(nnApiOp.outputs[0], activation, 0);
            } else {
                mNwCreator->addIntermediateNode(nnApiOp.outputs[0], addOp->outputs()[0]);
                mNwCreator->mapIntermediateNodeOutput(nnApiOp.outputs[0], addOp, 0);
            }
            break;
        case OperandLifeTime::SUBGRAPH_OUTPUT:
            ALOGD("Output lifetime SUBGRAPH OUTPUT");
            mNwCreator->addResultNode(nnApiOp.outputs[0], activationFn?activation:addOp);
            mNwCreator->addLayerMetadata(nnApiOp.outputs[0], LayerInfo(outputName, false), false);
            break;
        default:
            ALOGE("Unsupported lifetime for output node: %d", mModelInfo->getOperandLifetime(nnApiOp.outputs[0]));
            break;
    }

	return true;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android