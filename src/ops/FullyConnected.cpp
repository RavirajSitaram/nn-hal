#include <FullyConnected.h>
#include "Utils.h"
#include "NgraphNwManager.h"

#define LOG_TAG "FCOperation"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using FusedActivationFunc = V1_0::FusedActivationFunc;

static
std::string toString(const std::vector<uint32_t>& range) {
    std::string os = "[";
    for (size_t i = 0; i < range.size(); ++i) {
        os += (i == 0 ? "" : ", ") + std::to_string(range[i]);
    }
    return os += "]";
}

bool FullyConnected::validate(const Operation& op, NnapiModelInfo* modelInfo) { 
	const auto& input0 = modelInfo->getOperand(op.inputs[0]);
    const auto& input1 = modelInfo->getOperand(op.inputs[1]);
	const auto& input2 = modelInfo->getOperand(op.inputs[2]);

    ALOGD("Dimensions(op 0): %s Dimensions(op 1): %s Dimensions(op 2): %s", toString(input0.dimensions).c_str(),
																			toString(input1.dimensions).c_str(),
																			toString(input2.dimensions).c_str());
	

	ALOGD("Type(op 0): %d Type(op 1): %d Type(op 2): %d", input0.type, input1.type, input2.type);

	if ((static_cast<int>(input0.type) == static_cast<int>(V1_0::OperandType::TENSOR_QUANT8_ASYMM)) ||
        (input0.type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED) ||
		(static_cast<int>(input0.type) == static_cast<int>(V1_2::OperandType::TENSOR_QUANT16_ASYMM)) ||
        (static_cast<int>(input0.type) == static_cast<int>(V1_2::OperandType::TENSOR_QUANT16_SYMM))) {
        ALOGE("Unsupported data type format.. TENSOR_QUANT8_ASYMM Or TENSOR_QUANT8_ASYMM_SIGNED");
        return false;
    }

    if (input0.type != input1.type) {
        ALOGE("NNERR:input0 and input1 type not equal,aborting!!");
        return false;
    }

    const auto& output = modelInfo->getOperand(op.outputs[0]);
    if (output.type != input0.type) {
        ALOGE("NNERR: output type not equalt to input0 type ,aborting!!");
        return false;
    }

	if (input0.dimensions.size() < 2) {
		ALOGE("Invalid input parameter dimensions!!!");
		return false;
	}

    ALOGI("FullyConnected::Validate succeeded");
    return true;
}

// Make it std::unique pointer..
bool FullyConnected::createNode(const Operation& nnApiOp) {
    ALOGD("%s", __func__);
    
	std::shared_ptr<ngraph::Node> input, weights, bias, addOp, activation;
	input = weights = bias = activation = addOp = nullptr;

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
    			auto vals = mModelInfo->GetConstVecOperand<float>(inputIndex);

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
    input = createNode(nnApiOp, 0);
    ALOGD("========> Creating Node 1");
    weights = createNode(nnApiOp, 1);

    auto getNode = [&](uint32_t index) {
        std::shared_ptr<ngraph::Node> node;
        uint32_t outIndex;
        std::tie(node, outIndex) = mNwCreator->getIntermediateNodeOutput(index);
        return node->outputs()[outIndex];
    };

	auto multiply = std::make_shared<ngraph::opset3::MatMul>((input!=nullptr)?input:getNode(nnApiOp.inputs[0]),
                                                        	(weights!=nullptr)?weights:getNode(nnApiOp.inputs[1]),
                                                        	false,
															true);
    mNwCreator->appendNodeToMap(multiply);

	// Check if the bias is available
	if (mModelInfo->getOperand(nnApiOp.inputs[2]).dimensions.size() == 0) {
		ALOGD("No bias for the operation !!!");
	} else {
		bias = createNode(nnApiOp, 2);
		addOp = std::make_shared<ngraph::opset3::Add>(multiply,
													(bias != nullptr)?bias:getNode(nnApiOp.inputs[2]),
													ngraph::op::AutoBroadcastType::NUMPY);
	}

	uint32_t activationFn = 0;
    std::string activationFnName;
	activationFn = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 3);

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
                nnAssert("UNKNOWN ACTIVATION FUNCTION !!!!!");
                break;
        }

		if (activation) {
			activationFnName += std::to_string(mNwCreator->getNumber());
        	activation->set_friendly_name(activationFnName);
        	mNwCreator->appendNodeToMap(activation);
		}
    }

    auto outputName = activationFn?activation->outputs()[0].get_node()->get_friendly_name():addOp->outputs()[0].get_node()->get_friendly_name();
    ALOGD("Output name: %s", outputName.c_str());

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