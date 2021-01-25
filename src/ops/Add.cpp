#include <Add.h>
#include "Utils.h"
#include "NgraphNwManager.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using FusedActivationFunc = V1_0::FusedActivationFunc;

bool Add::validate(const Operation& op, NnapiModelInfo* modelInfo) { 
	const auto& input0 = modelInfo->getOperand(op.inputs[0]);
    const auto& input1 = modelInfo->getOperand(op.inputs[1]);

    if (input0.type != input1.type) {
        VLOG(L1, "NNERR:input0 and input1 type not equal,aborting!!");
        return false;
    }

    const auto& output = modelInfo->getOperand(op.outputs[0]);
    if (output.type != input0.type) {
        VLOG(L1, "NNERR: output type not equalt to input0 type ,aborting!!");
        return false;
    }

    return true;
}

// Make it std::unique pointer..
bool Add::createNode(const Operation& nnApiOp) {

	std::shared_ptr<ngraph::Node> inNode0,inNode1,activation = nullptr;

	auto createNode = [&](Operation op, uint32_t index) -> std::shared_ptr<ngraph::Node> {		
		auto nnOperand = mModelInfo->getOperand(nnApiOp.inputs[index]);

		if (nnOperand.lifetime == OperandLifeTime::SUBGRAPH_INPUT) {
            // Add it to base class to make it simple
            std::string name = "Add"+ std::to_string(mNwCreator->getNumber());
			auto in = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape(toNgraphShape(nnOperand.dimensions)));
            in->set_friendly_name(name);
            mNwCreator->addInputNode(nnApiOp.inputs[index], in);
            mNwCreator->addLayerMetadata(nnApiOp.inputs[index], LayerInfo(name, false), true);
            return in;
		} else { // Assuming a const operand
			auto vals = mModelInfo->GetConstVecOperand<float>(op.inputs[index]);
			auto in = std::make_shared<ngraph::opset3::Constant>(ngraph::element::f32,
															ngraph::Shape(toNgraphShape(nnOperand.dimensions)),
															vals);
            return in;
		}
	};

	inNode0 = createNode(nnApiOp, 0);
	inNode1 = createNode(nnApiOp, 1);

	auto addOp = std::make_shared<ngraph::opset3::Add>(inNode0, inNode1,
                                                 ngraph::op::AutoBroadcastType::NUMPY);
    mNwCreator->appendNodeToMap(addOp);
	
	uint32_t activationFn = 0;
    std::string activationFnName;
	activationFn = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 2);

    if (activationFn) {
        //Special case .. Need to add generic template to handle activation functions
        switch(activationFn) {
            case (int32_t)FusedActivationFunc::RELU:
                activation = std::make_shared<ngraph::opset3::Relu>(addOp);
                activationFnName = "relu";
                break;
            case (int32_t)FusedActivationFunc::RELU6:
                activation = std::make_shared<ngraph::opset3::Clamp>(addOp, -1, 1);
                activationFnName = "relu6";
                break;
            case (int32_t)FusedActivationFunc::RELU1:
                activation = std::make_shared<ngraph::opset3::Clamp>(addOp, 0, 6);
                activationFnName = "relu1";
                break;
            default:
                break;
        }
        activationFnName += std::to_string(mNwCreator->getNumber());
        activation->set_friendly_name(activationFnName);
        mNwCreator->appendNodeToMap(activation);
    }

    // Check if the output is output node or intermediate node in the graph
    switch(mModelInfo->getOperandLifetime(nnApiOp.outputs[0])) {
        case OperandLifeTime::TEMPORARY_VARIABLE:
            mNwCreator->addIntermediateNode(nnApiOp.outputs[0], activation->outputs()[0]);
            break;
        case OperandLifeTime::SUBGRAPH_OUTPUT:
            mNwCreator->addResultNode(nnApiOp.outputs[0], activationFn?activation:addOp);
            mNwCreator->addLayerMetadata(nnApiOp.outputs[0], LayerInfo(activationFnName, false), false);
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