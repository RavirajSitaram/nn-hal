#pragma once
#ifndef __FULLY_CONNECTED_H
#define __FULLY_CONNECTED_H
#include <BaseOp.h>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

// Add is supposed to create an Add Node based on the arguments/parameters.
class FullyConnected : public OperationsBase {
public:
    FullyConnected(NnapiModelInfo* model, NgraphNetworkCreator* nwCreator): 
							OperationsBase(model, nwCreator) {
	}

    //std::shared_ptr<ngraph::Node> createOp(Operation& op, uint32_t index);

    static bool validate(const Operation& op, NnapiModelInfo* modelInfo) ;
    bool createNode(const Operation& operation) override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif