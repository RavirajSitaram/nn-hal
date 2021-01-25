#pragma once

#include <Driver.h>
#include <android/log.h>
#include <log/log.h>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/node.hpp>

#include "ModelManager.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class NgraphNetworkCreator;


class OperationsBase {
protected:
    NnapiModelInfo* mModelInfo;
    NgraphNetworkCreator* mNwCreator;
    enum ConversionType { NHWC_NCHW, NCHW_NHWC };
    
    std::shared_ptr<ngraph::Node> transpose(ConversionType type,
                                            std::shared_ptr<ngraph::Node> input);

public:
    OperationsBase(NnapiModelInfo* model,
                    NgraphNetworkCreator* nwCreator) {
		mModelInfo = model;
        mNwCreator = nwCreator;
	}

    virtual bool createNode(const Operation& op) = 0;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android