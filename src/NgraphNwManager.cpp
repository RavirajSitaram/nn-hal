#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>

#include "NgraphNwManager.h"
#include "NgraphOpsFactory.h"

#define LOG_TAG "NGraphNwManager"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

std::map<OperationType, std::shared_ptr<OperationsBase>> NgraphOpsFactory::sOperationsMap;

bool NgraphNetworkCreator::init() {
	ALOGI("%s", __func__);
	return true;
}

InferenceEngine::CNNNetwork* NgraphNetworkCreator::generateIRGraph() {
	ALOGI("%s", __func__);
	
    auto operations = mModelInfo->getOperations();
	for (const auto& op: operations) {
		auto nGraphOp = NgraphOpsFactory::createNgraphOp(op.type, mModelInfo, this);
		if (!nGraphOp->createNode(op)) {
            ALOGE("Failed to createNode for op type:%d", op.type);
            return nullptr;
        }
	}

    ngraph::OutputVector opVec;
    for (auto iter = mNgraphResultNodes.begin(); iter != mNgraphResultNodes.end(); iter++) {
        opVec.push_back(iter->second);
    }

    ngraph::ParameterVector inVec;
    for (auto iter = mNgraphInputNodes.begin(); iter != mNgraphInputNodes.end(); iter++) {
        inVec.push_back(iter->second);
    }

    auto net = new InferenceEngine::CNNNetwork(std::make_shared<ngraph::Function>(opVec, inVec));
    return net;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
