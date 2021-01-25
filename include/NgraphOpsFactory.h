#ifndef __OPERATIONS_FACTORY_H
#define __OPERATIONS_FACTORY_H

#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>

#include <ops/BaseOp.h>
#include <ops/Add.h>

#include "NgraphNwManager.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class NgraphOpsFactory {
private:
    static std::map<OperationType, std::shared_ptr<OperationsBase>> sOperationsMap;
	
public:
	static std::shared_ptr<OperationsBase> createNgraphOp(OperationType opType,
															NnapiModelInfo* modelInfo,
															NgraphNetworkCreator* nwCreator) {
		std::shared_ptr<OperationsBase> nodePtr = nullptr;

		if (sOperationsMap.find(opType) == sOperationsMap.end()) {
			switch (opType) {
				case OperationType::ADD:
					nodePtr = std::make_shared<Add>(modelInfo, nwCreator);
					break;
				default:
					ALOGE("Operation of type: %d not supported", opType);
					break;
			}
		}

		sOperationsMap[opType] = nodePtr;
		return nodePtr;
	}

	static bool isOperationSupported(Operation op, NnapiModelInfo* modelInfo) {
		ALOGI("%s", __func__);
		if (sOperationsMap.find(op.type) == sOperationsMap.end()) {
			switch (op.type) {
				case OperationType::ADD:
					if (!Add::validate(op, modelInfo))
						return false;
					break;
				default:
					VLOG(L1, "Failed to validate operation: ", op.type);
					return false;
			}
		}

		return true;
	}
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif