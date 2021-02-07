#ifndef __OPERATIONS_FACTORY_H
#define __OPERATIONS_FACTORY_H

#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>

#include <ops/BaseOp.h>
#include <ops/Add.h>
#include <ops/FullyConnected.h>

#include "NgraphNwManager.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class NgraphOpsFactory {
private:
    static std::map<OperationType, std::shared_ptr<OperationsBase>> sOperationsMap;
	
public:
	static std::unique_ptr<OperationsBase> createNgraphOp(OperationType opType,
															NnapiModelInfo* modelInfo,
															NgraphNetworkCreator* nwCreator) {
		ALOGD("%s", __func__);
		std::unique_ptr<OperationsBase> nodePtr = nullptr;

		// if (sOperationsMap.find(opType) == sOperationsMap.end()) {
		// 	switch (opType) {
		// 		case OperationType::ADD:
		// 			nodePtr = std::make_shared<Add>(modelInfo, nwCreator);
		// 			sOperationsMap[opType] = nodePtr;
		// 			break;
		// 		default:
		// 			ALOGE("Operation of type: %d not supported", opType);
		// 			break;
		// 	}
		// } else {
		// 	ALOGD("Add operation is already present.. Returning the same op")
		// 	nodePtr = sOperationsMap[opType];
		// }
		//if (sOperationsMap.find(opType) == sOperationsMap.end()) {
			switch (opType) {
				case OperationType::ADD:
					nodePtr = std::make_unique<Add>(modelInfo, nwCreator);
					//sOperationsMap[opType] = nodePtr;
					break;
				case OperationType::FULLY_CONNECTED:
					nodePtr = std::make_unique<FullyConnected>(modelInfo, nwCreator);
					break;
				default:
					ALOGE("Operation of type: %d not supported", opType);
					break;
			}
		//} else {
		//	ALOGD("Add operation is already present.. Returning the same op");
			//nodePtr = sOperationsMap[opType];
		//}

		return nodePtr;
	}

	static bool isOperationSupported(Operation op, NnapiModelInfo* modelInfo) {
		ALOGD("%s", __func__);
		if (sOperationsMap.find(op.type) == sOperationsMap.end()) {
			switch (op.type) {
				case OperationType::ADD:
					if (!Add::validate(op, modelInfo))
						return false;
					break;
				case OperationType::FULLY_CONNECTED:
					if (!FullyConnected::validate(op, modelInfo))
						return false;
					break;
				default:
					ALOGE("Failed to validate operation: %d", op.type);
					return false;
			}
		}

		ALOGD("%s succeeded", __func__);
		return true;
	}
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif