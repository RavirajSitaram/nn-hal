#define LOG_TAG "GnaPreparedModel"

#include "GnaPreparedModel.h"
#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <fstream>
#include <thread>
#include "ValidateHal.h"
#include "Utils.h"

using namespace android::nn;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

void GnaPreparedModel::deinitialize() {
    VLOG(L1, "deinitialize");
    delete enginePtr;
    enginePtr = nullptr;
    VLOG(L1, "free engine");
}

bool GnaPreparedModel::initialize() {
    VLOG(L1, "initialize");
    bool success = false;

     // Check operation supoorted or not, user may not call getOpertionSupported()
    for (const auto& operation : mModel.operations) {
        success = isOperationSupported(operation, mModel);
        dumpOperationSupport(operation, success);
        if (!success) {
            VLOG(L1, "get unsupported operation in initialize()");
            return false;
        }
    }

    success = setRunTimePoolInfosFromHidlMemories(&mPoolInfos, mModel.pools);
    if (!success) {
        VLOG(L1, "setRunTimePoolInfosFromHidlMemories failed.");
        return false;
    }

    success = initializeRunTimeOperandInfo();
    if (!success) {
        VLOG(L1, "initializeRunTimeOperandInfo failed.");
        return false;
    }

    for (const auto& operation : mModel.operations) {
        VLOG(L1, "get operation %d ready to add", operation.type);
        dumpOperation(operation);
        switch (operation.type) {

            case OperationType::CONCATENATION:{
                success = concat::initialize(mTargetDevice, mCreateNgraphPtr);
            } break;
            case OperationType::FULLY_CONNECTED:{
                success = fullyconnected::initialize(mTargetDevice, mCreateNgraphPtr);
            } break;
            default:
                VLOG(L1, "unsupported operation %d", operation.type);
                return false;
        }
        if (success == false) {
            VLOG(L1, "failed to convert operation %d", operation.type);
            return false;
        }
    }

    initializeInput();
    success = finalizeOutput();

    initializeInput();
    success = finalizeOutput();

    InferenceEngine::CNNNetwork ngraph_net;
    ngraph_net = mCreateNgraph->generate(std::string("/data/vendor/neuralnetworks/ngraph_ir.xml"),
                                         std::string("/data/vendor/neuralnetworks/ngraph_ir.bin"));
    if (success == false) return success;

    mNet.buildNetwork();
    VLOG(L1, "initialize ExecuteNetwork for device %s", mTargetDevice.c_str());
    enginePtr = new ExecuteNetwork(ngraph_net, mNet, mTargetDevice);
    enginePtr->prepareInput();
    enginePtr->loadNetwork(ngraph_net);

    return true;

}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
