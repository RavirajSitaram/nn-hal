#include "DevicePlugin.h"
#include "ie_common.h"

#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <ie_blob.h>


namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

template<IntelDeviceType T>
bool DevicePlugin<T>::loadNetwork() {
    ALOGE("%s", __func__);

    InferenceEngine::Core ie;
    std::map<std::string, std::string> config;

    mExecutableNw = ie.LoadNetwork(*mNetwork, "CPU");
    mInferRequest = mExecutableNw.CreateInferRequest();
}

// Need to be called before loadnetwork.. But not sure whether need to be called for 
// all the inputs in case multiple input / output
template<IntelDeviceType T>
void DevicePlugin<T>::prepareInput(InferenceEngine::Precision precision, InferenceEngine::Layout layout) {
    ALOGE("%s", __func__);      

    auto inputInfoItem = *mInputInfo.begin();
    inputInfoItem.second->setPrecision(precision);
    inputInfoItem.second->setLayout(layout);
}

template<IntelDeviceType T>
void DevicePlugin<T>::prepareOutput(InferenceEngine::Precision precision, InferenceEngine::Layout layout) {
    InferenceEngine:: DataPtr& output = mOutputInfo.begin()->second;
    output->setPrecision(precision);
    output->setLayout(layout);
}

template<IntelDeviceType T>
void DevicePlugin<T>::setBlob(const std::string& inName, const InferenceEngine::Blob::Ptr& inputBlob) {
    ALOGI("setBlob input or output blob name : %s", inName.c_str());
    mInferRequest.SetBlob(inName, inputBlob);
}

template<IntelDeviceType T>
InferenceEngine::TBlob<float>::Ptr DevicePlugin<T>::getBlob(const std::string& outName) {
    InferenceEngine::Blob::Ptr outputBlob;
    outputBlob = mInferRequest.GetBlob(outName);
    return android::hardware::neuralnetworks::nnhal::As<InferenceEngine::TBlob<float>>(outputBlob);
    }

template<IntelDeviceType T>
void DevicePlugin<T>::infer() {
        ALOGI("Infer Network\n");
        mInferRequest.StartAsync();
        mInferRequest.Wait(10000);
        ALOGI("infer request completed");
}

template class DevicePlugin<IntelDeviceType::CPU>;
}
}
}
}