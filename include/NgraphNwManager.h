#ifndef __NGRAPH_NW_MANAGER_H
#define __NGRAPH_NW_MANAGER_H

#include "ModelManager.h"
#include "BaseOp.h"
#include <inference_engine.hpp>
#include <ngraph/node.hpp>
#include <ie_cnn_network.h>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using Parameter = ngraph::opset3::Parameter;

class NgraphNetworkCreator {
private:
    NnapiModelInfo* mModelInfo;
    std::vector<std::shared_ptr<ngraph::Node>> mNgraphNodes;
    std::map<uint32_t, std::shared_ptr<ngraph::Node>> mNgraphResultNodes;
	std::map<uint32_t, ngraph::Output<ngraph::Node>*> mIntermediateOutputs;
    std::map<uint32_t, std::shared_ptr<ngraph::opset3::Parameter>> mNgraphInputNodes;
    std::map<uint32_t, LayerInfo> mInLayerMap;
    std::map<uint32_t, LayerInfo> mOutLayerMap;
    std::map<uint32_t, std::tuple<std::shared_ptr<ngraph::Node>, uint32_t>> mIntermediates;

public:
    NgraphNetworkCreator(NnapiModelInfo* model): mModelInfo(model) {
	}

	bool init();
    InferenceEngine::CNNNetwork* generateIRGraph();

    int getNumber() {
        static int count = 0;
        return count++;
    }

    bool initializeModel() {
		return false;
	}

    // bool getMemoryLayerName(uint32_t index, std::string& name) {
    //     if (mMemoryLayerNames.find(index) != mMemoryLayerNames.end()) {
    //         name = mMemoryLayerNames[index];
    //         return true;
    //     }

    //     return false;
    // }

    const std::map<uint32_t, LayerInfo>& getInputLayerMap() {
        return mInLayerMap;
    }

    const std::map<uint32_t, LayerInfo>& getOutputLayerMap() {
        return mOutLayerMap;
    }

    void mapIntermediateNodeOutput(uint32_t index, std::shared_ptr<ngraph::Node> node, uint32_t opIndex) {
        if (mIntermediates.find(index) != mIntermediates.end()) {
            ALOGE("Overwriting intermediate node index");
        }

        mIntermediates[index] = std::make_tuple(node, opIndex);
    }

    std::tuple<std::shared_ptr<ngraph::Node>, uint32_t> getIntermediateNodeOutput(uint32_t index) {
        ALOGD("%s", __func__);
        return mIntermediates[index];
    }

    // Move these to protected and add operation base friend class???
    void addIntermediateNode(uint32_t index, ngraph::Output<ngraph::Node>& outputNode) {
        ALOGD("%s : index: %d", __func__, index);
        if (mIntermediateOutputs.find(index) != mIntermediateOutputs.end())
            ALOGE("%s Overwriting previous output node with new node at index: %d", __func__, index);

        mIntermediateOutputs[index] = &outputNode;
    }

    void addResultNode(uint32_t index, std::shared_ptr<ngraph::Node> node) {
        ALOGD("%s : %d", __func__, index);
        if (mNgraphResultNodes.find(index) != mNgraphResultNodes.end())
            ALOGE("%s Overwriting previous result node with new node at index: %d", __func__, index);

        mNgraphResultNodes[index] = node;
    }

    void addInputNode(uint32_t index, std::shared_ptr<ngraph::opset3::Parameter> node) {
        ALOGD("%s : index: %d", __func__, index);
        if (mNgraphInputNodes.find(index) != mNgraphInputNodes.end())
            ALOGE("%s Overwriting previous result node with new node at index: %d", __func__, index);

        mNgraphInputNodes[index] = node;
    }

    void appendNodeToMap(std::shared_ptr<ngraph::Node> node) {
        ALOGD("%s", __func__);
        mNgraphNodes.push_back(node);
    }

    void addLayerMetadata(uint32_t index,const LayerInfo& l, bool input) {
        ALOGD("%s index:%d type:%s", __func__, index, input?"input":"output");
        if (input)
            mInLayerMap[index] = l;
        else
            mOutLayerMap[index] = l;
    }

    ngraph::Output<ngraph::Node>* getIntermediateNodeAt(uint32_t index) {
        ALOGD("%s index:%d", __func__, index);
        if (mIntermediateOutputs.find(index) == mIntermediateOutputs.end())
            nnAssert("Failed to find the index in Intermediate node map");
        auto ptr = mIntermediateOutputs[index];
        if (!ptr)
            ALOGD("Pointer to output node is empty");
        return ptr;
    }
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif