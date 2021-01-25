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
	std::map<uint32_t, ngraph::Output<ngraph::Node>> mIntermediateOutputs;
    std::map<uint32_t, std::shared_ptr<ngraph::opset3::Parameter>> mNgraphInputNodes;
    std::map<uint32_t, LayerInfo> mInLayerMap;
    std::map<uint32_t, LayerInfo> mOutLayerMap;

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

    // Move these to protected and add operation base friend class???
    void addIntermediateNode(uint32_t index, ngraph::Output<ngraph::Node>& outputNode) {
        if (mIntermediateOutputs.find(index) != mIntermediateOutputs.end())
            ALOGE("%s Overwriting previous output node with new node at index: %d", __func__, index);

        mIntermediateOutputs[index] = outputNode;
    }

    void addResultNode(uint32_t index, std::shared_ptr<ngraph::Node> node) {
        if (mNgraphResultNodes.find(index) != mNgraphResultNodes.end())
            ALOGE("%s Overwriting previous result node with new node at index: %d", __func__, index);

        mNgraphResultNodes[index] = node;
    }

    void addInputNode(uint32_t index, std::shared_ptr<ngraph::opset3::Parameter> node) {
        if (mNgraphInputNodes.find(index) != mNgraphInputNodes.end())
            ALOGE("%s Overwriting previous result node with new node at index: %d", __func__, index);

        mNgraphInputNodes[index] = node;
    }

    void appendNodeToMap(std::shared_ptr<ngraph::Node> node) {
        mNgraphNodes.push_back(node);
    }

    void addLayerMetadata(uint32_t index,const LayerInfo& l, bool input) {
        if (input)
            mInLayerMap[index] = l;
        else
            mOutLayerMap[index] = l;
    }
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif