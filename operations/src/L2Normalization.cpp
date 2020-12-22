/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "L2Normalization.h"
#include "CpuPreparedModel.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace l2normalization{

OutputPort l2normDataPtr;

inline OutputPort L2Normalization(const OutputPort &src, bool isAcross, bool isShareChannel) {
    auto layer = Generic("Normalize", src);
    addAttr(layer, "across_spatial", isAcross ? 1 : 0);
    addAttr(layer, "channel_shared", isShareChannel ? 1 : 0);
    return output(layer);
}

bool validate(const Operation& operation, const Model& model){
    return true;
}

bool initialize(const char* device, const Operation& operation, const Model& model){
    if (strcmp(device, "CPU") == 0){
        VLOG(L1, "OperationType::L2_NORMALIZATION");
        dumpOperationParam(operation);
        sp<CpuPreparedModel> PreparedModelObj;
        /*
        * Inputs:
        * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
        *
        * Ouputs:
        * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth].
        */
        // mPorts[operation.outputs[0]] = L2Normalization(getPort(operation.inputs[0]), true, false);
        l2normDataPtr =
            L2Normalization(PreparedModelObj->getPort(operation.inputs[0]), false, false);  // passing accross false
        return true;
    } else if (strcmp(device, "GNA") == 0){
        return false;
    } else {
        return false;
    }
}

OutputPort updateDataPtr() {
    return l2normDataPtr;
}


}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android