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

#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>

#include "BasePreparedModel.h"
#include "IENetwork.h"
#include "IRLayer.h"
#include "IRLayers.h"
#include "Driver.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

OutputPort PreparedModel::handleFusion(const OutputPort& out, int32_t fusedOp) {
    VLOG(L1, "fusedOp: %d", fusedOp);
    OutputPort ret = out;
    if (fusedOp == (int32_t)FusedActivationFunc::RELU) {
        VLOG(L1, "fusedOp is RELU");
        ret = ReLU(out);
        mCreateNgraph->addRelu(ret->getName(), out->getName());
    } else if (fusedOp == (int32_t)FusedActivationFunc::RELU1) {
        VLOG(L1, "fusedOp is RELU1");
        ret = Clamp(out, -1, 1);
        mCreateNgraph->addClamp(ret->getName(), out->getName(), -1, 1);
    } else if (fusedOp == (int32_t)FusedActivationFunc::RELU6) {
        VLOG(L1, "fusedOp is RELU6");
        ret = Clamp(out, 0, 6);
        mCreateNgraph->addClamp(ret->getName(), out->getName(), 0, 6);
    }

    VLOG(L1, "No ActivationFunc");
    return ret;
}

void calculateExplicitPadding(int32_t in_size, int32_t stride, int32_t filter_size,
                              int32_t padding_implicit, int32_t* padding_head,
                              int32_t* padding_tail) {
    *padding_head = 0;
    *padding_tail = 0;

    if (padding_implicit == kPaddingSame) {
        int32_t out_size = (in_size + stride - 1) / stride;
        int32_t tmp = (out_size - 1) * stride + filter_size;
        if (tmp > in_size) {
            *padding_head = (tmp - in_size) / 2;
            *padding_tail = (tmp - in_size) - *padding_head;
        }
    }
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android