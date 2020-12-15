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
#ifndef COMMON_H
#define COMMON_H

#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>

#include "Driver.h"
#include "BasePreparedModel.h"
#include "utils.h"
// #include "CpuPreparedModel.h"

#include "IENetwork.h"
#include "IRLayer.h"
#include "IRLayers.h"
#include "create_ngraph.hpp"
#include "Relu.h"

// extern unsigned int debugMask = ((1 << (L1 + 1)) - 1);

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

extern unsigned int debugMask;

inline void calculateExplicitPadding(int32_t in_size, int32_t stride, int32_t filter_size,
                              int32_t padding_implicit, int32_t* padding_head,
                              int32_t* padding_tail) {
    *padding_head = 0;
    *padding_tail = 0;
    
    // if (padding_implicit == kPaddingSame) {
    if (padding_implicit == 1) {
        int32_t out_size = (in_size + stride - 1) / stride;
        int32_t tmp = (out_size - 1) * stride + filter_size;
        if (tmp > in_size) {
            *padding_head = (tmp - in_size) / 2;
            *padding_tail = (tmp - in_size) - *padding_head;
        }
    }
}

// template <typename T>
// void addAttr(IRLayer layer, const std::string &a_name, T val) {
//     std::stringstream oss;
//     oss << val;
//     layer->params[a_name] = oss.str();
// };

// template<class T>


}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // COMMON_H