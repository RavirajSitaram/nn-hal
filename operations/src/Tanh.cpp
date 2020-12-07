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

#include "common.h"
#include "Tanh.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace tanh{

bool validate(const Operation& operation, const Model& model){
    return true;
}

bool initialize(const std::string& device){
    if (device.compare("CPU")){
    
    } else if (device.compare("GNA")){
    
    } else {
        return false;
    }
}

}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android