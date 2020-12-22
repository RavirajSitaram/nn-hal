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
#ifndef RESHAPE_H
#define RESHAPE_H

#include "common.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace reshape{

bool validate(const Operation& operation, const Model& model);
bool initialize(const char* device, const Operation& operation, const Model& model);
OutputPort updateDataPtr();
std::string getNodeName();
std::string getInputName();
std::vector<size_t> getShape();

OutputPort Reshape(const TensorDims &newDims, const OutputPort &src);

}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // RESHAPE_H