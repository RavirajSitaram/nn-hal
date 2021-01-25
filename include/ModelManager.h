#ifndef ANDROID_ML_NN_MODELMANAGER_H
#define ANDROID_ML_NN_MODELMANAGER_H

#include <hidlmemory/mapping.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <ie_blob.h>

#include "Driver.h"
#include "Utils.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using ::android::hidl::memory::V1_0::IMemory;
using Blob = InferenceEngine::Blob;

// Utility class that provides functions and methods around NNAPI Model
class NnapiModelInfo {
    public:
        NnapiModelInfo(const Model& model) {
            mModel = model;
        }

        bool initRuntimeInfo() {
            if (!initializeRunTimeOperandInfo())
                return false;
            
            mPoolInfos.resize(mModel.pools.size());
            for (size_t i = 0; i < mModel.pools.size(); i++) {
                auto& poolInfo = (mPoolInfos)[i];
                if (!poolInfo.set(mModel.pools[i])) {
                    LOG(ERROR) << "Could not map pool";
                    return false;
                }
            }

            return true;
        }

        // Copy model input indices to a seperate vector
        auto getModelInputIndexes() {
            return mModel.main.inputIndexes;
        }

        uint32_t getModelInputIndex(uint32_t index) {
            return mModel.main.inputIndexes[index];
        }

        uint32_t getModelOutputIndex(uint32_t index) {
            return mModel.main.inputIndexes[index];
        }

        // Index into the operand vector
        OperandLifeTime getOperandLifetime(uint32_t operandIdx) {
            auto tmpOperand =  mModel.main.operands[operandIdx];
            return tmpOperand.lifetime;
        }

        bool isOperandLifeTimeTemp(uint32_t operandIdx) {
            if (getOperandLifetime(operandIdx) == OperandLifeTime::TEMPORARY_VARIABLE)
                return true;
            return false;
        }

        template <typename T>
        T GetConstOperand(uint32_t index) {
            dumpOperand(index, model);
            uint32_t len;
            const uint8_t* buf = GetOperandMemory(index, len);
            return GetConstFromBuffer<T>(buf, len);
        }        

        const std::vector<Operation>& getOperations() {
            return mModel.main.operations;
        }

        const Operand& getOperand(int index) {
            return mModel.main.operands[index];
        }

        RunTimeOperandInfo& getRuntimeOperand(uint32_t index) {
            return mOperands[mModel.main.inputIndexes[index]];
        }

        bool isConstOperand(int index) {
            VLOG(L1, "---------------------------------------------");
            VLOG(L1, "Operand index: %d", index);
            const auto op = mModel.main.operands[index];
            VLOG(L1, " %s", toString(op).c_str());
            bool ret = (op.lifetime == OperandLifeTime::CONSTANT_COPY ||
                        op.lifetime == OperandLifeTime::CONSTANT_REFERENCE);
            VLOG(L1, "%s", ret ? "Const" : "Non-Const");
            VLOG(L1, "---------------------------------------------");
            return ret;
        }

        const uint8_t* GetOperandMemory(int index, uint32_t& lenOut);
        IRBlob::Ptr GetConstOperandAsTensor(int operand_idx, int operation_idx);
        Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                                  uint32_t& len);
        IRBlob::Ptr GetConstWeightsOperandAsTensor(uint32_t index); // Redundant

        template <typename T>
        T ParseOperationInput(const Operation& operation, uint32_t index) {
            uint32_t inputIndex = operation.inputs[index];
            const auto operand = mModel.main.operands[inputIndex];
            const auto value = GetConstOperand<T>(inputIndex);
            VLOG(L1, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
            VLOG(L1, "Operation input index: %d, operand index: %d", index, inputIndex);
            VLOG(L1, "Operation: %s", toString(operation).c_str());
            printHelper<T>::print(value, toString(operand).c_str());
            VLOG(L1, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");

            return value;
        }

        // TODO: Move it to Utils class
        template <typename T>
        std::vector<T> GetConstVecFromBuffer(const uint8_t* buf, uint32_t len) {
            int n = len / sizeof(T);
            if (n * sizeof(T) != len) {
                VLOG(L1, "typeid(T).name() should be  multiples of %d bytes", sizeof(T));
                nnAssert(false);
            }

            std::vector<T> ret;
            for (int i = 0; i < n; i++) {
                ret.push_back(*(T*)buf);
                buf += sizeof(T);
            }
            return ret;
        }   

        template <typename T>
        std::vector<T> GetConstVecOperand(uint32_t index) {
            dumpOperand(index, model);
            uint32_t len;
            const uint8_t* buf = GetOperandMemory(index, len);
            return GetConstVecFromBuffer<T>(buf, len);
        }

        template <typename T>
        T GetConstFromBuffer(const uint8_t* buf, uint32_t len);

        Blob::Ptr getBlobFromMemoryPoolIn(const V1_3::Request& request, uint32_t index);
        void* getBlobFromMemoryPoolOut(const V1_3::Request& request, uint32_t index);
        
        //template <typename T>
        //std::vector<T> GetConstVecOperand(uint32_t index);

        Model getModel() {
            return mModel;
        }

        bool setRunTimePoolInfosFromHidlMemories(const hidl_vec<V1_3::Request::MemoryPool>& pools);
        
        bool updateRequestPoolInfos() {
            for (auto runtimeInfo : mRequestPoolInfos) {
                return runtimeInfo.update();
            }

            return true;
        }

    private:
        bool initializeRunTimeOperandInfo();

        Model mModel; // TODO: Do we need a new copy of model??
        std::vector<RunTimePoolInfo> mPoolInfos;
        std::vector<RunTimeOperandInfo> mOperands;
        std::vector<RunTimePoolInfo> mRequestPoolInfos;
};

}
}
}
}

#endif