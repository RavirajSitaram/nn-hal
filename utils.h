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
#include "Driver.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

enum DebugLevel {
    L0,
    L1,
    L2,
    L3,
    L4,
};

enum PaddingScheme {
    kPaddingUnknown = 0,
    /**
     * SAME padding.
     * Padding on both ends are the "same":
     *     padding_to_beginning =  total_padding / 2
     *     padding_to_end       = (total_padding + 1)/2.
     * i.e., for even number of padding, padding to both ends are exactly
     * the same; for odd number of padding, padding to the ending is bigger
     * than the padding to the beginning by 1.
     *
     * total_padding is a function of input, stride and filter size.
     * It could be computed as follows:
     *    out_size = (input + stride - 1) / stride;
     *    needed_input = (out_size - 1) * stride + filter_size
     *    total_padding = max(0, needed_input - output_size)
     *  The computation is the same for the horizontal and vertical directions.
     */
    kPaddingSame = 1,
    /**
     * VALID padding.
     * No padding. When the input size is not evenly divisible by
     * the filter size, the input at the end that could not fill
     * the whole filter tile will simply be ignored.
     */
    kPaddingValid = 2,
};

unsigned int debugMask = ((1 << (L1 + 1)) - 1);

#ifdef NN_DEBUG
#define VLOG(l, x, ...)                                                          \
    do {                                                                         \
        if (debugMask & (1 << l)) ALOGI("[%s] " x, __FUNCTION__, ##__VA_ARGS__); \
    } while (0)

#define VLOGDIMS(l, d, header)                                                         \
    do {                                                                               \
        auto size = (d).size();                                                        \
        VLOG(l, "%s: vectors {%d, %d, %d, %d}", header, (d)[0], size > 1 ? (d)[1] : 0, \
             size > 2 ? (d)[2] : 0, size > 3 ? (d)[3] : 0);                            \
    } while (0)

#define dumpOperand(index, model)                                      \
    do {                                                        \
        const auto op = model.operands[index];                 \
        ALOGI("---------------------------------------------"); \
        ALOGI("Operand index: %d", index);                      \
        ALOGI("%s", toString(op).c_str());                      \
        ALOGI("---------------------------------------------"); \
    } while (0)

#define dumpOperation(operation)                                \
    do {                                                        \
        ALOGI("---------------------------------------------"); \
        ALOGI("Operation:");                                    \
        ALOGI("%s", toString(operation).c_str());               \
        ALOGI("---------------------------------------------"); \
    } while (0)

#define dumpOperationSupport(operation, support)                    \
    do {                                                            \
        ALOGI("---------------------------------------------");     \
        ALOGI("Operation support: %s", support ? "True" : "False"); \
        ALOGI("%s", toString(operation).c_str());                   \
        ALOGI("---------------------------------------------");     \
    } while (0)

#define dumpOperationParam(operation)             \
    do {                                          \
        ALOGI("dumping operation-params");        \
        ALOGI("%s", toString(operation).c_str()); \
    } while (0)

#else
#define VLOG(...)
#define VLOGDIMS(l, d, header)
#define dumpOperand(...)
#define dumpOperation(operation)
#define dumpOperationSupport(operation, support)
#define dumpOperationParam(operation)
#endif

#define WRONG_DIM (-1)

#define nnAssert(v)                                                                            \
    do {                                                                                       \
        if (!(v)) {                                                                            \
            LOG(ERROR) << "nnAssert failed at " << __FILE__ << ":" << __LINE__ << " - '" << #v \
                       << "'\n";                                                               \
            abort();                                                                           \
        }                                                                                      \
    } while (0)

#define EXPL_PAD_PARAMS_CONV 10
#define IMPL_PAD_PARAMS_CONV 7
#define EXPL_PAD_PARAMS_DW_CONV 11
#define IMPL_PAD_PARAMS_DW_CONV 8
#define EXPL_PAD 1
#define IMPL_PAD 2
#define SOFTMAX_INPUT_PARAMS 2
#define NHWC_DIM_NUM 4
#define NHWC_CH_IDX 3
#define NHWC_HT_IDX 1
#define NHWC_WD_IDX 2
// operand index as from  1.1/type.hal
#define OP_INPUT_IDX_CONV 0
#define OP_FILTER_IDX_CONV 1
#define OP_BIAS_IDX_CONV 2
#define OP_PADSCHEME_IDX_CONV 3
#define OP_PADL_IDX_CONV 3
#define OP_PADR_IDX_CONV 4
#define OP_PADH_IDX_CONV 5
#define OP_PADW_IDX_CONV 6
#define OP_STRD_WD_IDX_EXPL_CONV 7
#define OP_STRD_HT_IDX_EXPL_CONV 8
#define OP_STRD_WD_IDX_IMPL_CONV 4
#define OP_STRD_HT_IDX_IMPL_CONV 5
#define OP_ACTV_FUNC_IDX_IMPL_CONV 6
#define OP_ACTV_FUNC_IDX_EXPL_CONV 9
#define OP_ACTV_FUNC_IDX_IMPL_DW_CONV 7
#define OP_ACTV_FUNC_IDX_EXPL_DW_CONV 10
#define OP_DW_CONV_DPM_IMPL 6  // depth multiplier
#define OP_DW_CONV_DPM_EXPL 9
#define OP_ADD_OPR1_IDX 0
#define OP_ADD_OPR1_IDX 1

// average_pooling_2d as in type.hal
#define EXPL_PAD_PARAMS_POOL 10
#define IMPL_PAD_PARAMS_POOL 7
#define OP_INPUT_IDX_POOL 0
#define OP_PADL_IDX_POOL 1
#define OP_PADR_IDX_POOL 2
#define OP_PADH_IDX_POOL 3
#define OP_PADW_IDX_POOL 4
#define OP_STRD_WD_IDX_EXPL_POOL 5
#define OP_STRD_HT_IDX_EXPL_POOL 6
#define OP_FLT_WD_IDX_EXPL_POOL 7
#define OP_FLT_HT_IDX_EXPL_POOL 8
#define OP_ACTV_FUNC_IDX_EXPL_POOL 9

#define OP_PADSCHEME_IDX_POOL 1
#define OP_STRD_WD_IDX_IMPL_POOL 2
#define OP_STRD_HT_IDX_IMPL_POOL 3
#define OP_FLT_WD_IDX_IMPL_POOL 4
#define OP_FLT_HT_IDX_IMPL_POOL 5
#define OP_ACTV_FUNC_IDX_IMPL_POOL 6

// fully_connected as in type.hal
#define OP_INPUT_IDX_FC 0
#define OP_WGHT_IDX_FC 1
#define OP_BIAS_IDX_FC 2
#define OP_ACTV_IDX_FC 3
#define FC_INPUT_PARAMS 4

// ADD operation
#define ADD_INPUT_PARAMS 3
#define OP_INPUT0_IDX_ADD 0
#define OP_INPUT1_IDX_ADD 1
#define OP_ACTV_IDX_ADD 2

#define CHECK_OPERAND_2D(params, idx_x, idx_y)                                                 \
    do {                                                                                       \
        VLOG(L1, "As found in %s", __func__);                                                  \
        if (params.x < 0 || params.y < 0) {                                                    \
            VLOG(L1, "Invalid Point2D Operands at index [%d ,%d] , aborting!!", idx_x, idx_y); \
            return false;                                                                      \
        }                                                                                      \
    } while (0)

#define EXP_MASK_F32 0x7F800000U
#define EXP_MASK_F16 0x7C00U

unsigned short float2half(unsigned f) {
    unsigned f_exp, f_sig;
    unsigned short h_sgn, h_exp, h_sig;

    h_sgn = (unsigned short)((f & 0x80000000u) >> 16);
    f_exp = (f & 0x7f800000u);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            /* Inf or NaN */
            f_sig = (f & 0x007fffffu);
            if (f_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                unsigned short ret = (unsigned short)(0x7c00u + (f_sig >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return (unsigned short)(h_sgn + 0x7c00u);
            }
        } else {
            /* overflow to signed inf */
#if NPY_HALF_GENERATE_OVERFLOW
            npy_set_floatstatus_overflow();
#endif
            return (unsigned short)(h_sgn + 0x7c00u);
        }
    }

    /* Exponent underflow converts to a subnormal half or signed zero */
    if (f_exp <= 0x38000000u) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero halfs.
         */
        if (f_exp < 0x33000000u) {
#if NPY_HALF_GENERATE_UNDERFLOW
            /* If f != 0, it underflowed to 0 */
            if ((f & 0x7fffffff) != 0) {
                npy_set_floatstatus_underflow();
            }
#endif
            return h_sgn;
        }
        /* Make the subnormal significand */
        f_exp >>= 23;
        f_sig = (0x00800000u + (f & 0x007fffffu));
#if NPY_HALF_GENERATE_UNDERFLOW
        /* If it's not exactly represented, it underflowed */
        if ((f_sig & (((unsigned)1 << (126 - f_exp)) - 1)) != 0) {
            npy_set_floatstatus_underflow();
        }
#endif
        f_sig >>= (113 - f_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
#if NPY_HALF_ROUND_TIES_TO_EVEN
        /*
         * If the last bit in the half significand is 0 (already even), and
         * the remaining bit pattern is 1000...0, then we do not add one
         * to the bit after the half significand.  In all other cases, we do.
         */
        if ((f_sig & 0x00003fffu) != 0x00001000u) {
            f_sig += 0x00001000u;
        }
#else
        f_sig += 0x00001000u;
#endif
        h_sig = (unsigned short)(f_sig >> 13);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return (unsigned short)(h_sgn + h_sig);
    }

    /* Regular case with no overflow or underflow */
    h_exp = (unsigned short)((f_exp - 0x38000000u) >> 13);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_sig = (f & 0x007fffffu);
#if NPY_HALF_ROUND_TIES_TO_EVEN
    /*
     * If the last bit in the half significand is 0 (already even), and
     * the remaining bit pattern is 1000...0, then we do not add one
     * to the bit after the half significand.  In all other cases, we do.
     */
    if ((f_sig & 0x00003fffu) != 0x00001000u) {
        f_sig += 0x00001000u;
    }
#else
    f_sig += 0x00001000u;
#endif
    h_sig = (unsigned short)(f_sig >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
#if NPY_HALF_GENERATE_OVERFLOW
    h_sig += h_exp;
    if (h_sig == 0x7c00u) {
        npy_set_floatstatus_overflow();
    }
    return h_sgn + h_sig;
#else
    return h_sgn + h_exp + h_sig;
#endif
}
void floattofp16(short* dst, float* src, unsigned nelem) {
    unsigned i;
    unsigned short* _dst = (unsigned short*)dst;
    unsigned* _src = (unsigned*)src;

    for (i = 0; i < nelem; i++) _dst[i] = float2half(_src[i]);
}

// Function to convert F32 into F16
// F32: exp_bias:127 SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM.
// F16: exp_bias:15  SEEEEEMM MMMMMMMM


// small helper function to represent uint32_t value as float32
float asfloat(uint32_t v) { return *reinterpret_cast<float*>(&v); }

// Function to convert F32 into F16
float f16tof32(short x) {
    // this is storage for output result
    uint32_t u = x;

    // get sign in 32bit format
    uint32_t s = ((u & 0x8000) << 16);

    // check for NAN and INF
    if ((u & EXP_MASK_F16) == EXP_MASK_F16) {
        // keep mantissa only
        u &= 0x03FF;

        // check if it is NAN and raise 10 bit to be align with intrin
        if (u) {
            u |= 0x0200;
        }

        u <<= (23 - 10);
        u |= EXP_MASK_F32;
        u |= s;
    } else if ((x & EXP_MASK_F16) ==
               0) {  // check for zero and denormals. both are converted to zero
        u = s;
    } else {
        // abs
        u = (u & 0x7FFF);

        // shift mantissa and exp from f16 to f32 position
        u <<= (23 - 10);

        // new bias for exp (f16 bias is 15 and f32 bias is 127)
        u += ((127 - 15) << 23);

        // add sign
        u |= s;
    }

    // finaly represent result as float and return
    return *reinterpret_cast<float*>(&u);
}

// This function convert f32 to f16 with rounding to nearest value to minimize error
// the denormal values are converted to 0.
short f32tof16(float x) {
    // create minimal positive normal f16 value in f32 format
    // exp:-14,mantissa:0 -> 2^-14 * 1.0
    static float min16 = asfloat((127 - 14) << 23);

    // create maximal positive normal f16 value in f32 and f16 formats
    // exp:15,mantissa:11111 -> 2^15 * 1.(11111)
    static float max16 = asfloat(((127 + 15) << 23) | 0x007FE000);
    static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

    // define and declare variable for intermidiate and output result
    // the union is used to simplify representation changing
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;

    // get sign in 16bit format
    uint32_t s = (v.u >> 16) & 0x8000;  // sign 16:  00000000 00000000 10000000 00000000

    // make it abs
    v.u &= 0x7FFFFFFF;  // abs mask: 01111111 11111111 11111111 11111111

    // check NAN and INF
    if ((v.u & EXP_MASK_F32) == EXP_MASK_F32) {
        if (v.u & 0x007FFFFF) {
            return s | (v.u >> (23 - 10)) | 0x0200;  // return NAN f16
        } else {
            return s | (v.u >> (23 - 10));  // return INF f16
        }
    }

    // to make f32 round to nearest f16
    // create halfULP for f16 and add it to origin value
    float halfULP = asfloat(v.u & EXP_MASK_F32) * asfloat((127 - 11) << 23);
    v.f += halfULP;

    // if input value is not fit normalized f16 then return 0
    // denormals are not covered by this code and just converted to 0
    if (v.f < min16 * 0.5F) {
        return s;
    }

    // if input value between min16/2 and min16 then return min16
    if (v.f < min16) {
        return s | (1 << 10);
    }

    // if input value more than maximal allowed value for f16
    // then return this maximal value
    if (v.f >= max16) {
        return max16f16 | s;
    }

    // change exp bias from 127 to 15
    v.u -= ((127 - 15) << 23);

    // round to f16
    v.u >>= (23 - 10);

    return v.u | s;
}

void f16tof32Arrays(float* dst, const short* src, uint32_t& nelem, float scale = 1,
                    float bias = 0) {
    VLOG(L1, "convert f16tof32Arrays...\n");
    const short* _src = reinterpret_cast<const short*>(src);

    for (uint32_t i = 0; i < nelem; i++) {
        dst[i] = f16tof32(_src[i]) * scale + bias;
    }
}

void f32tof16Arrays(short* dst, const float* src, uint32_t& nelem, float scale = 1,
                    float bias = 0) {
    VLOG(L1, "convert f32tof16Arrays...");
    for (uint32_t i = 0; i < nelem; i++) {
        dst[i] = f32tof16(src[i] * scale + bias);
    }
}

template <typename T>
struct printHelper {
    static void print(const T& value, const char* Obj) {}
};

template <>
struct printHelper<int32_t> {
    static void print(const int32_t& value, const char* operand) {
        VLOG(L1, "Operand: value: %d, %s", value, operand);
    }
};

template <>
struct printHelper<float> {
    static void print(const float& value, const char* operand) {
        VLOG(L1, "Operand: value: %f, %s", value, operand);
    }
};

template <typename T>
T GetConstFromBuffer(const uint8_t* buf, uint32_t len) {
    VLOG(L1, "buf: %p, len: %d", buf, len);
    if (len != sizeof(T)) {
        VLOG(L1, "fix me: typeid(T).name() should be %d bytes", sizeof(T));
        // fix me if buffer is of type float and if float and OperandLifeTime::CONSTANT_REFERENCE
        nnAssert(false);
    }
    return *(T*)(buf);
}

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

int sizeOfData(OperandType type, std::vector<uint32_t> dims) {
    int size;
    switch (type) {
        case OperandType::FLOAT32:
            size = 4;
            break;
        case OperandType::TENSOR_FLOAT32:
            size = 4;
            break;
        case OperandType::TENSOR_INT32:
            size = 4;
            break;
        case OperandType::TENSOR_QUANT8_ASYMM:
        case OperandType::INT32:
            size = 1;
            break;

        default:
            size = 0;
    }
    for (auto d : dims) size *= d;

    return size;
}

#ifdef NN_DEBUG
template <typename T>
void printBuffer(int level, T* buf, int num, int items, const char* format, uint32_t buf_len) {
    const size_t maxlen = 1024;
    char str[maxlen] = {0};
    int start = 0;
    int n = 0;
    while (n < num) {
        int offset = 0;
        n = (n + items) > num ? num : n + items;
        offset = snprintf(str, sizeof(str) - strnlen(str, maxlen), "[%d->%d]:\t", start, n);
        for (int i = start; i < n; i++) {
            if (i < buf_len) {
                offset +=
                    snprintf(str + offset, sizeof(str) - strnlen(str, maxlen), format, buf[i]);
            }
        }
        start = n;
        VLOG(level, "%s", str);
    }
}

void printOperandbuf(int level, const uint8_t* buffer, const std::vector<uint32_t>& dims,
                     uint32_t buffer_length, int limit = 0) {
    auto dimsize = dims.size();
    auto type = OperandType::TENSOR_FLOAT32;  // operand.type;
    int size = 1;
    for (int i = 0; i < dimsize; i++) size *= dims[i];

    if (limit > 0 && limit < size) size = limit;

    if (type == OperandType::TENSOR_FLOAT32) {
        // float *buf = static_cast<float *>(operand.buffer);
        printBuffer<float>(level, (float*)buffer, size, 10, "%f\t", buffer_length);
    } else if (type == OperandType::TENSOR_INT32) {
        // int32_t *buf = static_cast<int32_t *>(data_handle());
        // printBuffer<int32_t>(level, buf, size, 10, "%d\t");
    } else {
        VLOG(level, "Do not support type %d", type);
    }
}

#endif

template <typename T>
T getOperandConstVal(const Model& model, const Operand& operand) {
    const T* data = reinterpret_cast<const T*>(&model.operandValues[operand.location.offset]);
    return data[0];
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android