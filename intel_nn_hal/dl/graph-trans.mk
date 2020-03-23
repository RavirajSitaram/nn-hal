LOCAL_PATH := $(call my-dir)/../../../dldt
include $(CLEAR_VARS)

LOCAL_MODULE := libvpu_graph_transformer
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MODULE_RELATIVE_PATH := hw
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_SRC_FILES := \
	inference-engine/src/vpu/graph_transformer/src/blob_reader.cpp \
	inference-engine/src/vpu/graph_transformer/src/frontend/custom_layer.cpp \
	inference-engine/src/vpu/graph_transformer/src/graph_transformer.cpp \
	inference-engine/src/vpu/graph_transformer/src/parsed_config.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/pass_manager.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/stub_stage.cpp \
	inference-engine/src/vpu/graph_transformer/src/backend/backend.cpp \
	inference-engine/src/vpu/graph_transformer/src/backend/dump_to_dot.cpp \
	inference-engine/src/vpu/graph_transformer/src/backend/get_meta_data.cpp \
	inference-engine/src/vpu/graph_transformer/src/backend/serialize.cpp \
	inference-engine/src/vpu/graph_transformer/src/frontend/detect_network_batch.cpp \
	inference-engine/src/vpu/graph_transformer/src/frontend/frontend.cpp \
	inference-engine/src/vpu/graph_transformer/src/frontend/ie_parsed_network.cpp \
	inference-engine/src/vpu/graph_transformer/src/frontend/in_out_convert.cpp	 \
	inference-engine/src/vpu/graph_transformer/src/frontend/move_const_inputs_to_blobs.cpp	 \
	inference-engine/src/vpu/graph_transformer/src/frontend/parse_data.cpp \
	inference-engine/src/vpu/graph_transformer/src/frontend/pre_process.cpp \
	inference-engine/src/vpu/graph_transformer/src/frontend/remove_const_layers.cpp \
	inference-engine/src/vpu/graph_transformer/src/frontend/unroll_loops.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/mx_stage.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/pass_manager.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/special_stage_processor.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/allocator/allocator.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/allocator/shaves.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/hw/tiling.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/hw/utility.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/hw/conv_tiling/hw_convolution_tiler.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/hw/conv_tiling/hw_stage_tiler.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/hw/pooling_tiling/hw_pooling_tiler.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/hw/pooling_tiling/hw_stage_tiler.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/sw/utility.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/add_copy_for_outputs_inside_network.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/add_reorder_inputs_to_channel_minor.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/adjust_data_batch.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/adjust_data_layout.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/adjust_data_location.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/allocate_resources.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/count_stages_in_loops.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/eliminate_const_concat.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/eliminate_copy.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/final_check.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/finalize_hw_ops.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/gemm_transpose.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/hw_conv_tiling.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/hw_fc_tiling.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/hw_padding.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/hw_pooling_tiling.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/inject_sw.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/merge_eltwise_and_relu.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/merge_hw_stages.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/merge_parallel_fc.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/merge_permute_stages.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/merge_relu_and_bias.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/process_special_stages.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/remove_unused_stages_outputs.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/replace_deconv_by_conv.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/replace_fc_by_conv.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/replace_priorbox_with_const.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/replace_with_reduce_mean.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/replace_with_screlu.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/reshape_dilation_conv.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/split_grouped_conv.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/split_hw_conv_and_pool.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/split_hw_depth_convolution.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/strided_slice.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/swap_concat_and_hw_ops.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/sw_conv_adaptation.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/sw_deconv_adaptation.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/sw_fc_adaptation.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/sw_pooling_adaptation.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/uplift_activation_stages.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/weights_analysis.cpp \
	inference-engine/src/vpu/graph_transformer/src/middleend/passes/initial_check.cpp \
	inference-engine/src/vpu/graph_transformer/src/model/data.cpp \
	inference-engine/src/vpu/graph_transformer/src/model/data_desc.cpp \
	inference-engine/src/vpu/graph_transformer/src/model/model.cpp \
	inference-engine/src/vpu/graph_transformer/src/model/stage.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/argmax.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/batch_norm.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/bias.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/clamp.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/concat.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/convert.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/convolution.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/copy.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/crop.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/ctc_decoder.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/custom.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/convert.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/deconvolution.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/detection_output.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/eltwise.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/elu.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/erf.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/expand.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/exp.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/exp_detectionoutput.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/fc.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/floor.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/gather.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/gemm.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/grn.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/interp.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/log.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/loop_end.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/loop_start.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/mtcnn.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/mvn.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/mx_stage.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/nms.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/none.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/normalize.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/norm.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/oneHot.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/pad.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/permute.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/pooling.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/post_op_stage.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/power.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/prelu.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/priorbox.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/proposal.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/psroipooling.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/reduce.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/region_yolo.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/relu.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/reorg_yolo.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/resample.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/reshape.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/reverse_sequence.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/rnn.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/roi_feature_extractor.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/roipooling.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/scale.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/screlu.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/shrink.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/sigmoid.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/strided_slice.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/softmax.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/split.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/tanh.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/tensor_iterator.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/tile.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/topk.cpp \
	inference-engine/src/vpu/graph_transformer/src/stages/post_op_stage.cpp \

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/include/cpp \
	$(LOCAL_PATH)/inference-engine/include/details \
	$(LOCAL_PATH)/inference-engine/include/vpu \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/base \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/impl \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/interface \
	$(LOCAL_PATH)/inference-engine/src/vpu/myriad_plugin \
	$(LOCAL_PATH)/inference-engine/src \
	$(LOCAL_PATH)/inference-engine/src/vpu/common/include \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/src/middleend/allocator \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/backend \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/frontend \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/middleend \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/model \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/stages \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/utils \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/mvnc/include \
	$(LOCAL_PATH)/ngraph/src


LOCAL_CFLAGS += -std=c++11 -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -DIE_THREAD=IE_THREAD_OMP -DENABLE_VPU -DENABLE_MYRIAD -D__ANDROID__ -DNDEBUG -DIMPLEMENT_INFERENCE_ENGINE_API -D_FORTIFY_SOURCE=2 -fPIE
#LOCAL_CFLAGS += -DNNLOG


# LOCAL_STATIC_LIBRARIES := libvpu_common
LOCAL_SHARED_LIBRARIES := liblog libinference_engine
LOCAL_STATIC_LIBRARIES := libpugixml

include $(BUILD_STATIC_LIBRARY)
