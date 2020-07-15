python lib/pred_imgs.py \
--config-file-box ../maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2_4s_shop.yaml \
--config-file-kp ../maskrcnn-benchmark/configs/caffe2/e2e_keypoint_only_rcnn_R_101_FPN_1x_caffe2_4s_shop_pad.yaml \
--confidence-threshold 0.9991 \
--img_root data/test_imgs_root \
--gt_annos data/annotations/keypoints_val_root.json \
--program_time_list data/lists/test_v1/info_val_130000.txt \
--save_root data/test/result_v4_multi_frames_3 \
--vis_gts False \
--vis_preds False
