python lib/pred_imgs.py \
--config-file-box ../maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2_4s_shop.yaml \
--config-file-kp ../maskrcnn-benchmark/configs/caffe2/e2e_keypoint_only_rcnn_R_101_FPN_1x_caffe2_4s_shop_pad.yaml \
--confidence-threshold 0.9991 \
--img_root data/val_root \
--gt_annos data/annotations/keypoints_val_root.json \
--program_time_list data/lists/info_val.txt \
--save_root data/test_result/ \
--vis_gts False \
--vis_preds True \

