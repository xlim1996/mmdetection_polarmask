import torch
import torch.nn as nn

from mmcv.runner import force_fp32
from mmdet.core import distance2bbox, multi_apply
from mmcv.ops import  ModulatedDeformConv2dPack, nms
from ..builder import HEADS, build_loss
from mmcv.cnn import (ConvModule, Scale,bias_init_with_prob,bias_init_with_prob,build_norm_layer,
                      normal_init)
import numpy as np
import cv2
import math
import time
INF = 1e8

@HEADS.register_module
class PolarMask_Head(nn.Module):

    def __init__(self,
                 num_classes,# 81 if use all of coco classes=80 + 1 background
                 in_channels, # the output of neck NN is 256
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 use_dcn=False,
                 mask_nms=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_mask=dict(type='MaskIOULoss'),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),# GN->Group Normalization BN->Batch Normalization
                 **kwargs):

        super(PolarMask_Head, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1 #cls_out_channels without background
        self.in_channels = in_channels # 256
        self.feat_channels = feat_channels # 256
        self.stacked_convs = stacked_convs # 4
        self.strides = strides # (4, 8, 16, 32, 64)
        self.regress_ranges = regress_ranges # ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF))
        self.loss_cls = build_loss(loss_cls) # FocalLoss
        self.loss_bbox = build_loss(loss_bbox) # IoULoss
        self.loss_mask = build_loss(loss_mask) # MaskIOULoss
        self.loss_centerness = build_loss(loss_centerness) # CrossEntropyLoss
        self.conv_cfg = conv_cfg  # None
        self.norm_cfg = norm_cfg  # dict(type='GN', num_groups=32, requires_grad=True)
        self.fp16_enabled = False
        # xez add for polarmask
        self.use_dcn = use_dcn
        self.mask_nms = mask_nms

        # debug vis img
        self.vis_num = 1000 #number of visualization_imgs for debug
        self.count = 0

        # test
        # self.angles = torch.range(0, 350, 10).cuda() / 180 * math.pi
        self.angles = torch.range(0, 350, 10) / 180 * math.pi
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if not self.use_dcn:
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
                self.mask_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
            else:
                self.cls_convs.append(
                    ModulatedDeformConv2dPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.cls_convs.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.cls_convs.append(nn.ReLU(inplace=True))

                self.reg_convs.append(
                    ModulatedDeformConv2dPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.reg_convs.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.reg_convs.append(nn.ReLU(inplace=True))

                self.mask_convs.append(
                    ModulatedDeformConv2dPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.mask_convs.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.mask_convs.append(nn.ReLU(inplace=True))

        self.polar_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1) # classfication
        self.polar_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)#4->(x,y,w,h) bbox
        self.polar_mask = nn.Conv2d(self.feat_channels, 36, 3, padding=1)# 36 rays's distance
        self.polar_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1) # centerness

        self.scales_bbox = nn.ModuleList([Scale(1.0) for _ in self.strides]) #?
        self.scales_mask = nn.ModuleList([Scale(1.0) for _ in self.strides]) #?

    def init_weights(self):
        if not self.use_dcn:
            #normalization_initial
            for m in self.cls_convs:
                normal_init(m.conv, std=0.01)
            for m in self.reg_convs:
                normal_init(m.conv, std=0.01)
            for m in self.mask_convs:
                normal_init(m.conv, std=0.01)
        else:
            pass

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.polar_cls, std=0.01, bias=bias_cls)
        normal_init(self.polar_reg, std=0.01)
        normal_init(self.polar_mask, std=0.01)
        normal_init(self.polar_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales_bbox, self.scales_mask)

    def forward_single(self, x, scale_bbox, scale_mask):
        cls_feat = x
        reg_feat = x
        mask_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.polar_cls(cls_feat)
        centerness = self.polar_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale_bbox(self.polar_reg(reg_feat)).float().exp()

        for mask_layer in self.mask_convs:
            mask_feat = mask_layer(mask_feat)
        mask_pred = scale_mask(self.polar_mask(mask_feat)).float().exp()

        return cls_score, bbox_pred, centerness, mask_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'mask_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             mask_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_masks,
             gt_bboxes_ignore=None,
             _gt_labels=None,
             _gt_bboxes=None,
             _gt_masks=None):
        """

        :param cls_scores: (5)(i,80,h,w)
        :param bbox_preds: (5)(i,4,h,w)
        :param centernesses: (5)(i,1,h,w)
        :param mask_preds: (5)(i,36,h,w)
        :param gt_bboxes:[20460, 4]
        :param gt_labels:[20460]
        :param img_metas:
        :param cfg: {'nms_pre': 1000, 'min_bbox_size': 0, 'score_thr': 0.05, 'nms': {'type': 'nms', 'iou_thr': 0.5}, 'max_per_img': 100}
        :param gt_masks:
        :param gt_bboxes_ignore:
        :param extra_data: dict(_gt_labels=_gt_label
                              _gt_bboxes=_gt_bboxes,
                              _gt_masks=_gt_masks)
        :return:
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(mask_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores] # (5)[h,w] - > (5)[[160, 96], [80, 48], [40, 24], [20, 12], [10, 6]]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device) # list (5)[num_points,2] -> list([15350,2],[3840,2],[960,2],[240,2],[60,2])

        labels, bbox_targets, mask_targets = self.get_targets(all_level_points, _gt_labels, _gt_bboxes, _gt_masks)
        #number of image x [20460,],number of image x[20460,4],number of image x[20460,36]
        num_imgs = cls_scores[0].size(0) # i
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores] # (5)(i,80,h,w) -> (5)(i*h*w,80)
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]# (5)(i,4,h,w) -> (5)(i*h*w,4)
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]# (5)(i,1,h,w) -> (5)(i*h*w,)
        flatten_mask_preds = [
            mask_pred.permute(0, 2, 3, 1).reshape(-1, 36)
            for mask_pred in mask_preds
        ]# (5)(i,36,h,w) -> (5)(i*h*w,36)
        flatten_cls_scores = torch.cat(flatten_cls_scores)  # [num_pixel, 80] if image shape is 1280x768 -> num_pixel = 20460 * num_imgs
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)  # [num_pixel, 4]
        flatten_mask_preds = torch.cat(flatten_mask_preds)  # [num_pixel, 36]
        flatten_centerness = torch.cat(flatten_centerness)  # [num_pixel]

        flatten_labels = torch.cat(labels).long()  # [num_pixel]
        flatten_bbox_targets = torch.cat(bbox_targets)  # [num_pixel, 4]
        flatten_mask_targets = torch.cat(mask_targets)  # [num_pixel, 36]
        flatten_points = torch.cat([points.repeat(num_imgs, 1)
                                    for points in all_level_points])  # [num_pixel,2]
        pos_inds = flatten_labels.nonzero().reshape(-1) # get the index of positive pixel which belong to the foreground points
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0 focal loss
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_mask_preds = flatten_mask_preds[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_mask_targets = flatten_mask_targets[pos_inds]
            pos_centerness_targets = self.polar_centerness_target(pos_mask_targets) # polar centerness

            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds) #return the small region bbox
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)

            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_mask = self.loss_mask(pos_mask_preds,
                                       pos_mask_targets,
                                       weight=pos_centerness_targets,
                                       avg_factor=pos_centerness_targets.sum())

            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_mask = pos_mask_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_mask=loss_mask,
            loss_centerness=loss_centerness)

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        # +stride//2 is to shift the point of the feature maps back to original image scale
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def get_targets(self, points, _gt_labels, _gt_bboxes, _gt_masks):
        """

        :param points: list(5 ,[h*w, 2]) -> list([15360,2],[3840,2],[960,2],[240,2],[48,2])
        :param extra_data: dict(_gt_labels=_gt_labels, [20460]
                              _gt_bboxes=_gt_bboxes, [20460,4]
                              _gt_masks=_gt_masks)[20460 ,36]
        :return: #number of image x [20460,],number of image x[20460,4],number of image x[20460,36]
        """
        assert len(points) == len(self.regress_ranges)

        num_levels = len(points) # 5
        # if extra_data is None ? need to check the data type and shape of extra_data ????
        labels_list = _gt_labels
        bbox_targets_list = _gt_bboxes
        mask_targets_list = _gt_masks
        # labels_list, bbox_targets_list, mask_targets_list = extra_data.values()

        # split to per img, per level
        num_points = [center.size(0) for center in points] # [15360, 3840, 960, 240, 60]
        # number 0f image * list([15360,]],[3840,],[960,],[240,],[60,])
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]  # number 0f image * list([15360,4]],[3840,4],[960,4],[240,4],[60,4])
        mask_targets_list = [
            mask_targets.split(num_points, 0)
            for mask_targets in mask_targets_list
        ]#number 0f image * list([15360,36]],[3840,36],[960,36],[240,36],[60,36])

        # concat per level image
        concat_lvl_labels = [] #[20460,?]
        concat_lvl_bbox_targets = []
        concat_lvl_mask_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
            concat_lvl_mask_targets.append(
                torch.cat(
                    [mask_targets[i] for mask_targets in mask_targets_list]))

        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_mask_targets

    def polar_centerness_target(self, pos_mask_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        centerness_targets = (pos_mask_targets.min(dim=-1)[0] / pos_mask_targets.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   mask_preds,
                   img_metas,
                   cfg,
                   rescale=None):
        """

        :param cls_scores: list(5,[img_id,80,h,w])
        :param bbox_preds: list(5,[img_id,4,h,w])
        :param centernesses: list(5,[img_id,1,h,w])
        :param mask_preds: list(5,[img_id,36,h,w])
        :param  img_meta = [
            dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=False)
        ]
        :param cfg: cfg : {'nms_pre': 1000, 'min_bbox_size': 0, 'score_thr': 0.05, 'nms': {'type': 'nms', 'iou_thr': 0.5}, 'max_per_img': 100}
        :param rescale:
        :return:
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores) # 5

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores] #(5)[h,w] - > (5)[[160, 96], [80, 48], [40, 24], [20, 12], [10, 6]]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)# list (5)[num_points,2] -> list([15350,2],[3840,2],[960,2],[240,2],[60,2])
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels) # list(5,[80,h,w])
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels) # list(5,[4,h,w])
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels) # list(5,[1,h,w])
            ]
            mask_pred_list = [
                mask_preds[i][img_id].detach() for i in range(num_levels) # list(5,[36,h,w])
            ]
            img_shape = img_metas[img_id]['img_shape'] #[1280, 768]
            scale_factor = img_metas[img_id]['scale_factor'] #[1.0, 1.0] ? need to check the shape of img_metas
            det_bboxes = self.get_bboxes_single(cls_score_list,
                                                bbox_pred_list,
                                                mask_pred_list,
                                                centerness_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale) #torch.Size([1000, 5]) torch.Size([1000]) torch.Size([1000, 2, 36])
            result_list.append(det_bboxes)
        return result_list #list(image_len,(torch.Size([1000, 5]) torch.Size([1000]) torch.Size([1000, 2, 36]))

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mask_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        :param cls_scores: list(5,[80,h,w])
        :param bbox_preds: list(5,[4,h,w])
        :param mask_preds: list(5,[36,h,w])
        :param centernesses: list(5,[1,h,w])
        :param mlvl_points: list(5,[num_points,2]) -> list([15350,2],[3840,2],[960,2],[240,2],[60,2])
        :param img_shape: [768, 1280, 3]
        :param scale_factor: [1.0, 1.0]
        :param cfg:{'nms_pre': 1000, 'min_bbox_size': 0, 'score_thr': 0.05, 'nms': {'type': 'nms', 'iou_thr': 0.5}, 'max_per_img': 100}
        :param rescale: False
        :return: det_bboxes, det_labels, det_masks # torch.Size([1000, 5]) torch.Size([1000]) torch.Size([1000, 2, 36])
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_masks = []
        mlvl_centerness = []
        for cls_score, bbox_pred, mask_pred, centerness, points in zip(
                cls_scores, bbox_preds, mask_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid() # [15350,80], [3840,80], [960,80], [240,80], [60,80]

            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid() # [15350], [3840], [960], [240], [60]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4) # [15350,4], [3840,4], [960,4], [240,4], [60,4]
            mask_pred = mask_pred.permute(1, 2, 0).reshape(-1, 36) # [15350,36], [3840,36], [960,36], [240,36], [60,36]
            nms_pre = cfg.get('nms_pre', -1) # if nms_pre is key in cfg, return cfg['nms_pre'], else return -1
            #num_pre = 1000
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)  #centerness[:, None] -> [15350,1], [3840,1], [960,1], [240,1], [60,1]  max_scores -> [15350], [3840], [960], [240], [60]
                _, topk_inds = max_scores.topk(nms_pre) #get the topk_inds of max_scores [1000]
                points = points[topk_inds, :] # [1000,2]
                bbox_pred = bbox_pred[topk_inds, :] # [1000,4]
                mask_pred = mask_pred[topk_inds, :] # [1000,36]
                scores = scores[topk_inds, :] # [1000,80]
                centerness = centerness[topk_inds] # [1000]
            # get the bboxes of the topk_inds of like coco_seg.py just choose the small region
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape) # (1000, 4)
            masks = distance2mask(points, mask_pred, self.angles, max_shape=img_shape) # (1000, 2, 36)

            mlvl_bboxes.append(bboxes) # list([1000,4],[1000,4],[1000,4],[1000,4],[1000,4])
            mlvl_scores.append(scores) # list([1000,80],[1000,80],[1000,80],[1000,80],[1000,80])
            mlvl_centerness.append(centerness) # list([1000],[1000],[1000],[1000],[1000])
            mlvl_masks.append(masks) # list([1000,2,36],[1000,2,36],[1000,2,36],[1000,2,36],[1000,2,36])

        mlvl_bboxes = torch.cat(mlvl_bboxes) # [5000,4]
        mlvl_masks = torch.cat(mlvl_masks) # [5000,2,36]
        if rescale:
            _mlvl_bboxes = mlvl_bboxes / mlvl_bboxes.new_tensor(scale_factor)
            try:
                scale_factor = torch.Tensor(scale_factor)[:2].cuda().unsqueeze(1).repeat(1, 36)
                _mlvl_masks = mlvl_masks / scale_factor
            except:
                _mlvl_masks = mlvl_masks / mlvl_masks.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores) # [5000,80]
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1) # [5000,1]
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)# [5000,81]
        mlvl_centerness = torch.cat(mlvl_centerness) # [5000]

        centerness_factor = 0.5  # mask centerness is smaller than origin centerness, so add a constant is important or the score will be too low.
        #aviod overlappping bbox
        if self.mask_nms:
            '''1 mask->min_bbox->nms, performance same to origin box'''
            a = _mlvl_masks # [5000,2,36]
            #a[:, 0].min(1)[0] -> min_x_values [5000] , a[:, 1].min(1)[0] -> min_y_values [5000], a[:, 0].max(1)[0] -> max_x_values [5000], a[:, 1].max(1)[0] -> max_y_values [5000]
            _mlvl_bboxes = torch.stack([a[:, 0].min(1)[0],a[:, 1].min(1)[0],a[:, 0].max(1)[0],a[:, 1].max(1)[0]],-1) # [5000,4]
            det_bboxes, det_labels, det_masks = multiclass_nms_with_mask(
                _mlvl_bboxes,
                mlvl_scores,
                _mlvl_masks,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness + centerness_factor)

        else:
            '''2 origin bbox->nms, performance same to mask->min_bbox'''
            det_bboxes, det_labels, det_masks = multiclass_nms_with_mask(
                _mlvl_bboxes,
                mlvl_scores,
                _mlvl_masks,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness + centerness_factor)

        return det_bboxes, det_labels, det_masks # torch.Size([1000, 5]) torch.Size([1000]) torch.Size([1000, 2, 36])


# test
def distance2mask(points, distances, angles, max_shape=None):
    '''Decode distance prediction to 36 mask points
    Args:
        points (Tensor): Shape (n, 2), [x, y]. [1000,2]
        distance (Tensor): Distance from the given point to 36,from angle 0 to 350. [1000,36]
        angles (Tensor):
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded masks. (1000, 2, 36)
    '''

    num_points = points.shape[0] # 1000
    points = points[:, :, None].repeat(1, 1, 36) # (1000, 2, 36)
    c_x, c_y = points[:, 0], points[:, 1] # (1000, 36)

    sin = torch.sin(angles).to('cuda') # (36) get sin value from angle
    cos = torch.cos(angles).to('cuda') # (36) get cos value from angle
    sin = sin[None, :].repeat(num_points, 1) # (1000, 36)
    cos = cos[None, :].repeat(num_points, 1) # (1000, 36)

    x = distances * sin + c_x
    y = distances * cos + c_y
    # clip the mask to within image size
    if max_shape is not None:
        x = x.clamp(min=0, max=max_shape[1] - 1)
        y = y.clamp(min=0, max=max_shape[0] - 1)

    res = torch.cat([x[:, None, :], y[:, None, :]], dim=1) # (1000, 2, 36)
    return res



def multiclass_nms_with_mask(multi_bboxes,
                   multi_scores,
                   multi_masks,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels, masks = [], [], []
    nms_cfg_ = nms_cfg.copy()
    # remove type info in nms_cfg
    nms_cfg_.pop('type', None)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
            _masks  = multi_masks[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        # cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, index = nms(_bboxes, _scores, **nms_cfg_)
        cls_masks = _masks[index]
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
        masks.append(cls_masks)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        masks = torch.cat(masks)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
            masks = masks[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        masks = multi_bboxes.new_zeros((0, 2, 36))

    return bboxes, labels, masks