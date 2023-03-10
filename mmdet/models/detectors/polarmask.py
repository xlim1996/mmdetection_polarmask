from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn as nn

from mmdet.core import bbox2result
import numpy as np
import cv2
import time
import torch


@DETECTORS.register_module()
class PolarMask(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PolarMask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      _gt_labels=None,
                      _gt_bboxes=None,
                      _gt_masks=None
                      ):

        # if _gt_labels is not None:
        #     extra_data = dict(_gt_labels=_gt_labels,
        #                       _gt_bboxes=_gt_bboxes,
        #                       _gt_masks=_gt_masks)
        # else:
        #     extra_data = None

        #extract features from backbone and neck
        x = self.extract_feat(img)
        #get outputs from bbox_head(polarmask_head)
        outs = self.bbox_head(x) # dont know the output? -> forward(x) in polarmask_head.py ->  return list(x) (cls_score, bbox_pred, centerness, mask_pred)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        #call bounding box head loss function to get losses
        losses = self.bbox_head.loss(
            *loss_inputs,
            gt_masks = gt_masks,
            gt_bboxes_ignore=gt_bboxes_ignore,
            _gt_labels=_gt_labels,
            _gt_bboxes=_gt_bboxes,
            _gt_masks=_gt_masks
        )
        return losses


    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        #bbox decode and reconstruct
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        # bbox_results = [
        #     bbox2result(det_bbox, det_label, self.bbox_head.num_classes-1)
        #     for det_bbox,det_label ,det_mask in bbox_list
        # ]
        
        # mask_results = [ mask2result(det_mask, det_label, self.bbox_head.num_classes, img_meta[0]) for _, det_label, det_mask in bbox_list]
        

        results = [
            bbox_mask2result(det_bboxes, det_masks, det_labels, self.bbox_head.num_classes, img_meta[0])
            for det_bboxes, det_labels, det_masks in bbox_list]


        # result = []
        # for i in len(bbox_results):
        #     result.append(bbox_results[i], mask_results[i])    
        return results
 
 
 
# '''bbox and mask 转成result mask要画图'''
def mask2result(masks, labels, num_classes, img_meta):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        masks (Tensor): shape (n, 2, 36)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    ori_shape = img_meta['ori_shape']
    img_h, img_w, _ = ori_shape

    mask_results = [[] for _ in range(num_classes - 1)]

    for i in range(masks.shape[0]):
        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask = [masks[i].transpose(1,0).unsqueeze(1).int().data.cpu().numpy()]
        im_mask = cv2.drawContours(im_mask, mask, -1,1,-1)
        # rle = mask_util.encode(
        #     np.array(im_mask[:, :, np.newaxis], order='F'))[0]

        label = labels[i]

        mask_results[label].append(im_mask)

    return mask_results
    
# '''bbox and mask 转成result mask要画图'''
def bbox_mask2result(bboxes, masks, labels, num_classes, img_meta):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        masks (Tensor): shape (n, 2, 36)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    ori_shape = img_meta['ori_shape']
    img_h, img_w, _ = ori_shape

    mask_results = [[] for _ in range(num_classes - 1)]

    for i in range(masks.shape[0]):
        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask = [masks[i].transpose(1,0).unsqueeze(1).int().data.cpu().numpy()]
        im_mask = cv2.drawContours(im_mask, mask, -1,1,-1)
        # rle = mask_util.encode(
        #     np.array(im_mask[:, :, np.newaxis], order='F'))[0]

        label = labels[i]

        mask_results[label].append(im_mask)


    if bboxes.shape[0] == 0:
        bbox_results = [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
        return bbox_results, mask_results
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        bbox_results = [bboxes[labels == i, :] for i in range(num_classes - 1)]
        return bbox_results, mask_results

