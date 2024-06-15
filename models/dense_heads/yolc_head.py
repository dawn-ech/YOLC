# Copyright (c) OpenMMLab. All rights reserved.
from operator import gt
import re
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms, soft_nms, DeformConv2d
from mmcv.runner import force_fp32

from mmdet.core import bbox, multi_apply, MlvlPointGenerator
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import (get_local_maximum, transpose_and_gather_feat)
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin

from kornia.filters import gaussian_blur2d


@HEADS.register_module()
class YOLCHead(BaseDenseHead, BBoxTestMixin):
    """YOLC: You Only Look Clusters for Tiny Object Detection in Aerial Images.
    Paper link <https://arxiv.org/abs/2404.06180>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_xywh (dict | None): Config of xywh loss. Default: GWDLoss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_local=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_xywh=dict(type='GWDLoss', loss_weight=2.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(YOLCHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.local_head = self._build_loc_head(in_channel, num_classes)
        self._build_reg_head(in_channel, feat_channel)

        self.loss_center_local = build_loss(loss_center_local)
        self.loss_xywh_coarse = build_loss(loss_xywh)
        self.loss_xywh_refine = build_loss(loss_xywh)
        loss_l1 = dict(type='L1Loss', loss_weight=0.5)
        self.loss_xywh_coarse_l1 = build_loss(loss_l1)
        self.loss_xywh_refine_l1 = build_loss(loss_l1)


        strides=[1]
        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        dcn_base = np.arange(-1, 2).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, 3)
        dcn_base_x = np.tile(dcn_base, 3)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)


        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_loc_head(self, in_channel, out_channel):
        """Build head for high resolution heatmap branch."""
        last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(in_channel, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channel, self.num_classes * 8, 4, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.num_classes * 8, self.num_classes * 8, 4, stride=2, padding=1, output_padding=0),
            nn.Conv2d(self.num_classes * 8, out_channel, kernel_size=1, groups=self.num_classes) # Group Conv
        )
        return last_layer

    def _build_reg_head(self, in_channel, feat_channel):
        """Build head for regression branch."""
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, feat_channel, kernel_size=3, padding=1))
        # branch in xywh_init and bbox_offset
        self.xywh_init = nn.Conv2d(feat_channel, 4, kernel_size=1)
        self.bbox_offset = nn.Conv2d(feat_channel, 18, kernel_size=1)
        self.xywh_refine = DeformConv2d(feat_channel, 4, kernel_size=3, padding=1)


    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        for head in [self.local_head, self.reg_conv, self.xywh_init, self.bbox_offset, self.xywh_refine]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0.0, 0.02)

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=False)
        points_list = [multi_level_points[0].clone() for _ in range(num_imgs)]

        return points_list

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_local_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_local_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_local_pred = self.local_head(feat).sigmoid()
        
        reg_feat = self.reg_conv(feat).contiguous()
        xywh_pred_coarse = self.xywh_init(reg_feat)
        featmap_sizes = [xywh_pred_coarse.size()[-2:]]
        device = xywh_pred_coarse.device
        # target for initial stage
        center_points = self.prior_generator.grid_priors(featmap_sizes, device=device, with_stride=False)[0]
        bbox_pred = xywh_pred_coarse.detach().permute(0, 2, 3, 1).reshape(xywh_pred_coarse.size(0), -1, 4).contiguous()
        #[B, H, W, 4] (x, y, w/2, h/2)
        bbox_pred[:, :, :2] = bbox_pred[:, :, :2] + center_points.unsqueeze(0)    # [B, HxW, 2] + [1, HxW, 2]

        offset = self.bbox_offset(reg_feat).sigmoid()
        dcn_offset = self.gen_dcn_offset(bbox_pred.permute(0, 2, 1), offset, center_points)
        xywh_pred_refine = self.xywh_refine(reg_feat, dcn_offset)
        return center_local_pred, xywh_pred_coarse, xywh_pred_refine
    
    def gen_dcn_offset(self, bbox_pred, offset, center_points):
        '''
            bbox_pred: [B, H, W, 4] [x, y, w/2, h/2] detach
            offset : [B, ?, H, W] require_grad is True
            center_points : [HxW, 2], cordinate of anchor points
            
            Return:
                dcn_offset: [B, ?x2, H, W]
        '''
        B, _, H, W = offset.shape
        dcn_offset = offset.new(B, 9*2, H, W)
        bbox_pred = bbox_pred.view(B, 4, H, W)
        bbox_pred[:, 0:2, :, :,] = bbox_pred[:, 0:2, :, :,] - bbox_pred[:, 2:4, :, :,]
        bbox_pred[:, 2:4, :, :,] = 2 * bbox_pred[:, 2:4, :, :,]
        
        dcn_offset[:, 0::2, :, :] = bbox_pred[:, 0, :, :].unsqueeze(1) + bbox_pred[:, 2, :, :].unsqueeze(1) * offset[:, 0::2, :, :]
        dcn_offset[:, 1::2, :, :] = bbox_pred[:, 1, :, :].unsqueeze(1) + bbox_pred[:, 3, :, :].unsqueeze(1) * offset[:, 1::2, :, :]

        dcn_base_offset = self.dcn_base_offset.type_as(dcn_offset)
        dcn_anchor_offset = center_points.view(H, W, 2).repeat(B, 1, 1, 1).repeat(1, 1, 1, 9).permute(0, 3, 1, 2)
        dcn_anchor_offset += dcn_base_offset
        return dcn_offset - dcn_anchor_offset


    @force_fp32(apply_to=('center_local_preds', 'xywh_preds_coarse', 'xywh_preds_refine'))
    def loss(self,
             center_local_preds,
             xywh_preds_coarse,
             xywh_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_local_preds (list[Tensor]): center predict localization for
               all levels with shape (B, 10, H, W).
            xywh_preds_coarse (list[Tensor]): xywh predicts for all levels with
               shape (B, 4, H, W).
            xywh_preds_refine (list[Tensor]): xywh predicts for all levels with
               shape (B, 4, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_xywh (Tensor): loss of xyhw heatmap
        """
        assert len(center_local_preds) == len(xywh_preds_coarse) == len(xywh_preds_refine) == 1
        center_local_pred = center_local_preds[0]
        xywh_pred_coarse = xywh_preds_coarse[0]
        xywh_pred_refine = xywh_preds_refine[0]

        featmap_sizes = [featmap.size()[-2:] for featmap in xywh_preds_coarse]
        device = xywh_pred_coarse.device

        # target for initial stage
        center_points = self.get_points(featmap_sizes, img_metas, device)[0]

        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                     xywh_pred_coarse.shape,
                                                     img_metas[0]['pad_shape'])

        center_local_target = target_result['center_heatmap_target']
        
        
        xywh_target = target_result['xywh_target']
        xywh_target_weight = target_result['xywh_target_weight']

        # center_list is a list of all images and element is a list of all scales
        # center_point is [HxW, 2]   (x, y)
        B, _, H, W = xywh_pred_coarse.shape
        bbox_pred_coarse = xywh_pred_coarse.permute(0, 2, 3, 1).reshape(B, -1, 4)    #[B, H, W, 4] (x, y, w/2, h/2)
        bbox_pred_coarse[:, :, :2] = bbox_pred_coarse[:, :, :2] + center_points.unsqueeze(0)    # [B, HxW, 2] + [1, HxW, 2]
        
        bbox_pred_refine = xywh_pred_refine.permute(0, 2, 3, 1).reshape(B, -1, 4)    #[B, H, W, 4] (x, y, w/2, h/2)
        bbox_pred_refine[:, :, :2] = bbox_pred_refine[:, :, :2] + center_points.unsqueeze(0)    # [B, HxW, 2] + [1, HxW, 2]
        
        xywh_target = xywh_target.reshape(B, -1, 4)
        xywh_target_weight = xywh_target_weight.reshape(B, -1)

        xywh_l1target_weight = target_result['xywh_l1target_weight']
        xywh_l1target_weight = xywh_l1target_weight.reshape(B, -1, 4)

        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_local(
            center_local_pred, center_local_target, avg_factor=avg_factor)
        loss_xywh_coarse = self.loss_xywh_coarse(
            bbox_pred_coarse,
            xywh_target,
            xywh_target_weight,
            avg_factor=avg_factor)
        loss_xywh_refine = self.loss_xywh_refine(
            bbox_pred_refine,
            xywh_target,
            xywh_target_weight,
            avg_factor=avg_factor)
        loss_xywh_coarse_l1 = self.loss_xywh_coarse_l1(
            bbox_pred_coarse,
            xywh_target,
            xywh_l1target_weight,
            avg_factor=avg_factor)
        loss_xywh_refine_l1 = self.loss_xywh_refine_l1(
            bbox_pred_refine,
            xywh_target,
            xywh_l1target_weight,
            avg_factor=avg_factor)
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_xywh_coarse=loss_xywh_coarse,
            loss_xywh_coarse_l1=loss_xywh_coarse_l1,
            loss_xywh_refine=loss_xywh_refine,
            loss_xywh_refine_l1=loss_xywh_refine_l1)

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - xywh_target (Tensor): targets of xywh predict, shape \
                   (B, 4, H, W).
               - xywh_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 4, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h*4, feat_w*4])
        xywh_target = gt_bboxes[-1].new_zeros([bs, feat_h, feat_w, 4])
        xywh_target_weight = gt_bboxes[-1].new_zeros(
            [bs, feat_h, feat_w])
        xywh_l1target_weight = gt_bboxes[-1].new_zeros(
            [bs, feat_h, feat_w, 4])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)
            origin_center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) / 2
            origin_center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) / 2
            origin_gt_centers = torch.cat((origin_center_x, origin_center_y), dim=1)
            gt_centers_classwise = dict([(i,[]) for i in range(self.num_classes)])

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                box_h = (gt_bbox[j][3] - gt_bbox[j][1])
                box_w = (gt_bbox[j][2] - gt_bbox[j][0])
                radius = gaussian_radius([box_h, box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                ori_ctx_int, ori_cty_int = origin_gt_centers[j].int()
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ori_ctx_int, ori_cty_int], radius)

                if cty_int >= feat_h or ctx_int >= feat_w:
                    continue

                xywh_target[batch_id, cty_int, ctx_int, 0] = ctx
                xywh_target[batch_id, cty_int, ctx_int, 1] = cty
                xywh_target[batch_id, cty_int, ctx_int, 2] = scale_box_w/2
                xywh_target[batch_id, cty_int, ctx_int, 3] = scale_box_h/2
                

                xywh_target_weight[batch_id, cty_int, ctx_int] = 1
                xywh_l1target_weight[batch_id, cty_int, ctx_int, 0:2] = 1.0
                xywh_l1target_weight[batch_id, cty_int, ctx_int, 2:4] = 0.2

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            xywh_target=xywh_target,
            xywh_target_weight=xywh_target_weight,
            xywh_l1target_weight=xywh_l1target_weight)
        return target_result, avg_factor

    def get_bboxes(self,
                   center_heatmap_preds,
                   xywh_preds_init,
                   xywh_preds,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(xywh_preds) == 1
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        border_pixs = [img_meta['border'] for img_meta in img_metas]

        center_heatmap_preds[0] = gaussian_blur2d(center_heatmap_preds[0], (3, 3), sigma=(1, 1))
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            xywh_preds[0],
            img_metas[0]['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        batch_border = batch_det_bboxes.new_tensor(
            border_pixs)[:, [2, 0, 2, 0]].unsqueeze(1)
        batch_det_bboxes[..., :4] -= batch_border


        if rescale:
            batch_det_bboxes[..., :4] /= batch_det_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        if with_nms:
            det_results = []
            for (det_bboxes, det_labels) in zip(batch_det_bboxes,
                                                batch_labels):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,
                                                       self.test_cfg)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
            ]
        return det_results


    def decode_heatmap(self,
                       center_heatmap_pred,
                       xywh_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        
        # center_heatmap_pred [bs, 1, H, W]
        height, width = xywh_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = self.get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets
        
        xywh = transpose_and_gather_feat(xywh_pred, batch_index)


        topk_xs = topk_xs + xywh[..., 0]
        topk_ys = topk_ys + xywh[..., 1]
        tl_x = (topk_xs - xywh[..., 2]) * (inp_w / width)
        tl_y = (topk_ys - xywh[..., 3]) * (inp_h / height)
        br_x = (topk_xs + xywh[..., 2]) * (inp_w / width)
        br_y = (topk_ys + xywh[..., 3]) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def get_topk_from_heatmap(self, center_heatmap, k=20):
        """Get top k positions from heatmap.

        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            k (int): Target number. Default: 20.

        Returns:
            tuple[torch.Tensor]: Scores, indexes, categories and coords of
                topk keypoint. Containing following Tensors:

            - topk_scores (Tensor): Max scores of each topk keypoint.
            - topk_inds (Tensor): Indexes of each topk keypoint.
            - topk_clses (Tensor): Categories of each topk keypoint.
            - topk_ys (Tensor): Y-coord of each topk keypoint.
            - topk_xs (Tensor): X-coord of each topk keypoint.
        """
        # center_heatmap values in list [0, 1]
        # shape [bs, 1, H, W]
        batch, _, height, width = center_heatmap.size()
        topk_scores, topk_inds = torch.topk(center_heatmap.view(batch, -1), k)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = topk_inds // width
        topk_xs = (topk_inds % width).int().float()
        topk_ys = (topk_ys / 4).int().float()
        topk_xs = (topk_xs / 4).int().float()
        topk_inds = width // 4 * topk_ys + topk_xs
        topk_inds = topk_inds.long()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


    def get_local_minimum(self, heat, kernel=3):
        """Extract local minimum pixel with given kernel.

        Args:
            heat (Tensor): Target heatmap.
            kernel (int): Kernel size of max pooling. Default: 3.

        Returns:
            heat (Tensor): A heatmap where local minimum pixels maintain its
                own value and other positions are 0.
        """
        pad = (kernel - 1) // 2
        heat = 1 - torch.div(heat, 10)
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        out_bboxes, keep = batched_nms(bboxes[:, :4].contiguous(),
                                       bboxes[:, -1].contiguous(), labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels


    def simple_test(self, feats, img_metas, rescale=False, crop=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        if crop:
            return self.simple_test_bboxes(feats, img_metas, rescale=rescale, crop=True)
        else:
            return self.simple_test_bboxes(feats, img_metas, rescale=rescale)

    def simple_test_bboxes(self, feats, img_metas, rescale=False, crop=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        if crop:
            return self.LSM(outs[0], img_metas), results_list
        else:
            return results_list

    def LSM(self, center_heatmap_preds, img_metas):
        '''
        Args:
            center_heatmap_preds (list[Tensor]):  (N, C, H, W)
        '''
        center_heatmap_pred = center_heatmap_preds[0]
        locmap = torch.max(center_heatmap_pred, dim=1, keepdim=True)[0].cpu().numpy()
        
        coord = self.findclusters(locmap, find_max=True, fname=["test"])

        '''for visualization'''
        border_pixs = [img_meta['border'] for img_meta in img_metas]
        # coord [x, y, w, h]
        coord[:, 0] = coord[:, 0] - border_pixs[0][2]
        coord[:, 1] = coord[:, 1] - border_pixs[0][0]
        return coord

    def findclusters(self, heatmap, find_max, fname):
        heatmap = 1 - heatmap
        heatmap = 255*heatmap / np.max(heatmap)
        heatmap = heatmap[0][0]

        gray = heatmap.astype(np.uint8)
        Thresh = 10.0/11.0 * 255.0
        ret, binary = cv2.threshold(gray, Thresh, 255, cv2.THRESH_BINARY_INV)

        '''
            16 : 10
        '''
        binmap = binary.copy()
        binmap[binmap==255] = 1
        density_map = np.zeros((16, 10))
        w_stride = binary.shape[1]//16
        h_stride = binary.shape[0]//10
        for i in range(16):
            for j in range(10):
                x1 = w_stride*i
                y1 = h_stride*j
                x2 = min(x1+w_stride, binary.shape[1])
                y2 = min(y1+h_stride, binary.shape[0])
                density_map[i][j] = binmap[y1:y2,x1:x2].sum()

        d = density_map.flatten()
        topk = 15
        idx = d.argsort()[-topk:][::-1]
        grid_idx = idx.copy()
        idx_x = idx // 10 * w_stride
        idx_x = idx_x.reshape((-1, 1))
        idx_y = idx % 10 * h_stride
        idx_y = idx_y.reshape((-1, 1))
        idx = np.concatenate((idx_x, idx_y), axis=1)
        idx_2 = idx.copy()
        idx_2[:,0] = np.clip(idx[:,0]+w_stride, 0, binary.shape[1])
        idx_2[:,1] = np.clip(idx[:,1]+h_stride, 0, binary.shape[0])

        grid = np.zeros((16, 10))
        for item in grid_idx:
            x1 = item // 10
            y1 = item % 10
            grid[x1, y1] = 255
        result = split_overlay_map(grid)
        result = np.array(result)
        result[:,0::2] = np.clip(result[:, 0::2]*w_stride, 0,  binary.shape[1])
        result[:,1::2] = np.clip(result[:, 1::2]*h_stride, 0,  binary.shape[0])
        
        for i in range(len(result)):
            cv2.rectangle(binary, (result[i, 0], result[i, 1]), (result[i, 2], result[i, 3]), (255, 0, 0), 2)

        cv2.imwrite("binary_heatmap_%s4.jpg" %(fname[0]), binary)

        result[:, 2] = result[:, 2] - result[:, 0]
        result[:, 3] = result[:, 3] - result[:, 1]
        return result


def split_overlay_map(grid):
    # This function is modified from https://github.com/Cli98/DMNet
    """
        Conduct eight-connected-component methods on grid to connnect all pixel within the similar region
        :param grid: desnity mask to connect
        :return: merged regions for cropping purpose
    """
    if grid is None or grid[0] is None:
        return 0
    # Assume overlap_map is a 2d feature map
    m, n = grid.shape
    visit = [[0 for _ in range(n)] for _ in range(m)]
    count, queue, result = 0, [], []
    for i in range(m):
        for j in range(n):
            if not visit[i][j]:
                if grid[i][j] == 0:
                    visit[i][j] = 1
                    continue
                queue.append([i, j])
                top, left = float("inf"), float("inf")
                bot, right = float("-inf"), float("-inf")
                while queue:
                    i_cp, j_cp = queue.pop(0)
                    if 0 <= i_cp < m and 0 <= j_cp < n and grid[i_cp][j_cp] == 255:
                        top = min(i_cp, top)
                        left = min(j_cp, left)
                        bot = max(i_cp, bot)
                        right = max(j_cp, right)
                    if 0 <= i_cp < m and 0 <= j_cp < n and not visit[i_cp][j_cp]:
                        visit[i_cp][j_cp] = 1
                        if grid[i_cp][j_cp] == 255:
                            queue.append([i_cp, j_cp + 1])
                            queue.append([i_cp + 1, j_cp])
                            queue.append([i_cp, j_cp - 1])
                            queue.append([i_cp - 1, j_cp])

                            queue.append([i_cp - 1, j_cp - 1])
                            queue.append([i_cp - 1, j_cp + 1])
                            queue.append([i_cp + 1, j_cp - 1])
                            queue.append([i_cp + 1, j_cp + 1])
                count += 1
                result.append([max(0, top), max(0, left), min(bot+1, m), min(right+1, n)])

    return result
