import copy

import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_max

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .target_assigner.hungarian_assigner import HungarianMatcher3d, generalized_box3d_iou, \
    box_cxcyczlwh_to_xyxyxy
from ..model_utils.cdn import prepare_for_cdn, dn_post_process
from ..model_utils.seed_transformer import SEEDTransformer, MLP, get_clones


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is not None:
            not_mask = ~mask
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            size_h, size_w = x.shape[-2:]
            y_embed = torch.arange(1, size_h + 1, dtype=x.dtype, device=x.device)
            x_embed = torch.arange(1, size_w + 1, dtype=x.dtype, device=x.device)
            y_embed, x_embed = torch.meshgrid(y_embed, x_embed)
            x_embed = x_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)
            y_embed = y_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * dim_t.div(2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class Det3DHead(nn.Module):
    def __init__(self, hidden_dim, num_classes=3, code_size=7, num_layers=1):
        super().__init__()
        class_embed = MLP(hidden_dim, hidden_dim, num_classes, 3)
        bbox_embed = MLP(hidden_dim, hidden_dim, code_size, 3)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        class_embed.layers[-1].bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)

        self.class_embed = get_clones(class_embed, num_layers)
        self.bbox_embed = get_clones(bbox_embed, num_layers)

        iou_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        nn.init.constant_(iou_embed.layers[-1].weight.data, 0)
        nn.init.constant_(iou_embed.layers[-1].bias.data, 0)
        self.iou_embed = get_clones(iou_embed, num_layers)

    def forward(self, embed, anchors, layer_idx=0):
        cls_logits = self.class_embed[layer_idx](embed)
        box_coords = (self.bbox_embed[layer_idx](embed) + inverse_sigmoid(anchors)).sigmoid()
        pred_iou = (self.iou_embed[layer_idx](embed)).clamp(-1, 1)
        return cls_logits, box_coords, pred_iou


class MaskPredictor(nn.Module):
    def __init__(self, hidden_dim):
        super(MaskPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.reshape([b, c, -1]).permute(0, 2, 1)
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.hidden_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        out = out.reshape([b, h, w])
        return out


class SEEDHead(nn.Module):
    def __init__(
            self,
            model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
            predict_boxes_when_training=True,
    ):
        super(SEEDHead, self).__init__()

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = model_cfg.FEATURE_MAP_STRIDE
        self.num_classes = num_class

        self.model_cfg = model_cfg

        self.hidden_channel = self.model_cfg.HIDDEN_CHANNEL
        self.num_queries = self.model_cfg.NUM_QUERIES
        self.aux_loss = self.model_cfg.LOSS_CONFIG.AUX_LOSS
        self.keep_ratio = self.model_cfg.KEEP_RATIO
        self.iou_cls = self.model_cfg.IOU_CLS
        self.iou_rectifier = self.model_cfg.IOU_RECTIFIER

        num_heads = self.model_cfg.NUM_HEADS
        dropout = self.model_cfg.DROPOUT
        activation = self.model_cfg.ACTIVATION
        ffn_channel = self.model_cfg.FFN_CHANNEL
        num_decoder_layers = self.model_cfg.NUM_DECODER_LAYERS
        self.code_size = self.model_cfg.CODE_SIZE

        cp_flag = self.model_cfg.CP

        self.dn = self.model_cfg.DN

        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_channel, kernel_size=1),
            nn.GroupNorm(32, self.hidden_channel),
        )

        self.pos_embed = PositionEmbeddingSine(self.hidden_channel // 2)

        for module in self.input_proj.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.mask_predictor = MaskPredictor(self.hidden_channel)

        self.transformer = SEEDTransformer(
            d_model=self.hidden_channel,
            nhead=num_heads,
            nlevel=1,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ffn_channel,
            dropout=dropout,
            activation=activation,
            num_queries=self.num_queries,
            keep_ratio=self.keep_ratio,
            code_size=self.code_size,
            iou_rectifier=self.iou_rectifier,
            iou_cls=self.iou_cls,
            num_classes=num_class,
            cp_flag=cp_flag
        )

        self.transformer.proposal_head = Det3DHead(
            self.hidden_channel,
            code_size=self.code_size,
            num_classes=num_class,
            num_layers=1,
        )
        self.transformer.decoder.detection_head = Det3DHead(
            self.hidden_channel,
            code_size=self.code_size,
            num_classes=num_class,
            num_layers=num_decoder_layers,
        )

        if self.training and self.dn.enabled:
            contras_dim = self.model_cfg.CONTRASTIVE.dim
            self.eqco = self.model_cfg.CONTRASTIVE.eqco
            self.tau = self.model_cfg.CONTRASTIVE.tau
            self.contras_loss_coeff = self.model_cfg.CONTRASTIVE.loss_coeff
            self.projector = nn.Sequential(
                nn.Linear(self.code_size + self.num_classes, contras_dim),
                nn.ReLU(),
                nn.Linear(contras_dim, contras_dim),
            )
            self.predictor = nn.Sequential(
                nn.Linear(contras_dim, contras_dim),
                nn.ReLU(),
                nn.Linear(contras_dim, contras_dim),
            )
            self.similarity_f = nn.CosineSimilarity(dim=2)

            self.transformer.decoder_gt = copy.deepcopy(self.transformer.decoder)
            for param_q, param_k in zip(self.transformer.decoder.parameters(),
                                        self.transformer.decoder_gt.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        self.assigner = HungarianMatcher3d(
            cost_class=self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER.cls_cost,
            cost_bbox=self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER.bbox_cost,
            cost_giou=self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER.iou_cost,
            cost_rad=self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER.rad_cost,
            decode_bbox_func=self.decode_bbox,
            iou_cls=self.iou_cls
        )

        weight_dict = {
            "loss_ce": self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER.cls_cost,
            "loss_bbox": self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER.bbox_cost,
            "loss_giou": self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER.iou_cost,
            "loss_rad": self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER.rad_cost,
        }
        losses = ["focal_labels", "boxes"]
        self.losses = Det3DLoss(
            matcher=self.assigner,
            weight_dict=weight_dict,
            losses=losses,
            decode_func=self.decode_bbox
        )

        # setup aux loss weight
        if self.aux_loss:
            aux_weight_dict = {}
            if hasattr(self.losses, "weight_dict"):
                aux_weight_dict.update({k + f"_dn": v for k, v in self.losses.weight_dict.items()})
                for i in range(num_decoder_layers - 1):
                    aux_weight_dict.update({k + f"_dn_{i}": v for k, v in self.losses.weight_dict.items()})
                    aux_weight_dict.update({k + f"_{i}": v for k, v in self.losses.weight_dict.items()})
                self.losses.weight_dict.update(aux_weight_dict)

    def predict(self, batch_dict):
        batch_size = batch_dict['batch_size']
        spatial_features_2d = batch_dict['spatial_features_2d']

        features = []
        pos_encodings = []
        features.append(self.input_proj(spatial_features_2d))
        pos_encodings.append(self.pos_embed(spatial_features_2d))

        score_mask = self.mask_predictor(features[0])

        dn = self.dn
        if self.training and dn.enabled and dn.dn_number > 0:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[..., :-1]
            gt_labels_3d = gt_boxes[..., -1].long() - 1

            targets = list()
            for batch_idx in range(len(gt_bboxes_3d)):
                target = {}
                gt_bboxes = gt_bboxes_3d[batch_idx]
                valid_idx = []
                # filter empty boxes
                for i in range(len(gt_bboxes)):
                    if gt_bboxes[i][3] > 0 and gt_bboxes[i][4] > 0:
                        valid_idx.append(i)
                target['gt_boxes'] = self.encode_bbox(gt_bboxes[valid_idx])
                target['labels'] = gt_labels_3d[batch_idx][valid_idx]
                targets.append(target)

            input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_cdn(
                dn_args=(targets, dn.dn_number, dn.dn_label_noise_ratio, dn.dn_box_noise_scale),
                training=self.training,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_channel,
                label_enc=None,
                code_size=self.code_size,
            )
        else:
            input_query_bbox = input_query_label = attn_mask = dn_meta = None
            targets = None

        outputs = self.transformer(
            features,
            pos_encodings,
            input_query_bbox,
            input_query_label,
            attn_mask,
            targets=targets,
            score_mask=score_mask,
        )

        hidden_state, init_reference, inter_references, src_embed, src_ref_windows, src_indexes = outputs

        # decoder
        outputs_classes = []
        outputs_coords = []
        outputs_ious = []
        for idx in range(hidden_state.shape[0]):
            if idx == 0:
                reference = init_reference
            else:
                reference = inter_references[idx - 1]
            outputs_class, outputs_coord, outputs_iou = self.transformer.decoder.detection_head(hidden_state[idx],
                                                                                                reference, idx)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_ious.append(outputs_iou)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_iou = torch.stack(outputs_ious)

        # dn post process
        if dn.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord, outputs_iou = dn_post_process(
                outputs_class,
                outputs_coord,
                outputs_iou,
                dn_meta,
                self.aux_loss,
                self._set_aux_loss,
            )

        # only for supervision
        dqs_outputs = None
        if self.training:
            dqs_class, dqs_coords, dqs_ious = self.transformer.proposal_head(src_embed, src_ref_windows)
            dqs_outputs = {
                'topk_indexes': src_indexes,
                'pred_logits': dqs_class,
                'pred_boxes': dqs_coords,
                'pred_ious': dqs_ious
            }

        # compute decoder losses
        outputs = {
            "pred_scores_mask": score_mask,
            "pred_logits": outputs_class[-1][:, : self.num_queries],
            "pred_boxes": outputs_coord[-1][:, : self.num_queries],
            'pred_ious': outputs_iou[-1][:, : self.num_queries],
            "aux_outputs": self._set_aux_loss(
                outputs_class[:-1, :, : self.num_queries], outputs_coord[:-1, :, : self.num_queries],
                outputs_iou[:-1, :, : self.num_queries],
            ),
        }
        if self.training:
            pred_dicts = dict(dqs_outputs=dqs_outputs, outputs=outputs)
            return pred_dicts, outputs_class, outputs_coord, outputs_iou, dn_meta
        else:
            pred_dicts = dict(dqs_outputs=dqs_outputs, outputs=outputs)
            return pred_dicts

    def forward(self, batch_dict):
        if self.training:
            pred_dicts, outputs_class, outputs_coord, outputs_iou, dn_meta = self.predict(batch_dict)
        else:
            pred_dicts = self.predict(batch_dict)

        if not self.training:
            bboxes = self.get_bboxes(pred_dicts)
            batch_dict['final_box_dicts'] = bboxes
        else:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[..., :-1]
            gt_labels_3d = gt_boxes[..., -1].long() - 1

            loss, tb_dict = self.loss(gt_bboxes_3d, gt_labels_3d, pred_dicts, dn_meta, outputs_class, outputs_coord,
                                      outputs_iou)
            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = tb_dict
        return batch_dict

    def get_bboxes(self, pred_dicts):
        outputs = pred_dicts['outputs']
        out_logits = outputs['pred_logits']
        out_bbox = outputs['pred_boxes']
        out_iou = (outputs['pred_ious'] + 1) / 2
        batch_size = out_logits.shape[0]

        out_iou = out_iou.repeat([1, 1, out_logits.shape[-1]])
        out_iou = out_iou.view(out_logits.shape[0], -1)

        out_prob = out_logits.sigmoid()
        out_prob = out_prob.view(out_logits.shape[0], -1)
        out_bbox = self.decode_bbox(out_bbox)

        def _process_output(indices, bboxes):
            topk_boxes = indices.div(out_logits.shape[2], rounding_mode="floor").unsqueeze(-1)
            labels = indices % out_logits.shape[2]
            boxes = torch.gather(bboxes, 0, topk_boxes.repeat(1, out_bbox.shape[-1]))
            return labels + 1, boxes, topk_boxes

        new_ret_dict = []
        for i in range(batch_size):
            out_prob_i = out_prob[i]
            out_bbox_i = out_bbox[i]

            out_iou_i = out_iou[i]

            # out_prob_i_ori = out_prob_i.view(out_bbox_i.shape[0], -1)  # [1000, 3]
            # max_out_prob_i, pred_cls = torch.max(out_prob_i_ori, dim=-1)
            # out_iou_i_ori = out_iou_i.view(out_bbox_i.shape[0], -1)
            #
            # out_prob_i_list = []
            # out_bbox_i_list = []
            # out_iou_i_list = []
            #
            # ONLY_FOR_CAR = True
            #
            # for cls in range(self.num_classes):
            #     cls_mask = pred_cls == cls
            #     if cls_mask.sum() >= 1:
            #         out_prob_i_cls_valid = out_prob_i_ori[cls_mask]
            #         out_bbox_i_cls_valid = out_bbox_i[cls_mask]
            #         out_iou_i_cls_valid = out_iou_i_ori[cls_mask]
            #         max_out_prob_i_cls_valid = max_out_prob_i[cls_mask]
            #
            #         if ONLY_FOR_CAR & (cls == 0):  # 0 is for CAR
            #             out_bbox_i_xy = (out_bbox_i_cls_valid[:, :2] - torch.Tensor(
            #                 self.point_cloud_range[0:2]).unsqueeze(0).type_as(
            #                 out_bbox_i_cls_valid)) * 4  # 4 for large objects, such as car
            #             out_bbox_i_xy = torch.round(out_bbox_i_xy).int()
            #             sort_xy = out_bbox_i_xy[:, 0] * 1000 + out_bbox_i_xy[:, 1]
            #             unq_coords, unq_inv, unq_cnt = torch.unique(sort_xy, return_inverse=True, return_counts=True,
            #                                                         dim=0)
            #
            #             out, valid_mask = scatter_max(max_out_prob_i_cls_valid, unq_inv)
            #
            #             out_prob_i_tmp = out_prob_i_cls_valid[valid_mask].view(-1)
            #             out_bbox_i_tmp = out_bbox_i_cls_valid[valid_mask]
            #             out_iou_i_tmp = out_iou_i_cls_valid[valid_mask].view(-1)
            #
            #         else:
            #             out_bbox_i_xy = (out_bbox_i_cls_valid[:, :2] - torch.Tensor(
            #                 self.point_cloud_range[0:2]).unsqueeze(0).type_as(
            #                 out_bbox_i_cls_valid)) * 20
            #             out_bbox_i_xy = torch.round(out_bbox_i_xy).int()
            #             sort_xy = out_bbox_i_xy[:, 0] * 1000 + out_bbox_i_xy[:, 1]
            #             unq_coords, unq_inv, unq_cnt = torch.unique(sort_xy, return_inverse=True, return_counts=True,
            #                                                         dim=0)
            #
            #             out, valid_mask = scatter_max(max_out_prob_i_cls_valid, unq_inv)
            #
            #             out_prob_i_tmp = out_prob_i_cls_valid[valid_mask].view(-1)
            #             out_bbox_i_tmp = out_bbox_i_cls_valid[valid_mask]
            #             out_iou_i_tmp = out_iou_i_cls_valid[valid_mask].view(-1)
            #
            #         out_prob_i_list.append(out_prob_i_tmp)
            #         out_bbox_i_list.append(out_bbox_i_tmp)
            #         out_iou_i_list.append(out_iou_i_tmp)
            #
            # out_prob_i = torch.cat(out_prob_i_list, dim=0)
            # out_bbox_i = torch.cat(out_bbox_i_list, dim=0)
            # out_iou_i = torch.cat(out_iou_i_list, dim=0)

            topk_indices_i = torch.nonzero(out_prob_i >= 0.1, as_tuple=True)[0]
            scores = out_prob_i[topk_indices_i]

            labels, boxes, topk_indices = _process_output(topk_indices_i.view(-1), out_bbox_i)

            ious = out_iou_i[topk_indices_i]

            scores_list = list()
            labels_list = list()
            boxes_list = list()

            for c in range(self.num_classes):
                mask = (labels - 1) == c
                scores_temp = scores[mask]
                ious_temp = ious[mask]
                labels_temp = labels[mask]
                boxes_temp = boxes[mask]

                if c in self.iou_cls:
                    if isinstance(self.iou_rectifier, list):
                        iou_rectifier = torch.tensor(self.iou_rectifier).to(out_prob)[c]
                        scores_temp = torch.pow(scores_temp, 1 - iou_rectifier) * torch.pow(ious_temp,
                                                                                            iou_rectifier)
                    elif isinstance(self.iou_rectifier, float):
                        scores_temp = torch.pow(scores_temp, 1 - self.iou_rectifier) * torch.pow(ious_temp,
                                                                                                 self.iou_rectifier)
                    else:
                        raise TypeError('only list or float')


                scores_list.append(scores_temp)
                labels_list.append(labels_temp)
                boxes_list.append(boxes_temp)

            scores = torch.cat(scores_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
            boxes = torch.cat(boxes_list, dim=0)
            ret = dict(pred_boxes=boxes, pred_scores=scores, pred_labels=labels)
            new_ret_dict.append(ret)

        return new_ret_dict

    def compute_losses(self, outputs, targets, dn_meta=None):
        loss_dict = self.losses(outputs, targets, dn_meta=dn_meta)

        weight_dict = self.losses.weight_dict
        for k, v in loss_dict.items():
            if k in weight_dict:
                loss_dict[k] = v * weight_dict[k]

        return loss_dict

    def compute_score_losses(self, pred_scores_mask, gt_bboxes_3d, foreground_mask):
        gt_bboxes_3d = copy.deepcopy(gt_bboxes_3d)
        grid_size = torch.ceil(torch.from_numpy(self.grid_size).to(gt_bboxes_3d) / self.feature_map_stride)
        pc_range = torch.from_numpy(self.point_cloud_range).to(gt_bboxes_3d)
        stride = (pc_range[3:5] - pc_range[0:2]) / grid_size[0:2]
        gt_score_map = list()
        yy, xx = torch.meshgrid(torch.arange(grid_size[1]), torch.arange(grid_size[0]))
        points = torch.stack([yy, xx]).permute(1, 2, 0).flip(-1)
        points = torch.cat([points, torch.ones_like(points[..., 0:1]) * 0.5], dim=-1).reshape([-1, 3])
        for i in range(len(gt_bboxes_3d)):
            boxes = gt_bboxes_3d[i]
            boxes = boxes[(boxes[:, 3] > 0) & (boxes[:, 4] > 0)]
            ones = torch.ones_like(boxes[:, 0:1])
            bev_boxes = torch.cat([boxes[:, 0:2], ones * 0.5, boxes[:, 3:5], ones * 0.5, boxes[:, 6:7]], dim=-1)
            bev_boxes[:, 0:2] -= pc_range[0:2]
            bev_boxes[:, 0:2] /= stride
            bev_boxes[:, 3:5] /= stride

            box_ids = roiaware_pool3d_utils.points_in_boxes_gpu(
                points[:, 0:3].unsqueeze(dim=0).float().cuda(),
                bev_boxes[:, 0:7].unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()
            box_ids = box_ids.reshape([grid_size[1].long(), grid_size[0].long()])
            mask = torch.from_numpy(box_ids != -1).to(bev_boxes)
            gt_score_map.append(mask)
        gt_score_map = torch.stack(gt_score_map)

        num_pos = max(gt_score_map.eq(1).float().sum().item(), 1)
        loss_score = ClassificationLoss.sigmoid_focal_loss(
            pred_scores_mask.flatten(0),
            gt_score_map.flatten(0),
            0.25,
            gamma=2.0,
            reduction="sum",
        )
        loss_score /= num_pos

        return loss_score

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, dn_meta=None, outputs_class=None, outputs_coord=None,
             outputs_iou=None):
        loss_all = 0
        loss_dict = dict()
        targets = list()

        for batch_idx in range(len(gt_bboxes_3d)):
            target = {}
            gt_bboxes = gt_bboxes_3d[batch_idx]
            valid_idx = []
            # filter empty boxes
            for i in range(len(gt_bboxes)):
                if gt_bboxes[i][3] > 0 and gt_bboxes[i][4] > 0:
                    valid_idx.append(i)
            target['gt_boxes'] = self.encode_bbox(gt_bboxes[valid_idx])
            target['labels'] = gt_labels_3d[batch_idx][valid_idx]
            targets.append(target)

        dqs_outputs = pred_dicts['dqs_outputs']
        bin_targets = copy.deepcopy(targets)
        [tgt["labels"].fill_(0) for tgt in bin_targets]
        dqs_losses = self.compute_losses(dqs_outputs, bin_targets)
        for k, v in dqs_losses.items():
            loss_all += v
            loss_dict.update({k + "_dqs": v.item()})

        outputs = pred_dicts['outputs']
        dec_losses = self.compute_losses(outputs, targets, dn_meta)
        for k, v in dec_losses.items():
            loss_all += v
            loss_dict.update({k: v.item()})

        # compute contrastive loss
        if dn_meta is not None:
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets]
            max_gt = max(per_gt_num)
            num_gts = sum(per_gt_num)
            if num_gts > 0:
                for li in range(self.model_cfg.NUM_DECODER_LAYERS):
                    contrastive_loss = 0.0
                    projs = torch.cat((outputs_class[li], outputs_coord[li]), dim=-1)
                    gt_projs = self.projector(projs[:, self.num_queries:].detach())
                    pred_projs = self.predictor(self.projector(projs[:, : self.num_queries]))
                    # num_gts x num_locs

                    pos_idxs = list(range(1, dn_meta["num_dn_group"] + 1))
                    for bi, idx in enumerate(outputs["matched_indices"]):
                        sim_matrix = (
                                self.similarity_f(
                                    gt_projs[bi].unsqueeze(1),
                                    pred_projs[bi].unsqueeze(0),
                                )
                                / self.tau
                        )
                        matched_pairs = torch.stack(idx, dim=-1)
                        neg_mask = projs.new_ones(self.num_queries).bool()
                        neg_mask[matched_pairs[:, 0]] = False
                        for pair in matched_pairs:
                            pos_mask = torch.tensor([int(pair[1] + max_gt * pi) for pi in pos_idxs],
                                                    device=sim_matrix.device)
                            pos_pair = sim_matrix[pos_mask, pair[0]].view(-1, 1)
                            neg_pairs = sim_matrix[:, neg_mask][pos_mask]
                            loss_gti = (
                                    torch.log(torch.exp(pos_pair) + torch.exp(neg_pairs).sum(dim=-1, keepdim=True))
                                    - pos_pair
                            )
                            contrastive_loss += loss_gti.mean()
                    loss_contrastive_dec_li = self.contras_loss_coeff * contrastive_loss / num_gts
                    loss_all += loss_contrastive_dec_li
                    loss_dict.update({'loss_contrastive_dec_' + str(li): loss_contrastive_dec_li.item()})

        pred_scores_mask = outputs['pred_scores_mask']
        loss_score = self.compute_score_losses(pred_scores_mask, gt_bboxes_3d, None)
        loss_all += loss_score
        loss_dict.update({'loss_score': loss_score.item()})

        return loss_all, loss_dict

    def encode_bbox(self, bboxes):
        z_normalizer = 10
        targets = torch.zeros([bboxes.shape[0], self.code_size]).to(bboxes.device)
        targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (
                self.point_cloud_range[3] - self.point_cloud_range[0])
        targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (
                self.point_cloud_range[4] - self.point_cloud_range[1])
        targets[:, 2] = (bboxes[:, 2] + z_normalizer) / (2 * z_normalizer)
        targets[:, 3] = bboxes[:, 3] / (self.point_cloud_range[3] - self.point_cloud_range[0])
        targets[:, 4] = bboxes[:, 4] / (self.point_cloud_range[4] - self.point_cloud_range[1])
        targets[:, 5] = bboxes[:, 5] / (2 * z_normalizer)
        targets[:, 6] = (bboxes[:, 6] + np.pi) / (np.pi * 2)
        if self.code_size > 7:
            targets[:, 7] = (bboxes[:, 7]) / (self.point_cloud_range[3] - self.point_cloud_range[0])
            targets[:, 8] = (bboxes[:, 8]) / (self.point_cloud_range[4] - self.point_cloud_range[1])

        return targets

    def decode_bbox(self, pred_boxes):
        z_normalizer = 10
        pred_boxes[..., 0] = pred_boxes[..., 0] * (self.point_cloud_range[3] - self.point_cloud_range[0]) + \
                             self.point_cloud_range[0]
        pred_boxes[..., 1] = pred_boxes[..., 1] * (self.point_cloud_range[4] - self.point_cloud_range[1]) + \
                             self.point_cloud_range[1]
        pred_boxes[..., 2] = pred_boxes[..., 2] * 2 * z_normalizer + -1 * z_normalizer
        pred_boxes[..., 3] = pred_boxes[..., 3] * (self.point_cloud_range[3] - self.point_cloud_range[0])
        pred_boxes[..., 4] = pred_boxes[..., 4] * (self.point_cloud_range[4] - self.point_cloud_range[1])
        pred_boxes[..., 5] = pred_boxes[..., 5] * 2 * z_normalizer
        pred_boxes[..., -1] = pred_boxes[..., -1] * np.pi * 2 - np.pi
        if self.code_size > 7:
            pred_boxes[:, 7] = (pred_boxes[:, 7]) * (self.point_cloud_range[3] - self.point_cloud_range[0])
            pred_boxes[:, 8] = (pred_boxes[:, 8]) * (self.point_cloud_range[4] - self.point_cloud_range[1])
        return pred_boxes

    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_iou):
        return [{"pred_logits": a, "pred_boxes": b, "pred_ious": c} for a, b, c in
                zip(outputs_class, outputs_coord, outputs_iou)]


class ClassificationLoss(nn.Module):
    def __init__(self, focal_alpha):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.target_classes = None
        self.src_logits = None

    @staticmethod
    def sigmoid_focal_loss(
            logits,
            targets,
            alpha: float = -1,
            gamma: float = 2,
            reduction: str = "none",
    ):
        p = torch.sigmoid(logits)

        ce_loss = F.binary_cross_entropy(p, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def forward(self, outputs, targets, indices, num_boxes):
        outputs["matched_indices"] = indices
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        target_classes_onehot = torch.zeros_like(src_logits)

        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # 0, 1, 2

        # for metrics calculation
        self.target_classes = target_classes_o

        if "topk_indexes" in outputs.keys():
            topk_indexes = outputs["topk_indexes"]
            self.src_logits = torch.gather(src_logits, 1, topk_indexes.expand(-1, -1, src_logits.shape[-1]))[idx]
            target_classes_onehot[idx[0], topk_indexes[idx].squeeze(-1), target_classes_o] = 1
        else:
            self.src_logits = src_logits[idx]
            # 0 for bg, 1 for fg
            # N, L, C
            target_classes_onehot[idx[0], idx[1], target_classes_o] = 1

        loss_ce = (
                self.sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    alpha=self.focal_alpha,
                    gamma=2.0,
                    reduction="sum"
                )
                / num_boxes
        )

        losses = {
            "loss_ce": loss_ce,
        }

        return losses


class RegressionLoss(nn.Module):
    def __init__(self, decode_func=None):
        super().__init__()
        self.decode_func = decode_func

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = _get_src_permutation_idx(indices)

        if "topk_indexes" in outputs.keys():
            pred_boxes = torch.gather(
                outputs["pred_boxes"],
                1,
                outputs["topk_indexes"].expand(-1, -1, outputs["pred_boxes"].shape[-1]),
            )
            pred_ious = torch.gather(
                outputs["pred_ious"],
                1,
                outputs["topk_indexes"].expand(-1, -1, outputs["pred_ious"].shape[-1]),
            )

        else:
            pred_boxes = outputs["pred_boxes"]
            pred_ious = outputs["pred_ious"]

        target_boxes = torch.cat([t["gt_boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        src_boxes, src_rads = pred_boxes[idx].split(6, dim=-1)
        target_boxes, target_rads = target_boxes.split(6, dim=-1)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_rad = F.l1_loss(src_rads, target_rads, reduction="none")

        gt_iou = torch.diag(
            generalized_box3d_iou(
                box_cxcyczlwh_to_xyxyxy(src_boxes),
                box_cxcyczlwh_to_xyxyxy(target_boxes),
            )
        )

        loss_giou = 1 - gt_iou

        losses = {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": loss_giou.sum() / num_boxes,
            "loss_rad": loss_rad.sum() / num_boxes,
        }

        pred_ious = pred_ious[idx]
        box_preds = self.decode_func(torch.cat([src_boxes, src_rads], dim=-1))
        box_target = self.decode_func(torch.cat([target_boxes, target_rads], dim=-1))
        iou_target = iou3d_nms_utils.paired_boxes_iou3d_gpu(box_preds, box_target)
        iou_target = iou_target * 2 - 1
        iou_target = iou_target.detach()
        loss_iou = F.l1_loss(pred_ious, iou_target.unsqueeze(-1), reduction="none")
        losses.update({"loss_iou": loss_iou.sum() / num_boxes})

        return losses


class Det3DLoss(nn.Module):
    def __init__(self, matcher, weight_dict, losses, decode_func):
        super().__init__()

        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        self.det3d_losses = nn.ModuleDict()
        for loss in losses:
            if loss == "boxes":
                self.det3d_losses[loss] = RegressionLoss(decode_func=decode_func)
            elif loss == "focal_labels":
                self.det3d_losses[loss] = ClassificationLoss(0.25)
            else:
                raise ValueError(f"Only boxes|focal_labels are supported for det3d losses. Found {loss}")

    @staticmethod
    def get_world_size() -> int:
        if not dist.is_available():
            return 1
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()

    def get_target_classes(self):
        for k in self.det3d_losses.keys():
            if "labels" in k:
                return self.det3d_losses[k].src_logits, self.det3d_losses[k].target_classes

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
        num_dn_groups, pad_size = dn_meta["num_dn_group"], dn_meta["pad_size"]
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups

    def forward(self, outputs, targets, dn_meta=None):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum([len(t["labels"]) for t in targets])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if self.get_world_size() > 1:
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / self.get_world_size(), min=1).item()

        losses = {}

        if dn_meta is not None:
            # prepare for computing denosing loss
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["labels"]) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            l_dict = {}
            for loss in self.losses:
                l_dict.update(
                    self.det3d_losses[loss](
                        output_known_lbs_bboxes,
                        targets,
                        dn_pos_idx,
                        num_boxes * scalar,
                    )
                )
            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.det3d_losses[loss](aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if dn_meta is not None:
                    aux_outputs_known = output_known_lbs_bboxes["aux_outputs"][i]
                    l_dict = {}
                    for loss in self.losses:
                        l_dict.update(
                            self.det3d_losses[loss](
                                aux_outputs_known,
                                targets,
                                dn_pos_idx,
                                num_boxes * scalar,
                            )
                        )

                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        for loss in self.losses:
            losses.update(self.det3d_losses[loss](outputs, targets, indices, num_boxes))

        return losses


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx
