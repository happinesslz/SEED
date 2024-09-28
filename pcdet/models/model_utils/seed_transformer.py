import collections
import copy

import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn

from pcdet.models.model_utils.box_attention import Box3dAttention


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableBox3dAttention(Box3dAttention):
    def __init__(self, d_model, num_level, num_head, with_rotation=True, kernel_size=5):
        super(DeformableBox3dAttention, self).__init__(d_model=d_model, num_level=num_level, num_head=num_head,
                                                       with_rotation=with_rotation, kernel_size=kernel_size)

        self.sampling_offsets = nn.Linear(d_model, num_head * num_level * self.num_point * 2)

        self._create_kernel_indices(kernel_size, "kernel_indices")
        self._reset_deformable_parameters()

    def _reset_deformable_parameters(self):
        thetas = torch.arange(self.num_head, dtype=torch.float32) * (
                2.0 * math.pi / self.num_head
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_head, 1, 1, 2)
            .repeat(1, self.num_level, self.num_point, 1)
        )
        for i in range(self.num_point):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

    def _where_to_attend(self, query, v_valid_ratios, ref_windows, h_size=188.0):
        B, L = ref_windows.shape[:2]

        offset_boxes = self.linear_box(query)
        offset_boxes = offset_boxes.view(B, L, self.num_head, self.num_level, self.num_variable)

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
        else:
            ref_windows = ref_windows.unsqueeze(3)

        ref_boxes = ref_windows[..., [0, 1, 3, 4]]
        ref_angles = ref_windows[..., [6]]

        if self.with_rotation:
            offset_boxes, offset_angles = offset_boxes.split(4, dim=-1)
            angles = (ref_angles + offset_angles / 16) * 2 * math.pi
        else:
            angles = ref_angles.expand(B, L, self.num_head, self.num_level, 1)

        boxes = ref_boxes + offset_boxes / 8 * ref_boxes[..., [2, 3, 2, 3]]
        center, size = boxes.unsqueeze(-2).split(2, dim=-1)

        cos_angle, sin_angle = torch.cos(angles), torch.sin(angles)
        rot_matrix = torch.stack([cos_angle, -sin_angle, sin_angle, cos_angle], dim=-1)
        rot_matrix = rot_matrix.view(B, L, self.num_head, self.num_level, 1, 2, 2)

        sampling_offsets = self.sampling_offsets(query).view(
            B, L, self.num_head, self.num_level, self.num_point, 2
        )
        deformable_grid = sampling_offsets / h_size

        fixed_grid = self.kernel_indices * torch.relu(size)
        fixed_grid = center + (fixed_grid.unsqueeze(-2) * rot_matrix).sum(-1)

        grid = fixed_grid + deformable_grid

        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios

        return grid.contiguous()


class SEEDTransformer(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            nlevel=4,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            code_size=7,
            num_queries=300,
            keep_ratio=0.5,
            iou_rectifier=None,
            iou_cls=None,
            cp_flag=False,
            num_classes=3,
            mom=0.999,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.code_size = code_size
        self.iou_rectifier = iou_rectifier
        self.iou_cls = iou_cls
        self.m = mom

        self.dga_layer = DGALayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation, keep_ratio)

        decoder_layer = SEEDDecoderLayer(d_model, nhead, nlevel, dim_feedforward, dropout, activation,
                                         code_size=code_size, num_classes=num_classes)
        self.decoder = SEEDDecoder(d_model, decoder_layer, num_decoder_layers, cp_flag)

    def _create_ref_windows(self, tensor_list):
        device = tensor_list[0].device

        ref_windows = []
        for tensor in tensor_list:
            B, _, H, W = tensor.shape
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_xy = torch.stack((ref_x, ref_y), -1)
            ref_wh = torch.ones_like(ref_xy) * 0.025
            placeholder = torch.zeros_like(ref_xy)[..., :1]
            ref_box = torch.cat((ref_xy, placeholder + 0.5, ref_wh, placeholder + 0.5, placeholder), -1)
            extra = self.code_size - ref_box.shape[-1]
            ref_box = torch.cat([ref_box, torch.zeros_like(ref_box[..., 0:extra])], dim=-1)
            ref_box = ref_box.expand(B, -1, -1)
            ref_windows.append(ref_box)
        ref_windows = torch.cat(ref_windows, dim=1)
        return ref_windows

    def _quality_query_selection(self, enc_embed, ref_windows, indexes=None):
        B, L = enc_embed.shape[:2]
        out_logits, out_ref_windows, out_ious = self.proposal_head(enc_embed, ref_windows)

        out_logits_max, out_labels = out_logits.max(dim=-1, keepdim=True)
        out_probs = out_logits_max[..., 0].sigmoid()
        out_labels = out_labels[..., 0]

        # if not self.training:
        ###
        mask = torch.ones_like(out_probs)
        for i in range(out_logits.shape[-1]):
            if i not in self.iou_cls:
                mask[out_labels == i] = 0.0

        score_mask = out_probs > 0.3
        mask = mask * score_mask.type_as(mask)

        if isinstance(self.iou_rectifier, list):
            out_ious = (out_ious + 1) / 2
            iou_rectifier = torch.tensor(self.iou_rectifier).to(out_probs)
            temp_probs = torch.pow(out_probs, 1 - iou_rectifier[out_labels]) * torch.pow(
                out_ious[..., 0], iou_rectifier[out_labels])
            out_probs = out_probs * (1 - mask) + mask * temp_probs

        elif isinstance(self.iou_rectifier, float):
            temp_probs = torch.pow(out_probs, 1 - self.iou_rectifier) * torch.pow(out_ious, self.iou_rectifier)
            out_probs = out_probs * (1 - mask) + mask * temp_probs
        else:
            raise TypeError('only list or float')
        #

        topk_probs, indexes = torch.topk(out_probs, self.num_queries, dim=1, sorted=False)
        indexes = indexes.unsqueeze(-1)

        out_ref_windows = torch.gather(out_ref_windows, 1, indexes.expand(-1, -1, out_ref_windows.shape[-1]))
        topk_probs_class = torch.gather(out_logits, 1, indexes.expand(-1, -1, out_logits.shape[-1]))
        out_ref_windows = torch.cat(
            (
                out_ref_windows.detach(),
                topk_probs_class.sigmoid().detach(),
            ),
            dim=-1,
        )

        out_pos = None
        out_embed = None

        return out_embed, out_pos, out_ref_windows, indexes

    @torch.no_grad()
    def _momentum_update_gt_decoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.decoder.parameters(), self.decoder_gt.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, src, pos, noised_gt_box=None, noised_gt_onehot=None, attn_mask=None, targets=None,
                score_mask=None):
        assert pos is not None, "position encoding is required!"
        src_anchors = self._create_ref_windows(src)
        src, _, src_shape = flatten_with_shape(src, None)
        src_pos = []
        for pe in pos:
            b, c = pe.shape[:2]
            pe = pe.view(b, c, -1).transpose(1, 2)
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1)
        src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]])
        score_mask = score_mask.flatten(-2)

        coarse_memory = self.dga_layer(src, src_pos, src_shape, src_start_index, src_anchors, score_mask)
        query_embed, query_pos, topk_proposals, topk_indexes = self._quality_query_selection(coarse_memory, src_anchors)

        if noised_gt_box is not None:
            noised_gt_proposals = torch.cat(
                (
                    noised_gt_box,
                    noised_gt_onehot,
                ),
                dim=-1,
            )
            topk_proposals = torch.cat(
                (
                    noised_gt_proposals,
                    topk_proposals,
                ),
                dim=1,
            )

        init_reference_out = topk_proposals[..., :self.code_size]

        hs, inter_references = self.decoder(query_embed, query_pos, coarse_memory, src_shape, src_start_index,
                                            topk_proposals, attn_mask)

        # optional gt forward
        if targets is not None:
            batch_size = len(targets)
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets]
            max_gt_num = max(per_gt_num)
            batched_gt_boxes_with_score = coarse_memory.new_zeros(batch_size, max_gt_num, self.code_size + self.num_classes)
            for bi in range(batch_size):
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], :self.code_size] = targets[bi]["gt_boxes"]
                batched_gt_boxes_with_score[bi, : per_gt_num[bi], self.code_size:] = F.one_hot(
                    targets[bi]["labels"], num_classes=self.num_classes
                )

            with torch.no_grad():
                self._momentum_update_gt_decoder()
                if noised_gt_box is not None:
                    dn_group_num = noised_gt_proposals.shape[1] // max((max_gt_num * 2), 1)
                    pos_idxs = list(range(0, dn_group_num * 2, 2))
                    pos_noised_gt_proposals = torch.cat(
                        [noised_gt_proposals[:, pi * max_gt_num: (pi + 1) * max_gt_num] for pi in pos_idxs],
                        dim=1,
                    )
                    gt_proposals = torch.cat((batched_gt_boxes_with_score, pos_noised_gt_proposals), dim=1)
                    # create attn_mask for gt groups
                    gt_attn_mask = coarse_memory.new_ones(
                        (dn_group_num + 1) * max_gt_num, (dn_group_num + 1) * max_gt_num
                    ).bool()
                    for di in range(dn_group_num + 1):
                        gt_attn_mask[
                        di * max_gt_num: (di + 1) * max_gt_num,
                        di * max_gt_num: (di + 1) * max_gt_num,
                        ] = False
                else:
                    gt_proposals = batched_gt_boxes_with_score
                    gt_attn_mask = None

                hs_gt, inter_references_gt = self.decoder_gt(
                    None,
                    None,
                    coarse_memory,
                    src_shape,
                    src_start_index,
                    gt_proposals,
                    gt_attn_mask,
                )

            init_reference_out = torch.cat(
                (
                    init_reference_out,
                    gt_proposals[..., :self.code_size],
                ),
                dim=1,
            )

            hs = torch.cat(
                (
                    hs,
                    hs_gt,
                ),
                dim=2,
            )
            inter_references = torch.cat(
                (
                    inter_references,
                    inter_references_gt,
                ),
                dim=2,
            )

        inter_references_out = inter_references

        return hs, init_reference_out, inter_references_out, coarse_memory, src_anchors, topk_indexes


class DGALayer(nn.Module):
    def __init__(self, d_model, nhead, nlevel, dim_feedforward, dropout, activation, keep_ratio):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.cross_attn = DeformableBox3dAttention(d_model, nlevel, nhead, with_rotation=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, src_shape, src_start_idx, ref_windows, score_mask):
        foreground_num = math.ceil(score_mask.shape[1] * self.keep_ratio)
        select_score_mask, indices = torch.topk(score_mask, k=foreground_num, dim=-1)
        select_src = torch.gather(src, 1, indices.unsqueeze(-1).repeat(1, 1, src.size(-1)))
        select_pos = torch.gather(pos, 1, indices.unsqueeze(-1).repeat(1, 1, pos.size(-1)))
        select_ref_windows = torch.gather(ref_windows, 1, indices.unsqueeze(-1).repeat(1, 1, ref_windows.size(-1)))

        query_num = math.ceil(foreground_num * self.keep_ratio)
        query_indices = torch.topk(select_score_mask, k=query_num, dim=-1).indices
        query_src = torch.gather(select_src, 1, query_indices.unsqueeze(-1).repeat(1, 1, select_src.size(-1)))
        query_pos = torch.gather(select_pos, 1, query_indices.unsqueeze(-1).repeat(1, 1, select_pos.size(-1)))

        q = k = self.with_pos_embed(query_src, query_pos)
        query_src2 = self.self_attn(q, k, query_src)[0]
        query_src = query_src + query_src2
        query_src = self.query_norm(query_src)
        select_src = select_src.scatter(1, query_indices.unsqueeze(-1).repeat(1, 1, select_src.size(-1)), query_src)

        output = src

        src2 = self.cross_attn(
            self.with_pos_embed(select_src, select_pos),
            src,
            src_shape,
            None,
            src_start_idx,
            None,
            select_ref_windows,
        )

        select_src = select_src + self.dropout1(src2[0])
        select_src = self.norm1(select_src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(select_src))))
        select_src = select_src + self.dropout2(src2)
        select_src = self.norm2(select_src)

        output = output.scatter(1, indices.unsqueeze(-1).repeat(1, 1, output.size(-1)), select_src)

        return output


class SEEDDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, nlevel, dim_feedforward, dropout, activation, code_size=7, num_classes=3):
        super().__init__()
        self.code_size = code_size
        self.num_classes = num_classes

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = DeformableBox3dAttention(d_model, nlevel, nhead, with_rotation=True)

        self.pos_embed_layer = MLP(code_size + num_classes, d_model, d_model, 3)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, idx, query, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask=None):
        if idx == 0:
            query = self.pos_embed_layer(ref_windows)
            q = k = query
        elif query_pos is None:
            query_pos = self.pos_embed_layer(ref_windows)
            q = k = self.with_pos_embed(query, query_pos)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = query.transpose(0, 1)

        query2 = self.self_attn(q, k, v, attn_mask=attn_mask)[0]
        query2 = query2.transpose(0, 1)
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.cross_attn(
            self.with_pos_embed(query, query_pos),
            memory,
            memory_shape,
            None,
            memory_start_idx,
            None,
            ref_windows[..., :self.code_size],
        )[0]

        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        return query


class SEEDDecoder(nn.Module):
    def __init__(self, d_model, decoder_layer, num_layers, cp_flag):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.cp_flag = cp_flag
        self.code_size = decoder_layer.code_size

    def forward(self, query, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask=None):
        output = query
        intermediate = []
        intermediate_ref_windows = []

        for idx, layer in enumerate(self.layers):
            if self.cp_flag:
                output = cp.checkpoint(layer, idx, output, query_pos, memory, memory_shape, memory_start_idx,
                                       ref_windows, attn_mask)
            else:
                output = layer(idx, output, query_pos, memory, memory_shape, memory_start_idx, ref_windows, attn_mask)
            new_ref_logits, new_ref_windows, new_ref_ious = self.detection_head(output, ref_windows[..., :self.code_size], idx)
            new_ref_probs = new_ref_logits.sigmoid()  # .max(dim=-1, keepdim=True).values
            ref_windows = torch.cat(
                (
                    new_ref_windows.detach(),
                    new_ref_probs.detach(),
                ),
                dim=-1,
            )
            intermediate.append(output)
            intermediate_ref_windows.append(new_ref_windows)
        return torch.stack(intermediate), torch.stack(intermediate_ref_windows)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def flatten_with_shape(tensor_list, mask_list):
    """
    Params:
    :tensor_list: [(B, C, H1, W1), ..., (B, C, HN, WN)]
    :mask_list: [(B, H1, W1), ..., (B, HN, WN)]

    Return:
    :tensor_flatten: (B, L, C)
    :mask_flatten: (B, L)
    :tensor_shape: (N, 2)
    """
    assert isinstance(tensor_list, collections.abc.Sequence)
    assert len(tensor_list) > 0

    N = len(tensor_list)
    tensor_shape = torch.zeros(N, 2, dtype=torch.int64, device=tensor_list[0].device)
    tensor_flatten = []

    if mask_list is not None:
        mask_flatten = []

    for i, tensor in enumerate(tensor_list):
        new_tensor = tensor.flatten(2).permute(0, 2, 1)
        tensor_flatten.append(new_tensor)

        if mask_list is not None:
            mask = mask_list[i]
            new_mask = mask.flatten(1)
            mask_flatten.append(new_mask)
            assert tensor.shape[2] == mask.shape[1]
            assert tensor.shape[3] == mask.shape[2]
        tensor_shape[i, 0] = tensor.shape[2]
        tensor_shape[i, 1] = tensor.shape[3]

    mask_flatten = torch.cat(mask_flatten, dim=1) if mask_list is not None else None
    tensor_flatten = torch.cat(tensor_flatten, dim=1)

    return tensor_flatten, mask_flatten, tensor_shape
