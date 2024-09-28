from torch.utils.checkpoint import checkpoint
import torch
from .detector3d_template import Detector3DTemplate
from ..model_utils import model_nms_utils


class SEED(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'], batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}

        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']
            pred_scores = final_pred_dict[index]['pred_scores']
            pred_labels = final_pred_dict[index]['pred_labels']

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                selected, selected_scores = model_nms_utils.multi_classes_nms_mmdet(
                    box_scores=pred_scores, box_preds=pred_boxes,
                    box_labels=pred_labels - 1, nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                )

                final_boxes = pred_boxes[selected]
                final_scores = selected_scores
                final_labels = pred_labels[selected]

                if post_process_cfg.get('NOT_APPLY_NMS_FOR_VEL', False):
                    pedcyc_mask = final_labels != 1
                    final_scores_pedcyc = final_scores[pedcyc_mask]
                    final_labels_pedcyc = final_labels[pedcyc_mask]
                    final_boxes_pedcyc = final_boxes[pedcyc_mask]

                    car_mask = pred_labels == 1
                    final_scores_car = pred_scores[car_mask]
                    final_labels_car = pred_labels[car_mask]
                    final_boxes_car = pred_boxes[car_mask]

                    final_scores = torch.cat([final_scores_car, final_scores_pedcyc], 0)
                    final_labels = torch.cat([final_labels_car, final_labels_pedcyc], 0)
                    final_boxes = torch.cat([final_boxes_car, final_boxes_pedcyc], 0)

                final_pred_dict[index]['pred_boxes'] = final_boxes
                final_pred_dict[index]['pred_scores'] = final_scores
                final_pred_dict[index]['pred_labels'] = final_labels

            recall_dict = self.generate_recall_record(
                box_preds=final_pred_dict[index]['pred_boxes'],
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict