from pathlib import Path
import math
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import Detector
from loss import SetCriterion
from loss import HungarianMatcher
from torch.utils.data import RandomSampler, BatchSampler
from coco import CocoDetection, make_coco_transforms
from utils import collate_fn, fix_targets, DualYolosmAP
from tqdm import tqdm
from torch import nn
from mean_average_precision import MetricBuilder


sys.path.append(os.getcwd())


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 1
    num_classes = 91
    num_detection_tokens = 100
    # checkpoint = '/dccstor/epochs/gwallace/huggingface/patent/weights/version_002/checkpoint.pth'
    checkpoint = '/dccstor/epochs/gwallace/huggingface/patent/weights/version_001/checkpoint.pth'
    # checkpoint = '/dccstor/epochs/gwallace/huggingface/archive/EARLY_EXIT_ARCHIVE/checkpoints/big_split1/checkpoint.pth'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = Detector(
        num_classes=num_classes,
        pre_trained='/dccstor/epochs/gwallace/huggingface/patent/backbones/base.pth',
        det_token_num=num_detection_tokens,
        backbone_name='base',
        init_pe_size=[800, 1333]
    ).to(device)

    matcher = HungarianMatcher(
        cost_class=1, 
        cost_bbox=5, 
        cost_giou=2
    )
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(
        num_classes, 
        matcher=matcher, 
        weight_dict=weight_dict,
        eos_coef=0.1, 
        losses=losses
    ).to(device)

    val_dataset = CocoDetection(
        img_folder='/dccstor/epochs/gwallace/huggingface/data/val2017', 
        ann_file='/dccstor/epochs/gwallace/huggingface/data/annotations/instances_val2017.json', 
        transforms=make_coco_transforms('val', 'tiny', 800), 
        return_masks=False
    )

    val_batch_sampler = BatchSampler(
        RandomSampler(val_dataset), 
        batch_size, 
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_sampler=val_batch_sampler,
        collate_fn=collate_fn, 
        num_workers=2
    )

    metric_fn = MetricBuilder.build_evaluation_metric(
        "map_2d",
        num_classes=num_classes,
        async_mode=True,
    )

    output_dir = Path('/' + '/'.join(checkpoint.split('/')[:-1]))
    if os.path.exists(f'{output_dir}/testing_log.txt'):
        os.remove(f'{output_dir}/testing_log.txt')

    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.eval()
    criterion.eval()
    total_loss = 0
    total_mAP = 0
    iteration = 0
    for batch_num, batch in tqdm(enumerate(val_dataloader, 0), unit="batch", total=len(val_dataloader)):
        images, targets = batch
        images, targets = images.to(device), fix_targets(targets, device)

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_value = loss.item()
        total_loss += loss_value
        iteration += 1

        logits, bboxes = outputs['pred_logits'], outputs['pred_boxes']

        # import torchvision.transforms as T
        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        # from transformers import YolosForObjectDetection
        # class UnNormalize(object):
        #     def __init__(self, mean, std):
        #         self.mean = mean
        #         self.std = std

        #     def __call__(self, tensor):
        #         """
        #         Args:
        #             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        #         Returns:
        #             Tensor: Normalized image.
        #         """
        #         for t, m, s in zip(tensor, self.mean, self.std):
        #             t.mul_(s).add_(m)
        #             # The normalize code -> t.sub_(m).div_(s)
        #         return tensor
        # i = 0
        # for i in range(batch_size):
        #     _, ax = plt.subplots()
        #     unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        #     img = T.ToPILImage()(unorm(images.tensors[i]).detach().cpu())
        #     ax.imshow(img)
        #     width, height = img.width, img.height
        #     for logit, box in zip(logits[i], bboxes[i]):
        #         prob = nn.functional.softmax(logit, -1)
        #         score, label = prob[..., :-1].max(-1)
        #         if score > 0.85:
        #             rect = patches.Rectangle(
        #                 ((box[0].item() - 0.5 * box[2].item()) * width, (box[1].item() - 0.5 * box[3].item()) * height),
        #                 (box[2].item()) * width,
        #                 (box[3].item()) * height,
        #                 linewidth=1,
        #                 edgecolor="r",
        #                 facecolor="none",
        #             )
        #             ax.add_patch(rect)
        #             label = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny').config.id2label[label.item()]
        #             ax.text((box[0].item() - 0.5 * box[2].item()) * width, (box[1].item() - 0.5 * box[3].item()) * height, label, color='red')
        #     plt.savefig(f"pics_out/test_boxes_{i}.png")
        # exit()

        for i in range(batch_size):
            ## filter detections for confidence > 0.85 and reformat predictions -> [[xmin, ymin, xmax, ymax, class_id, confidence]]
            preds = np.zeros((1, 6))
            for logit, box in zip(logits[i], bboxes[i]):
                prob = nn.functional.softmax(logit, -1)
                score, label = prob[..., :-1].max(-1)
                temp = [
                    [
                        box[0].item() - 0.5 * box[2].item(),
                        box[1].item() - 0.5 * box[3].item(),
                        box[0].item() + 0.5 * box[2].item(),
                        box[1].item() + 0.5 * box[3].item(),
                        label.item(),
                        score.item(),
                    ]
                ]
                preds = np.append(preds, np.array(temp), axis=0)
            if preds.shape[0] == 1:
                preds = np.empty((1, 6))
            else:
                preds = preds[1:, :]

            ## reformat groundtruths -> [[xmin, ymin, xmax, ymax, class_id, difficult, crowd]]
            gt = np.zeros((1, 7))
            for box, label in zip(targets[i]['boxes'], targets[i]['labels']):
                if box[0] != -1:
                    temp = [
                        [
                            box[0].item() - 0.5 * box[2].item(),
                            box[1].item() - 0.5 * box[3].item(),
                            box[0].item() + 0.5 * box[2].item(),
                            box[1].item() + 0.5 * box[3].item(),
                            label.item(),
                            0,
                            0,
                        ]
                    ]
                    gt = np.append(gt, np.array(temp), axis=0)
            if gt.shape[0] == 1:
                gt = np.empty((1, 7))
            else:
                gt = gt[1:, :]

            ## add image to mAP metric function
            metric_fn.add(preds, gt)

        mAP = metric_fn.value(
            iou_thresholds=np.arange(0.5, 1.0, 0.05),
            recall_thresholds=np.arange(0.0, 1.01, 0.01),
            mpolicy="soft",
        )['mAP']
        total_mAP += mAP

        if batch_num % 10 == 0:
            with open(f'{output_dir}/testing_log.txt', 'a+') as f:
                f.write(f'testing loss: {total_loss / iteration}, testing mAP: {total_mAP / iteration}\n')


if __name__ == '__main__':
    main()