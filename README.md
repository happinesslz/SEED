# SEED
## [ECCV 2024] SEED: A Simple and Effective 3D DETR in Point Clouds

## News
- [2024.07.02] SEED is accepted by ECCV 2024!

### Results on Waymo Open

#### Validation set (single frame)
| Model | mAP/mAPH_L1 | mAP/mAPH_L2 | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |
|-------------------------------------------------------------------------------------------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| SEED-S   | 81.1/79.1 | 73.1/70.8 | 78.2/77.7 | 70.2/69.7 | 81.3/75.8 | 73.3/68.1 | 78.4/77.2 | 75.7/74.5 |
| SEED-B  | 81.2/79.2 | 74.9/72.8 | 79.7/79.2 | 71.8/71.4 | 83.1/78.3 | 75.5/70.8 | 80.0/78.8 | 77.3/76.1 |
| SEED-L    | 81.4/79.5 | 75.5/73.5 | 79.8/79.3 | 71.9/71.5 | 83.6/79.1 | 76.2/71.8 | 81.2/80.0 | 78.4/77.3 |

### Test set  (3 frame)
| Model | mAP/mAPH_L1 | mAP/mAPH_L2 | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Leaderboard |
|-------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| SEED-L  | 82.2/80.2 | 76.9/75.0 | 84.3/83.9 | 77.5/77.1 | 85.2/82.3 | 79.9/77.0 | 81.0/80.1 | 78.7/77.8 | [link](https://waymo.com/open/challenges/detection-3d/results/13405607-47a3/1709782897550042/)

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/).



## TODO
- [ ] Release the paper.
- [ ] Release the code of SEED on Waymo.
- [ ] Release the code of SEED on nuScenes.


## Citation
```
@inproceedings{
  liu2024seed,
  title={SEED: A Simple and Effective 3D DETR in Point Clouds},
  author={Liu, Zhe and Hou, Jinghua and Ye, Xiaoqing and Wang, Tong, and Wang, Jingdong and Bai, Xiang},
  booktitle={ECCV},
  year={2024},
}
```

## Acknowledgements
We thank these great works and open-source repositories:
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [ConQueR](https://github.com/V2AI/EFG), [FocalFormer3d](https://github.com/NVlabs/FocalFormer3D) and [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).
