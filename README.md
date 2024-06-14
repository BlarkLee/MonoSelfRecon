# MonoSelfRecon: Purely Self-Supervised Explicit Generalizable 3D Reconstruction of Indoor Scenes from Monocular RGB Views
This is the code implementation of MonoSelfRecon. https://openaccess.thecvf.com/content/CVPR2024W/3DMV/papers/Li_MonoSelfRecon_Purely_Self-Supervised_Explicit_Generalizable_3D_Reconstruction_of_Indoor_Scenes_CVPRW_2024_paper.pdf or arxive
https://arxiv.org/pdf/2404.06753

## Prepare Environments
## Prepare Dataset

## Train

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --cfg ./config/train.yaml
```

## Evaluation
```
python main.py --cfg ./config/test.yaml
python tools/evaluation.py --model ./results/scene_scannet_allfrag_ckpt_fragloss_att_fusion_eval_28 --n_proc 16
python tools/evaluation.py --model ./results/scene_scannet_allfrag_ckpt_fragloss_att_gru_semi_fusion_eval_20 --n_proc 16
```

## Citation 
```
@InProceedings{MonoSelfRecon,
    author    = {Li, Runfa and Mahbub, Upal and Bhaskaran, Vasudev and Nguyen, Truong},
    title     = {MonoSelfRecon: Purely Self-Supervised Explicit Generalizable 3D Reconstruction of Indoor Scenes from Monocular RGB Views},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {656-666}
}
```

## Acknowledgments
Thanks for the contribution from "NeuralRecon" - https://github.com/zju3dv/NeuralRecon and "Atlas" - https://github.com/magicleap/Atlas. 
