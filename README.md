# MonoSelfRecon: Purely Self-Supervised Explicit Generalizable 3D Reconstruction of Indoor Scenes from Monocular RGB Views
This is the code implementation of MonoSelfRecon. https://openaccess.thecvf.com/content/CVPR2024W/3DMV/papers/Li_MonoSelfRecon_Purely_Self-Supervised_Explicit_Generalizable_3D_Reconstruction_of_Indoor_Scenes_CVPRW_2024_paper.pdf or arxiv
https://arxiv.org/pdf/2404.06753

## Prepare Environments
## Prepare Dataset
We use ScanNet standard training and testing split, as Atlas and NeuralRecon. Please

## Train
Train with two GPUs
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --cfg ./config/train.yaml
```
You can also download our pretrained checkpoints from https://drive.google.com/drive/folders/1MG91GdkYn1Almryek6ehd5kgeHJDviOp?usp=drive_link. `ckpt_fragloss_att` is the purely self-supervision, `ckpt_fragloss_att_gru_semi` is the weak supervision. Put the two folders both at the project root directory.

## Evaluation
This step will generate mesh files for every scene and save at `./results`. You can use our pretrained checkpoints or the one trained by yourself.
```
python main.py --cfg ./config/test.yaml
```
Or you can skip this step and download the mesh files we generated for all scenes from https://drive.google.com/drive/folders/1MG91GdkYn1Almryek6ehd5kgeHJDviOp?usp=drive_link and directly evaluate it. Unzip `scene_scannet_allfrag_ckpt_fragloss_att_fusion_eval_28.zip` and `scene_scannet_allfrag_ckpt_fragloss_att_gru_semi_fusion_eval_20.zip`, put them under `./result`

To evaluate the purely self-supervised training, use
```
python tools/evaluation.py --model ./results/scene_scannet_allfrag_ckpt_fragloss_att_fusion_eval_28 --n_proc 16
```
To evaluate the weakly-supervised training, use
```
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
