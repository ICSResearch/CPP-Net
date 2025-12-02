# [CPP-Net: Embracing Multi-Scale Feature Fusion into Deep Unfolding CP-PPA Network for Compressive Sensing](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_CPP-Net_Embracing_Multi-Scale_Feature_Fusion_into_Deep_Unfolding_CP-PPA_Network_CVPR_2024_paper.html)

## Abstract

In the domain of compressive sensing (CS), deep unfolding networks (DUNs) have garnered attention for their good performance and certain degree of interpretability rooted in CS domain, achieved by marrying traditional optimization solvers with deep networks.
However, current DUNs are ill-suited for the intricate task of capturing fine-grained image details, leading to perceptible distortions and blurriness in reconstructed images, particularly at low CS ratios, e.g., 0.10 and below.
In this paper, we propose CPP-Net, a novel deep unfolding CS framework, inspired by the primal-dual hybrid strategy of the Chambolle and Pock Proximal Point Algorithm (CP-PPA).
First, we derive three iteration submodules, $\mathbf{X}^{(k)}$, $\mathbf{V}^{(k)}$ and $\mathbf{Y}^{(k)}$, by incorporating customized deep learning modules to solve the sparse basis related proximal operator within CP-PPA.
Second, we design the Dual Path Fusion Block (DPFB) to adeptly extract and fuse multi-scale feature information, enhancing sensitivity to feature information at different scales and improving detail reconstruction.
Third, we introduce the Iteration Fusion Strategy (IFS) to effectively weight the fusion of outputs from diverse reconstruction stages, maximizing the utilization of feature information and mitigating the information loss during reconstruction stages.
Extensive experiments demonstrate that CPP-Net effectively reduces distortion and blurriness while preserving richer image details, outperforming current state-of-the-art methods.

## Test
```
python test.py --model=cpp8 --cs_ratio=10 --dataset=Set11
```
Place the test dataset into the "./data/" folder and replace the option "dataset" with the name of the test dataset.
./data/Set11/
    01.png
    02.png
    ...

Then replace the argument `--dataset` with your dataset name (e.g., Set11).
The results.csv and reconstructed images will be generated in the folder "./results/{model}/{dataset}/{cs_ratio}/".

The results.csv will save the results in "{image name},{PSNR (dB)},{SSIM},{LPIPS}" format.

The reconstructed images will be saved under the name "{image name}_{cs_ratio}\_PSNR\_{PSNR:.2f}\_SSIM\_{SSIM:.4f}\_LPIPS\_{LPIPS:.4f}.png".

## Train

1. Multi-GPUs
```
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --model=cpp8 --data_path="../data/train" --eval_data_path="../data/val" --cs_ratio=10 --blr=4e-5 --min_lr=1e-6 --epochs=400 --batch_size=8 --warmup_epochs=10 --input_size=96
```

2. Single GPU
```
python train.py --model=cpp8 --data_path="../data/train" --eval_data_path="../data/val" --cs_ratio=10 --blr=4e-5 --min_lr=1e-6 --epochs=400 --batch_size=8 --warmup_epochs=10 --input_size=96
```
Replace the options "data_path" and "eval_data_path" with the path to the train dataset and eval dataset, respectively.

The model parameters will be saved under the name "checkpoint-{model}-{cs_ratio}-best.pth" in the ". /model" folder.

## Pretrained Models
- [Baidu Drive](https://pan.baidu.com/s/1Na4uATo8B2E_5zwYut6itw?pwd=d3y7)

## Requirements
- Python == 3.10.12
- Pytorch == 1.12.0

## Citation
```
@inproceedings{guoCPPNet2024,
  title={CPP-Net: Embracing Multi-Scale Feature Fusion into Deep Unfolding CP-PPA Network for Compressive Sensing},
  author={Guo, Zhen and Gan, Hongping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
## Acknowledge
This work is based on the awesome work [ConvNeXt](https://github.com/facebookresearch/ConvNeXt).
