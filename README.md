<div align="center">

<h1>HORIZON: High-Resolution Semantically Controlled Panorama Synthesis</h1>

<div>
    <a href='' target='_blank'>Kun Yan</a>&emsp;
    <a href='' target='_blank'>Lei Ji</a>&emsp;
    <a href='' target='_blank'>Chenfei Wu</a>&emsp;
    <a href='' target='_blank'>Jian Liang</a>&emsp;
    <a href='' target='_blank'>Ming Zhou</a>&emsp;
    <a href='' target='_blank'>Nan Duan</a>&emsp;
    <a href='' target='_blank'>Shuai Ma</a>&emsp;
</div>

<strong><a href='https://aaai.org/aaai-conference/' target='_blank'>AAAI 2024</a></strong>


<h4>Official implementation of HORIZON: High-Resolution Semantically Controlled Panorama Synthesis <br> A novel framework that generates high-quality, semantically-controlled 360-degree panoramas with minimal distortion</h4>

[![arXiv](https://img.shields.io/badge/arXiv-2210.04522-b31b1b.svg)](https://arxiv.org/abs/2210.04522)
[![Azure Blob](https://img.shields.io/badge/Model%20Weights-4285F4?style=for-the-badge&logo=Microsoft%20Azure&logoColor=white)]()
</div>
![](https://github.com/naykun/HORIZON/blob/master/assets/task.png?raw=true)

## Updates

[Soon] Model weights release.

[01/2024] Code released.

[12/2023] Paper Accepted by AAAI 2024

[10/2022] Paper uploaded to [arXiv](https://arxiv.org/abs/2210.04522).



## Installation
Use [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) to setup environment for HORIZON. Setup the required environment by the following command:
```bash
conda env create -f env.yml
conda activate horizon
```

### Download Pretrained Models
Please download our checkpoints from [Azure Blob]() to run the following inference scripts.
After download, make a ckpt folder follow the belown structure:
```
ckpt
├── CLIP
│   └── ViT-L-14.pt
├── dataset
│   └── pano #dataset folder
├── last.pth 
└── vqg
    └── VQGan16384F16.pth
```
### Inference 

```
bash script/inference.sh
```
When the inference finish, you'll get all the generated panorama under `/ckpt/pano/horizon_mini/eval_visu/pred/`

## Citation
```
@misc{yan2022horizon,
      title={HORIZON: A High-Resolution Panorama Synthesis Framework}, 
      author={Kun Yan and Lei Ji and Chenfei Wu and Jian Liang and Ming Zhou and Nan Duan and Shuai Ma},
      year={2022},
      eprint={2210.04522},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


