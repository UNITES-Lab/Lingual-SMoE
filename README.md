# Sparse MoE with Language-Guided Routing for Multilingual Machine Translation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for "Sparse MoE with Language-Guided Routing for Multilingual Machine Translation". Our implementation is based on [fairseq-moe](https://github.com/ZhiYuanZeng/fairseq-moe).

- Authors: [Xinyu Zhao](https://zhaocinyu.github.io/), [Xuxi Chen](http://xxchen.site/), [Yu Cheng](https://www.linkedin.com/in/chengyu05/) and [Tianlong Chen](https://tianlong-chen.github.io/)
- Paper: [OpenReview](https://openreview.net/pdf?id=ySS7hH1smL)

## **Setup**

```
conda create -n fairseq python=3.8 -y && conda activate fairseq

git clone https://github.com/pytorch/fairseq && cd fairseq
pip install --editable ./

pip install fairscale==0.4.0 hydra-core==1.0.7 omegaconf==2.0.6
pip install boto3 zmq iopath tensorboard nltk
pip install sacrebleu[ja] sacrebleu[ko] wandb

wandb login 
```

## **Data Preprocessing**

1. Raw data download: [OPUS-100](https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz && tar -xzf opus-100-corpus-v1.0.tar.gz)
2. Preprocessing pipeline: [num-multi](https://github.com/cordercorder/nmt-multi/blob/main/scripts/opus-100/data_process/multilingual_preprocess.sh)

## **Training**

1. For the training script, see the example in [train_scripts/train.sh](train_scripts/train.sh)
2. To load language embedding, please first replace the lang_dict.txt  in the processed data folder with [assets/lang_dict.txt](assets/lang_dict.txt) with a consistent index.

## **Inference**

1. First, generate translation data: `bash eval_scripts/generate.sh -d opus16 -s save_dir -n 8 -c 1.0` (8 gpus. 1.0 capacity factor)
2. Then compute BLEU: `bash eval_scripts/eval.sh -d opus16 -s save_dir`


## Citation

```bibtex
@inproceedings{
zhao2024sparse,
title={Sparse MoE with Language Guided Routing for Multilingual Machine Translation},
author={Xinyu Zhao and Xuxi Chen and Yu Cheng and Tianlong Chen},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=ySS7hH1smL}
}
```