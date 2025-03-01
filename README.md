# Adversarial Malware Generation Using GANs

[![docs](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ZaydH/MalwareGAN/blob/master/LICENSE)

Implementation of a Generative Adversarial Network (GAN) that can create adversarial malware examples.  The work is inspired by **MalGAN** in the paper "[*Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN*](https://arxiv.org/abs/1702.05983)" by Weiwei Hu and Ying Tan.

Framework written in [PyTorch](https://pytorch.org/) and supports CUDA.

## Environment

To have a working environment, you need to setup a `.env` file. Use the following template:

```bash
MLFLOW_TRACKING_URI=~
MLFLOW_S3_ENDPOINT_URL=~
AWS_ACCESS_KEY_ID=~
AWS_SECRET_ACCESS_KEY=~
```

## Running the Script

Use DVC pipelines

## Current State

- [x] Reducing dimensions with MCA
- [ ] Extracting normal feature list from LIME
    - [ ] For MCA ![in Testing](https://img.shields.io/badge/In_Testing-BrightGreen)
    <!-- 
        TODO(Adapt LIME for all features): Write an experiment for features other than binary
     -->
    - [ ] For Other Features ![Not started](https://img.shields.io/static/v1?label=&message=Not%20Started&color=red)
<!-- 
    TODO(Classification study): Check classification with/without modifications
 -->
- [ ] Classification performance study ![Not started](https://img.shields.io/static/v1?label=&message=Not%20Started&color=red)