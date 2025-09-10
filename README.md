# A Green Learning Approach to LDCT Image Restoration

[Paper Link](https://ieeexplore.ieee.org/document/11084379)

This repository provides the implementation of **A Green Learning Approach to LDCT Image Restoration**. We propose a **Green U-shaped Learning (GUSL)** method for LDCT restoration, developed under the Green Learning methodology.

<p align="center">
  <img src="https://github.com/user-attachments/assets/39998089-97a2-4800-9fd1-946d2738c378">
</p>
<p align="center">
  <em>Fig. 1: Overall architecture of GUSL.</em>
</p>

---

## Data Preparation
```bash
python prep.py
```

## Model Training and Testing
```bash
python train.py --exp_name demo
python test.py --exp_name demo
```

## Inference with Pretrained Model
```bash
python test.py --exp_name pretrain
```

---

## Experimental Results
<p align="center">
  <em>Tab. 1: Quantitative evaluation results.</em>
</p>
<p align="center">
  <img width="2708" height="776" alt="res" src="https://github.com/user-attachments/assets/2d2b877c-106d-4e1b-80aa-e2c47f045b58" />
</p>

---

Part of the code in the `prep` folder is adapted from [CTformer](https://github.com/wdayang/CTformer).
