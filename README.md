# Local Path Integration for Attribution

This code implements Local Path Integration from the following paper:

> Peiyu Yang, Naveed Akhtar, Zeyi Wen, and Ajmal Mian
>
> [Local Path Integration for Attribution](https://scholar.google.com/scholar?cluster=4845895326140495709&hl=en&oi=scholarr)

## Introduction

Path-based attributions account for the weak dependence property by choosing a reference from the local distribution of the input. We devise a method to identify the local input distribution and propose a technique to stochastically integrate the model gradients over the paths defined by the references sampled from that distribution. Our local path integration (LPI) method consistently outperforms existing path attribution techniques on deep visual models. Contributing to the ongoing search of reliable evaluation metrics for the interpretation methods, we introduce DiffID metric that uses the relative difference between insertion and deletion games to alleviate the distribution shift problem faced by existing metrics.
![LPI](figs/LPI.png)


## Prerequisites

- python 3.9.2
- matplotlib 3.5.1
- numpy 1.21.5
- pytorch 1.12.0
- torchvision 0.13.1


## Estimate Attributions with LPI

### Step 1: Preparing dataset.
```
dataset\IMAGENET
```

### Step 2: Preparing model.
```
pretrained_models\YOUR_MODEL
```

### Step 3: Estimate dataset distribution.
```
python distribution_estimation.py -dataset ImageNet -model resnet34 -center_num 0,11 -ref_num 10,20
```

### Step 4: Estimate attributions with LPI.
```
python main.py -attr_method=LPI -model resnet34 -dataset ImageNet -metric visualize -k 5 -bg_size 10 -num_center 11
```

## Quantitatively evaluate attributions with DiffID
```
python main.py -attr_method=LPI -model resnet34 -dataset ImageNet -metric DiffID -k 5 -bg_size 10 -num_center 11
```

## Bibtex
If you found this work helpful for your research, please cite the following paper:
```
@artical{yang2023local,
    title={Local Path Integration for Attribution},
    author={Peiyu, Yang and Naveed, Akhtar and Zeyi, Wen and Ajmal, Mian},
    booktitle={AAAI Conference on Artificial Intelligence {AAAI}},
    year={2023}
}
```