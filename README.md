
## [Visual Domain Adaptation Challenge (VISDA-21)](http://ai.bu.edu/visda-2021/)


## Dataset Preparation

### Source Domain Training Data : 
The source domain training data consists of the ImageNet-1K dataset. It is 
available for download on [kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview). 
Note that you need to sign up to kaggle and install the api (instructions for 
installing the api and adding credentials are [here](https://github.com/Kaggle/kaggle-api#kaggle-api)).
Once downloaded, untar `imagenet_object_localization_patched2019.tar.gz`, and data 
is available in the directory `ILSVRC/Data/CLS-LOC/train`.
Classes are indexed from 0-999 in the sorted order of wordnet id.


### Validation Data :
To download the validation data (1.7G) run
```bash
bash data_prep_utils/download_val_data.sh
```

Validation data contains a subset of images from 4 different datasets:

1. [ImageNet-R](https://github.com/hendrycks/imagenet-r)
2. [ImageNet-C](https://zenodo.org/record/2235448#.YM6VdzopCV4)
3. [ObjectNet](https://objectnet.dev/index.html)
4. [ImageNet-O](https://github.com/hendrycks/natural-adv-examples)

Filelists (which contain a list of `<image_path> <class_label>`) for each can be found in `./val_filelists/`.

Classes that do not overlap with the classes in ImageNet have been given a class-label of 1000.

The images need to be stored under ./val_data directory. 
```
./val_data/imagenet_c_and_r/*
./val_data/imagenet_o/*
./val_data/objectnet/*
``` 

### Test Data :

Test Data that would be available later and used for final evaluations would be a set of images similar to the above datasets.

Note that labels provided for validation data allow for evaluation and tuning any model hyperparameters and as such those labels should not be used for training. The contest leaderboard based on validation results could be different from the final leaderboard based on test results.

## Evaluation Metrics

1. [Accuracy](https://github.com/VisionLearningGroup/visda21-dev/blob/6b08d9600418d5a413d6f13459786a298ea6df87/eval.py#L75) on 1000 classes in ImageNet
2. [AUROC](https://github.com/VisionLearningGroup/visda21-dev/blob/6b08d9600418d5a413d6f13459786a298ea6df87/eval.py#L76) to evaluate separation between known and unknown classes


## Baselines


### Evaluation on ImageNet pre-trained model

---

<imagenet_data_path> should be specified.

(1) ImageNet -> ObjectNet:

python eval_pretrained_resnet.py --config ./configs/image_to_objectnet.yaml --source_data <imagenet_data_path>/ILSVRC2012_train/ --target_data ./data_prep_utils/val_filelists/objectnet_filelist.txt


(2) ImageNet -> ImageNet-C,R,O:

python eval_pretrained_resnet.py --config ./configs/image_to_imagenet_c_r_o.yaml --source_data <imagenet_data_path>/ILSVRC2012_train/ --target_data ./data_prep_utils/val_filelists/imagenet_c_r_o_filelist.txt


|Target Dataset | Accuracy | AUROC  |
|:---: | :---: | :---:|
| ObjectNet |21.6 | 55.5 |
| ImageNet-R,C,O|  36.0 | 11.0 |

### [OVANet](https://arxiv.org/pdf/2104.03344.pdf)

---
In the paper, OVANet has one parameter (multi) to be tuned. 

(1) ImageNet -> ObjectNet:

python eval_pretrained_resnet.py --config ./configs/image_to_objectnet.yaml --source_data <imagenet_data_path>/ILSVRC2012_train/ --target_data ./data_prep_utils/val_filelists/objectnet_filelist.txt 

(2) ImageNet -> ImageNet-C,R,O:

python eval_pretrained_resnet.py --config ./configs/image_to_imagenet_c_r_o.yaml --source_data <imagenet_data_path>/ILSVRC2012_train/ --target_data ./data_prep_utils/val_filelists/imagenet_c_r_o_filelist.txt

|Target Dataset | Accuracy | AUROC  |
|:---: | :---: | :---:|
| ObjectNet |   21.7 | 54.8 |
| ImageNet-R,C,O| 35.3 | 15.3 |
