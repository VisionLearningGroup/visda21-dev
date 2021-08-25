
## [Visual Domain Adaptation Challenge (VISDA-21)](http://ai.bu.edu/visda-2021/)
## Updates
***07/03/2021***
1. Add instructions on submission to leaderboard.
2. Change source-only evaluation and OVANet training following the format of leaderboard submission. Target data is changed to be the unified list of imagenet-r-c-o and objectron.


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

## Corruption blacklist

Participants shall not manually apply any of the corruptions present in ImageNet-C during training, namely Gaussian noise, Shot noise, Impulse noise, Defocus blur, Frosted Glass Blur, Motion blur, Zoom blur, Snow, Frost, Fog, Brightness, Contrast, Elastic, Pixelation and JPEG compression. 

## Submission

**Submission for Validation Leaderboard**

During validation phase, we open a validation leaderboard to let participants know the process of submission and other teams' performances.

During validation phase, we utilize the whole imagenet-r-c-o and objectnet as the target domain.
`objectnet_imagenet_c_r_o_filelist.txt` will be our groundtruth file.
Then, when submitting to the validation leaderboard, ***Please use the objectnet_imagenet_c_r_o_filelist.txt as a target domain to train your model.***
This is to compare the validation submission in a fair way. (3) in Baselines are the correct way of training during validation phase.

Participants will need to submit ***two corresponding prediction files, source_only_pred.txt and adapt_pred.txt***.
The source only prediction file needs to contain the prediction produced by a model trained only with source samples, i.e, source imagenet samples.
The adapt file will be predictions of adapted model.

Note that the source only prediction file will be collected to see the gap between source only and adapted models.
The adapted_pred.txt will be used to rank participants.


**Submission Format**

The sample of submission file is stored in `./submission/sample_submit.txt` .
Each line shows a filename, class prediction (closed-set), and anomaly score.
Corresponding gt file will be the provided filelist.
eval_submission.py will be our temporary evaluation script.
See these files before creating submission files.



## Baselines


### Evaluation on ImageNet pre-trained model

---

<imagenet_data_path> should be specified.

(1) ImageNet -> ObjectNet:
```
python eval_pretrained_resnet.py --config ./configs/image_to_objectnet.yaml --source_data <imagenet_data_path>/ILSVRC2012_train/ --target_data ./data_prep_utils/val_filelists/objectnet_filelist.txt
```

(2) ImageNet -> ImageNet-C,R,O:

```
python eval_pretrained_resnet.py --config ./configs/image_to_imagenet_c_r_o.yaml --source_data <imagenet_data_path>/ILSVRC2012_train/ --target_data ./data_prep_utils/val_filelists/imagenet_c_r_o_filelist.txt
```

(3) ImageNet -> ImageNet-C,R,O and ObjectNet:

```
python eval_pretrained_resnet.py --config ./configs/image_to_objectnet_imagenet_c_r_o.yaml --source_data <imagenet_data_path>/ILSVRC2012_train/ --target_data ./data_prep_utils/val_filelists/objectnet_imagenet_c_r_o_filelist.txt
```

|Target Dataset | Accuracy | AUROC  |
|:---: | :---: | :---:|
| ObjectNet |21.6 | 55.5 |
| ImageNet-R,C,O|  36.0 | 11.0 |
| ObjectNet + ImageNet-R,C,O | 32.7 | 51.0 |


### [OVANet](https://arxiv.org/pdf/2104.03344.pdf)

---
In the paper, OVANet has one parameter (multi) to be tuned. 

(1) ImageNet -> ObjectNet +  ImageNet-C,R,O:

```
python train_ovanet.py --config ./configs/image_to_objectnet.yaml --source_data <imagenet_data_path>/ILSVRC2012_train/ --target_data ./data_prep_utils/val_filelists/objectnet_filelist.txt 
```

(2) ImageNet -> ImageNet-C,R,O:

```
python train_ovanet.py --config ./configs/image_to_imagenet_c_r_o.yaml --source_data <imagenet_data_path>/ILSVRC2012_train/ --target_data ./data_prep_utils/val_filelists/imagenet_c_r_o_filelist.txt
```

(3) ImageNet -> ImageNet-C,R,O and ObjectNet:

```
python train_ovanet.py --config ./configs/image_to_objectnet_imagenet_c_r_o.yaml --source_data <imagenet_data_path>/ILSVRC2012_train/ --target_data ./data_prep_utils/val_filelists/objectnet_imagenet_c_r_o_filelist.txt
```

|Target Dataset | Accuracy | AUROC  |
|:---: | :---: | :---:|
| ObjectNet |   22.4 | 54.1 |
| ImageNet-R,C,O| 35.6 | 15.8 |
| ObjectNet + ImageNet-R,C,O|   32.6 | 48.1 |

### Citation

If you use data, code or its derivatives, please consider citing our tech report avaliable on <a href="https://arxiv.org/abs/2107.11011">arxiv</a>:

```
@misc{visda2021,
      title={VisDA-2021 Competition Universal Domain Adaptation to Improve Performance on Out-of-Distribution Data}, 
      author={Dina Bashkirova and Dan Hendrycks and Donghyun Kim and Samarth Mishra and Kate Saenko and Kuniaki Saito and Piotr Teterwak and Ben Usman},
      year={2021},
      eprint={2107.11011},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
