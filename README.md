
## Visda 21

## Dataset Preparation

### Source Domain Training Data : 
The source domain training data consists of the Imagenet-1K dataset. It is 
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

1. [Imagenet-R](https://github.com/hendrycks/imagenet-r)
2. [Imagenet-C](https://zenodo.org/record/2235448#.YM6VdzopCV4)
3. [Objectnet](https://objectnet.dev/index.html)
4. [Imagenet-O](https://github.com/hendrycks/natural-adv-examples)

Filelists (which contain a list of `<image_path> <class_label>`) for each can be found in `val_filelists/`.
Classes that do not overlap with the classes in Imagenet have been given a class-label of 1000.

Ovanet
--
python train.py --config ./configs/image_to_objectnet.yaml --source_data /research/diva2/donhk/imagenet/ILSVRC2012_train/ --target_data ./data_loader/filelist/objectnet_filelist.txt

python train.py --config ./configs/image_to_imagenet_c_r.yaml --source_data /research/diva2/donhk/imagenet/ILSVRC2012_train/ --target_data ./data_loader/filelist/imagenet_c_and_r_filelist.txt.txt


Pretrained model
--
python train_resnet.py --config ./configs/image_to_objectnet.yaml --source_data /research/diva2/donhk/imagenet/ILSVRC2012_train/ --target_data ./data_loader/filelist/objectnet_filelist.txt

python train_resnet.py --config ./configs/image_to_imagenet_c_r.yaml --source_data /research/diva2/donhk/imagenet/ILSVRC2012_train/ --target_data ./data_loader/filelist/imagenet_c_and_r_filelist.txt.txt
