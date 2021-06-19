
## Visda 21

Ovanet
--
python train.py --config ./configs/image_to_objectnet.yaml --source_data /research/diva2/donhk/imagenet/ILSVRC2012_train/ --target_data ./data_loader/filelist/objectnet_filelist.txt

python train.py --config ./configs/image_to_imagenet_c_r.yaml --source_data /research/diva2/donhk/imagenet/ILSVRC2012_train/ --target_data ./data_loader/filelist/imagenet_c_and_r_filelist.txt.txt


Pretrained model
--
python train_resnet.py --config ./configs/image_to_objectnet.yaml --source_data /research/diva2/donhk/imagenet/ILSVRC2012_train/ --target_data ./data_loader/filelist/objectnet_filelist.txt

python train_resnet.py --config ./configs/image_to_imagenet_c_r.yaml --source_data /research/diva2/donhk/imagenet/ILSVRC2012_train/ --target_data ./data_loader/filelist/imagenet_c_and_r_filelist.txt.txt
