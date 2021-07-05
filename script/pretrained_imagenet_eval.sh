
python eval_pretrained_resnet.py --config ./configs/image_to_objectnet_imagenet_c_r_o.yaml --source_data /research/diva2/donhk/imagenet/ILSVRC2012_train/ --target_data ./val_filelists/objectnet_c_r_o.txt --entropy

python eval_pretrained_resnet.py --config ./configs/image_to_objectnet_imagenet_c_r_o.yaml --source_data /research/diva2/donhk/imagenet/ILSVRC2012_train/ --target_data ./val_filelists/objectnet_c_r_o.txt --probability

python eval_pretrained_resnet.py --config ./configs/image_to_objectnet_imagenet_c_r_o.yaml --source_data /research/diva2/donhk/imagenet/ILSVRC2012_train/ --target_data ./val_filelists/objectnet_c_r_o.txt --logit
