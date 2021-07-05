#!/bin/bash

echo "Downloading Val Data"
wget http://csr.bu.edu/ftp/visda/2021/val_data.zip
echo "Unzipping"
unzip -qq val_data.zip
echo "Removing zip"
rm val_data.zip
python data_prep_utils/modify_filelists.py
cat ./val_filelists/imagenet_c_r_o_filelist.txt ./val_filelists/objectnet_filelist.txt > ./val_filelists/objectnet_imagenet_c_r_o_filelist.txt

