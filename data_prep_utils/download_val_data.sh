#!/bin/bash

echo "Downloading Val Data"
wget http://csr.bu.edu/ftp/visda/2021/val_data.zip
echo "Unzipping"
unzip -qq val_data.zip
echo "Removing zip"
rm val_data.zip
#python data_prep_utils/modify_filelists.py

