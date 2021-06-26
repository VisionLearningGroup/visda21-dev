import os

root = "."#os.path.abspath('.')


for filelist in os.listdir('val_filelists'):
    with open(os.path.join('val_filelists', filelist+'1'), 'w') as fout:
        with open(os.path.join('val_filelists', filelist), 'r') as f:
            lines = f.readlines()
            for line in lines:
                print(line.replace('<root>', root), file=fout, end='')
    os.remove(os.path.join('val_filelists', filelist))
    os.rename(
        os.path.join('val_filelists', filelist+'1'), 
        os.path.join('val_filelists', filelist))


