
import numpy as np


lines = open('./imagenet_c_and_r_filelist.txt', 'r').readlines()

class_list = {}
labels = []
for i in lines:
    i = i.strip()
    path, label  = i.split(' ')[0], i.split(' ')[1]
    class_list[path.split('/')[-2]] = int(label)


for i in class_list:
    labels.append(class_list[i])


print(class_list)

labels = np.array(labels)

in_class = (labels<1000).sum()
print('in : {}'.format(in_class))
out_class = (labels>1000).sum()
print('out : {}'.format(out_class))



