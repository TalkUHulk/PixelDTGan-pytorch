import numpy as np
import os
import shutil

lookbook = "./data/lookbook/"


def _get_train_set(path):
    file_names = [x.path for x in os.scandir(path)
                  if x.name.endswith('jpg') and 'CLEAN0' in x.name]
    target_names = [x.path for x in os.scandir(path)
                    if x.name.endswith('jpg') and 'CLEAN1' in x.name]
    pid = [p.split('/')[-1].split('_')[0] for p in target_names]

    target_files = {}
    for i in range(len(target_names)):
        tmp = {pid[i]: target_names[i]}
        target_files.update(tmp)

    return [np.sort(file_names), target_files]


total_source, total_target = _get_train_set(lookbook + 'train')
total_num = len(total_target)
val_num = int(total_num * .05 // 100 * 100)
test_num = int(total_num * .05 // 100 * 100)

a = []
for i in total_target.keys():
    a.append(i)
np.random.shuffle(a)

total_val = a[:val_num]
total_test = a[-test_num:]

for fv in total_val:
    source = [f for f in total_source if f.find(fv) != -1]
    target = total_target[fv]
    shutil.move(target, lookbook + 'validation/')
    for i in source:
        shutil.move(i, lookbook + 'validation/')

for ft in total_test:
    source = [f for f in total_source if f.find(ft) != -1]
    target = total_target[ft]
    shutil.move(target, lookbook + 'test/')
    for i in source:
        shutil.move(i, lookbook + 'test/')








