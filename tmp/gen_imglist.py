import os.path as osp
import os
import sys
import numpy as np


if __name__ == '__main__':
    PATH = '/media/huangrui/cacd/cacd_mtcnn128'
    n = 0
    with open('/media/huangrui/cacd/cacd_mtcnn128_train.txt', 'w') as f:
        for i in range(1900):
            id = i
            if not os.path.exists(osp.join(PATH, str(i))):
                os.mkdir(osp.join(PATH, str(i)))
            img_list = os.listdir(osp.join(PATH, str(i)))

            if len(img_list) > 0:
                for img_name in img_list:
                    age = img_name.split('_')[0]
                    line = str(i) + '/' + img_name + ' ' + str(n) + ' ' + age + '\n'
                    f.write(line)
                n += 1
    
    with open('/media/huangrui/cacd/cacd_mtcnn128_test.txt', 'w') as f:
        for i in range(1900, 1999):
            id = i
            if not os.path.exists(osp.join(PATH, str(i))):
                os.mkdir(osp.join(PATH, str(i)))
            img_list = os.listdir(osp.join(PATH, str(i)))
            if len(img_list) > 0:
                for img_name in img_list:
                    age = img_name.split('_')[0]
                    line = str(i) + '/' + img_name + ' ' + str(n) + ' ' + age + '\n'
                    f.write(line)
                n += 1

       