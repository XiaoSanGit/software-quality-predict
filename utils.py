import numpy as np
import pandas as pd
import logging
import os

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img


class Logger:
    def __init__(self):
        self.logger = logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        # default stdout
        logger.addHandler(self.getHandler(logging.StreamHandler(),
                                          '%(asctime)s - %(message)s',
                                          logging.WARNING))

    @staticmethod
    def getHandler(handler, format, level):
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format))
        return handler


    def addHandler(self, type, file_path=None, format='%(asctime)s - %(message)s', level=logging.DEBUG):
        assert type in ('file','stdout')
        if type=='file' and file_path is None:
            file_path = 'logger.txt'
            self.prompt(f'logger: no file path given, set to default {file_path}.')
        if ~os.path.exists(file_path):
            os.mkdir("/".join(file_path.split("/")[:-1]))
        handler = logging.FileHandler(file_path) if type=='file' else logging.StreamHandler()
        self.logger.addHandler(
            self.getHandler(handler,
                            format,
                            level))

    def prompt(self, s, level=0):
        if level==0: self.logger.debug(s)
        elif level==1: self.logger.info(s)
        else: self.logger.warning(s)

def fake_datasets(path="./dataset/train",num=10):
    features = []
    names_pre = ['dem_','dev_','label_']
    features = [[np.random.rand(5, 20), np.random.rand(5, 20), np.random.randint(low=1, high=50, size=1)] for i in
                range(10)]
    for j,item in enumerate(features):
        [np.save(os.path.join(path, names_pre[i] + str(j)), item[i]) for i in range(3)]
        line = []
        line.append([names_pre[0]+str(j)+".npy",names_pre[1]+str(j)+".npy",names_pre[2]+str(j)+".npy"])
        line = pd.DataFrame(line)
        line.to_csv(os.path.join(path,"index.csv"),header=None,index=False,mode="a")

# fake_datasets()