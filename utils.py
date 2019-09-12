import numpy as np
import tensorflow as tf
import logging
# import os

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

        handler = logging.FileHandler(file_path) if type=='file' else logging.StreamHandler()
        self.logger.addHandler(
            self.getHandler(handler,
                            format,
                            level))

    def prompt(self, s, level=0):
        if level==0: self.logger.debug(s)
        elif level==1: self.logger.info(s)
        else: self.logger.warning(s)
