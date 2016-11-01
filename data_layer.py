import caffe
import scipy.io as sio
import yaml
import os
import cv2
from random import shuffle


class HumanParsingDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['data', 'seg']
        params = eval(self.param_str)
        self.phase = params['phase']
        self.batch_size = 1
        self.gts = os.listdir('./annotation')
        self.testimgs = os.listdir('./test')
        self.cur = 0
        top[0].reshape(self.batch_size, 3, 144, 96)
        if self.phase == 'TRAIN':
            top[1].reshape(self.batch_size, 1, 144, 96)
            top[2].reshape(self.batch_size, 18)

    def forward(self, bottom, top):
        if self.phase == 'TRAIN':
            for iters in range(self.batch_size):
                image, seg, vector = self.load_next_batch()
                top[0].data[iters, ...] = image
                top[1].data[iters, ...] = seg
                top[2].data[iters, ...] = vector


    def reshape(self, bottom, top):
        """
        No need to reshape.
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate.
        """
        pass

    def load_next_batch(self):
        if self.cur == len(self.gts):
            self.cur = 0
            shuffle(self.gts)

        mat = sio.loadmat('./annotation/%s' % self.gts[self.cur])
        image = mat['image']
        seg = mat['seg']
        vector = mat['vector']
        # print vector
        # print vector.shape
        # stop
        channel_swap = (2, 0, 1)
        image = image.transpose(channel_swap)
        self.cur += 1
        return image, seg, vector
