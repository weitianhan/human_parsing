import caffe
import numpy as np
import yaml

class CmapLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self._height = params['height']
        self._width = params['width']
        self.vector = bottom[0].data
        self.batch_size = self.vector.shape[0]
        self.class_num = self.vector.shape[1]
        top[0].reshape(self.batch_size, self.class_num, self._height, self._width)

    def reshape(self, bottom, top):
        """
        Reshaping happens during the call to forward
        """
        pass

    def forward(self, bottom, top):
        vector = bottom[0].data
        output = np.zeros((self.batch_size, self.class_num, self._height, self._width), dtype=np.float32)
        for batch in range(self.batch_size):
            for i in range(self.class_num):
                output[batch][i].fill(vector[0][i])
        top[0].data[...] = output

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate.
        """
        pass
