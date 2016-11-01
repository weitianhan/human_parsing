import os
import cv2
import caffe
import numpy as np
from timer import Timer
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(0)
    timer = Timer()
    prototxt = './test.prototxt'
    model = './snapshot/_iter_675000.caffemodel'
    net = caffe.Net(prototxt, model, caffe.TEST)
    testimgs = os.listdir('./human_resize')
    cnt = 0

    fig = plt.figure()
    # color = np.random.randint(256, size=(18,3))
    # color[0, :] = 0
    with open('color_template.pkl', 'rb') as f:
        template = pickle.load(f)
    color = np.array(template)
    result = np.zeros((144,96,3))

    for imgname in testimgs:
        cnt += 1
        ori_img = cv2.imread('./human_resize/%s' % imgname)
        # mat = sio.loadmat('./test/4565_2655.mat')
        # ori_img = mat['image']
        img = ori_img.transpose(2,0,1)
        img = img.reshape(1, 3, 144, 96)
        blob = {
            'data': img.astype(np.float32, copy=False)
        }
        net.blobs['data'].reshape(*blob['data'].shape)
        timer.tic()
        blobs_out = net.forward(**blob)
        output = net.blobs['loss_seg'].data[...]
        conf_map = output[0] # (18,144,96)
        class_map = np.argmax(conf_map, axis=0) # (144,96)
        # f = class_map.flatten()
        # print np.bincount(f)
        # stop
        for x in range(144):
            for y in range(96):
                c = class_map[x,y]
                if conf_map[c,x,y] < 0.1:
                    c = 0
                result[x,y,:] = color[c,:]

        plt.subplot(121)
        plt.imshow(ori_img); plt.axis('off'); plt.title('Original Img')

        plt.subplot(122)
        plt.imshow(result); plt.axis('off'); plt.title('result')
        # plt.show()
        # stop
        plt.savefig('./output1/%s.jpg' % imgname.replace('.jpg', ''), format='jpg')
        plt.cla()
        plt.clf()
        plt.close()
        timer.toc()
        print 'process image %d/%d, forward average time %f' % (cnt, len(testimgs), timer.average_time)
