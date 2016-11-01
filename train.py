import caffe
import sys
from timer import Timer

if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(1)

    solver = caffe.SGDSolver('solver.prototxt')
    # solver.net.copy_from()
    timer = Timer()
    for iters in range(675000):
        timer.tic()
        solver.step(1);
        timer.toc()
        if solver.iter % (20) == 0:
            print 'speed: {:.3f}s / iter'.format(timer.average_time)
        # print solver.net.blobs['conv1_1'].data.shape
        # print solver.net.blobs['conv2_1'].data.shape
        # print solver.net.blobs['conv3_1'].data.shape
        # print solver.net.blobs['conv4_1'].data.shape
        # print solver.net.blobs['fc2'].data
        # print solver.net.blobs['sum5'].data.shape
        # print solver.net.blobs['conv6_2'].data.shape
        # print solver.net.blobs['conv7_2'].data.shape
        # print solver.net.blobs['cmap1'].data
        # for key in solver.net.blobs:
        #     print key
        # for key in solver.net.params:
        #     print key,np.max(solver.net.params[key][0].data)
