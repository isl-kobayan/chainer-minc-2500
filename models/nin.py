import math

import chainer
import chainer.functions as F
import chainer.links as L
import config

class NIN(chainer.Chain):

    """Network-in-Network example model."""

    insize = 227
    finetuned_model_path = './models/nin_imagenet.caffemodel'

    def __init__(self, labelsize=config.labelsize):
        self.labelsize = labelsize
        w = math.sqrt(2)  # MSRA scaling
        super(NIN, self).__init__(
            conv1=L.Convolution2D(3, 96, 11, stride=4, pad=0, wscale=w),
            cccp1=L.Convolution2D(96, 96, 1, wscale=w),
            cccp2=L.Convolution2D(96, 96, 1, wscale=w),
            conv2=L.Convolution2D(96, 256, 5, pad=2, wscale=w),
            cccp3=L.Convolution2D(256, 256, 1, wscale=w),
            cccp4=L.Convolution2D(256, 256, 1, wscale=w),
            conv3=L.Convolution2D(256, 384, 3, pad=1, wscale=w),
            cccp5=L.Convolution2D(384, 384, 1, wscale=w),
            cccp6=L.Convolution2D(384, 384, 1, wscale=w),
        )
        super(NIN, self).add_link('conv4-1024', F.Convolution2D(384, 1024, 3, pad=1, wscale=w))
        super(NIN, self).add_link('cccp7-1024', F.Convolution2D(1024, 1024, 1, wscale=w))
        super(NIN, self).add_link('cccp8', F.Convolution2D(1024, self.labelsize, 1, wscale=w))

        self.train = True

    def __call__(self, x, t):
        h = F.relu(self.cccp2(F.relu(self.cccp1(F.relu(self.conv1(x))))))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self.cccp4(F.relu(self.cccp3(F.relu(self.conv2(h))))))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self.cccp6(F.relu(self.cccp5(F.relu(self.conv3(h))))))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.dropout(h, train=self.train)
        h = F.relu(self['cccp7-1024'](F.relu(self['conv4-1024'](h))))
        h = F.relu(self['cccp8'](h))
        h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], self.labelsize))

        #self.loss = F.softmax_cross_entropy(h, t)
        #self.accuracy = F.accuracy(h, t)
        #return self.loss
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss
