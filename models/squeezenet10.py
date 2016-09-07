import chainer
import chainer.functions as F
import chainer.links as L
import config

class SqueezeNet10(chainer.Chain):

    """SqueezeNet v1.0"""

    insize = 227
    finetuned_model_path = './models/squeezenet_v1.0.caffemodel'
    mean_value = (104, 117, 123)

    def call_fire(self, x, name):
        s1 = F.relu(self[name + '/squeeze1x1'](x))
        e1 = self[name + '/expand1x1'](s1)
        e3 = self[name + '/expand3x3'](s1)
        y = F.relu(F.concat((e1, e3), axis=1))
        return y

    def add_fire(self, name, in_channels, s1, e1, e3):
        super(SqueezeNet10, self).add_link(name + '/squeeze1x1', L.Convolution2D(in_channels, s1, 1))
        super(SqueezeNet10, self).add_link(name + '/expand1x1', L.Convolution2D(s1, e1, 1))
        super(SqueezeNet10, self).add_link(name + '/expand3x3', L.Convolution2D(s1, e3, 3, pad=1))

    def __init__(self, labelsize=config.labelsize):
        self.labelsize = labelsize
        super(SqueezeNet10, self).__init__()
        super(SqueezeNet10, self).add_link('conv1', L.Convolution2D(3,  96, 7, stride=2))
        self.add_fire('fire2', 96, 16, 64, 64)
        self.add_fire('fire3', 128, 16, 64, 64)
        self.add_fire('fire4', 128, 32, 128, 128)
        self.add_fire('fire5', 256, 32, 128, 128)
        self.add_fire('fire6', 256, 48, 192, 192)
        self.add_fire('fire7', 384, 48, 192, 192)
        self.add_fire('fire8', 384, 64, 256, 256)
        self.add_fire('fire9', 512, 64, 256, 256)
        super(SqueezeNet10, self).add_link('conv10', L.Convolution2D(
            512, self.labelsize, 1, pad=1,
            initialW=np.random.normal(0, 0.01, (self.labelsize, 512, 1, 1))))
        self.train = True

    def __call__(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.call_fire(h, 'fire2')
        h = self.call_fire(h, 'fire3')
        h = self.call_fire(h, 'fire4')

        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.call_fire(h, 'fire5')
        h = self.call_fire(h, 'fire6')
        h = self.call_fire(h, 'fire7')
        h = self.call_fire(h, 'fire8')

        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.call_fire(h, 'fire9')
        h = F.dropout(h, ratio=0.5, train=self.train)

        h = F.relu(self.conv10(h))
        h = F.reshape(F.average_pooling_2d(h, 13), (x.data.shape[0], self.labelsize))

        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        return loss
