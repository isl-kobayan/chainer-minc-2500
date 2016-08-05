import chainer
import chainer.functions as F
import chainer.links as L
import finetune

class InvAlex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def set_finetune(self):
        finetune.load_param('./bvlc_alexnet.pkl', self)

    def __init__(self):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8_fmd=L.Linear(4096, self.labelsize),
            deconv1=L.Deconvolution2D(96, 3, 11, stride=4),
            deconv2=L.Deconvolution2D(256, 96,  5, pad=2),
            deconv3=L.Deconvolution2D(384, 256,  3, pad=1),
            deconv4=L.Deconvolution2D(384, 384,  3, pad=1),
            deconv5=L.Deconvolution2D(256, 384,  3, pad=1),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss

    def clear_inv(self):
        self.x = None

    def inv(self, y, layer='conv2'):
        self.clear_inv()
        unpool5 = lambda x: F.unpooling_2d(x, 3, stride=2)
        unpool2 = lambda x: F.unpooling_2d(x, 3, stride=2)
        unpool1 = lambda x: F.unpooling_2d(x, 3, stride=2)
        funcs=[unpool5, deconv5, deconv4, deconv3, unpool2, deconv2, unpool1, deconv1]
        func=funcs
        if layer == 'conv5':
            func=funcs[1:]
        if layer == 'conv4':
            func=funcs[2:]
        if layer == 'conv3':
            func=funcs[3:]
        if layer == 'unpool2':
            func=funcs[4:]
        if layer == 'conv2':
            func=funcs[5:]
        if layer == 'unpool1':
            func=funcs[6:]
        if layer == 'conv1':
            func=funcs[7:]
	
        h = y
        for f in funcs:
            h = f(h)

        self.x = x
        return self.x
