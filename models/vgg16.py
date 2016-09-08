import chainer
import chainer.functions as F
import chainer.links as L
import config

class VGG16(chainer.Chain):

    """VGG 16 layer model."""

    insize = 224
    finetuned_model_path = './models/VGG_ILSVRC_16_layers.caffemodel'
    mean_value = (103.939, 116.779, 123.68)

    layer_rank = {'conv1_1':1, 'conv1_2':3, 'pool1':5,
        'conv2_1':6, 'conv2_2':8, 'pool2':10,
        'conv3_1':11, 'conv3_2':13, 'conv3_3':15, 'pool3':17,
        'conv4_1':18, 'conv4_2':20, 'conv4_3':22, 'pool4':24,
        'conv5_1':25, 'conv5_2':27, 'conv5_3':29, 'pool5':31,
        'fc6':32, 'fc7':34, 'fc8':36}

    def __init__(self, labelsize=config.labelsize):
        self.labelsize = labelsize
        super(VGG16, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, self.labelsize)
        )
        self.train = True

    def __call__(self, x, t):
        h = self.conv1_2(F.relu(self.conv1_1(x)))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.conv2_2(F.relu(self.conv2_1(h)))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(h)))))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(h)))))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(h)))))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.5)
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss
