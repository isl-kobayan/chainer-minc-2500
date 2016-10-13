import chainer
import chainer.functions as F
import chainer.links as L
import config

class MINC_Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227
    finetuned_model_path = './models/minc-alexnet.caffemodel'
    mean_value = (104, 117, 124)

    layer_rank = {'conv1':1, 'relu1':2, 'norm1':3, 'pool1':4,
            'conv2':5, 'relu2':6, 'norm2':7, 'pool2':8,
            'conv3':9, 'relu3':10, 'conv4':11, 'relu4':12,
            'conv5':13, 'relu5':14, 'pool5':15,
            'fc6':16, 'relu6':17, 'fc7':18, 'relu7':19, 'fc8':20}

    def __init__(self, labelsize=config.labelsize):
        self.labelsize = labelsize
        super(MINC_Alex, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096)
        )
        super(MINC_Alex, self).add_link('fc8-20', L.Linear(4096, self.labelsize))
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x)), alpha=(1e-4)/5, k=1), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h)), alpha=(1e-4)/5, k=1), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self['fc8-20'](h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss
