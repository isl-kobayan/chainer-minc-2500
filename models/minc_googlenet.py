import chainer
import chainer.functions as F
import chainer.links as L
import config

class MINC_GoogLeNet(chainer.Chain):

    insize = 224
    finetuned_model_path = './models/minc-googlenet.caffemodel'
    mean_value = (104, 117, 124)

    layer_rank = {'conv1':1, 'relu1':2, 'pool1':3, 'norm1':4,
        'conv2_reduce':5, 'relu2_reduce':6,
        'conv2':7, 'relu2':8, 'norm2':9, 'pool2':10,
        'inception_3a':15, 'inception_3b':20, 'pool3':21,
        'inception_4a':26, 'inception_4b':31, 'inception_4c':36, 'inception_4d':41, 'inception_4e':46,
        'pool4':47, 'inception_5a':52, 'inception_5b':57,
        'pool5':58, 'fc8-20':59}

    def call_inception(self, x, name):
        out1 = self[name + '/1x1'](x)
        out3 = self[name + '/3x3'](F.relu(self[name + '/3x3_reduce'](x)))
        out5 = self[name + '/5x5'](F.relu(self[name + '/5x5_reduce'](x)))
        pool = self[name + '/pool_proj'](F.max_pooling_2d(x, 3, stride=1, pad=1))
        y = F.relu(F.concat((out1, out3, out5, pool), axis=1))
        return y

    def add_inception(self, name, in_channels, out1, proj3, out3, proj5, out5, proj_pool):
        super(MINC_GoogLeNet, self).add_link(name + '/1x1', F.Convolution2D(in_channels, out1, 1))
        super(MINC_GoogLeNet, self).add_link(name + '/3x3_reduce', F.Convolution2D(in_channels, proj3, 1))
        super(MINC_GoogLeNet, self).add_link(name + '/3x3', F.Convolution2D(proj3, out3, 3, pad=1))
        super(MINC_GoogLeNet, self).add_link(name + '/5x5_reduce', F.Convolution2D(in_channels, proj5, 1))
        super(MINC_GoogLeNet, self).add_link(name + '/5x5', F.Convolution2D(proj5, out5, 5, pad=2))
        super(MINC_GoogLeNet, self).add_link(name + '/pool_proj', F.Convolution2D(in_channels, proj_pool, 1))

    def __init__(self, labelsize=config.labelsize):
        self.labelsize = labelsize
        super(MINC_GoogLeNet, self).__init__()
        super(MINC_GoogLeNet, self).add_link('conv1/7x7_s2', F.Convolution2D(3,  64, 7, stride=2, pad=3))
        super(MINC_GoogLeNet, self).add_link('conv2/3x3_reduce', F.Convolution2D(64,  64, 1))
        super(MINC_GoogLeNet, self).add_link('conv2/3x3', F.Convolution2D(64, 192, 3, stride=1, pad=1))
        self.add_inception('inception_3a', 192,  64,  96, 128, 16,  32,  32)
        self.add_inception('inception_3b', 256, 128, 128, 192, 32,  96,  64)
        self.add_inception('inception_4a', 480, 192,  96, 208, 16,  48,  64)
        self.add_inception('inception_4b', 512, 160, 112, 224, 24,  64,  64)
        self.add_inception('inception_4c', 512, 128, 128, 256, 24,  64,  64)
        self.add_inception('inception_4d', 512, 112, 144, 288, 32,  64,  64)
        self.add_inception('inception_4e', 528, 256, 160, 320, 32, 128, 128)
        self.add_inception('inception_5a', 832, 256, 160, 320, 32, 128, 128)
        self.add_inception('inception_5b', 832, 384, 192, 384, 48, 128, 128)

        super(MINC_GoogLeNet, self).add_link('fc8-20', L.Linear(1024, self.labelsize))

        super(MINC_GoogLeNet, self).add_link('loss1/conv', F.Convolution2D(512, 128, 1))
        super(MINC_GoogLeNet, self).add_link('loss1/fc', L.Linear(4 * 4 * 128, 1024))
        super(MINC_GoogLeNet, self).add_link('loss1/classifier', L.Linear(1024, self.labelsize))

        super(MINC_GoogLeNet, self).add_link('loss2/conv', F.Convolution2D(528, 128, 1))
        super(MINC_GoogLeNet, self).add_link('loss2/fc', L.Linear(4 * 4 * 128, 1024))
        super(MINC_GoogLeNet, self).add_link('loss2/classifier', L.Linear(1024, self.labelsize))

        self.train = True

    def __call__(self, x, t):
        h = F.relu(self['conv1/7x7_s2'](x))
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5, alpha=(1e-4)/5, k=1)
        h = F.relu(self['conv2/3x3_reduce'](h))
        h = F.relu(self['conv2/3x3'](h))
        h = F.max_pooling_2d(F.local_response_normalization(
            h, n=5, alpha=(1e-4)/5, k=1), 3, stride=2)

        h = self.call_inception(h, 'inception_3a')
        h = self.call_inception(h, 'inception_3b')
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.call_inception(h, 'inception_4a')

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self['loss1/conv'](l))
        l = F.dropout(F.relu(self['loss1/fc'](l)), 0.7, train=self.train)
        l = self['loss1/classifier'](l)
        loss1 = F.softmax_cross_entropy(l, t)

        h = self.call_inception(h, 'inception_4b')
        h = self.call_inception(h, 'inception_4c')
        h = self.call_inception(h, 'inception_4d')

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self['loss2/conv'](l))
        l = F.dropout(F.relu(self['loss2/fc'](l)), 0.7, train=self.train)
        l = self['loss2/classifier'](l)
        loss2 = F.softmax_cross_entropy(l, t)

        h = self.call_inception(h, 'inception_4e')
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.call_inception(h, 'inception_5a')
        h = self.call_inception(h, 'inception_5b')

        h = F.average_pooling_2d(h, 7, stride=1)
        h = self['fc8-20'](F.dropout(h, 0.4, train=self.train))
        loss3 = F.softmax_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        accuracy = F.accuracy(h, t)
        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy
        }, self)
        return loss

    def layer2rank(self, layer):
        l2r = {'conv1':1, 'relu1':2, 'pool1':3, 'norm1':4,
            'conv2_reduce':5, 'relu2_reduce':6,
            'conv2':7, 'relu2':8, 'norm2':9, 'pool2':10,
            'inception_3a':15, 'inception_3b':20, 'pool3':21,
            'inception_4a':26, 'inception_4b':31, 'inception_4c':36, 'inception_4d':41, 'inception_4e':46,
            'pool4':47, 'inception_5a':52, 'inception_5b':57,
            'pool5':58, 'fc8-20':59}
        return l2r[layer]

    def getRankVariableFromLoss(self, rank):
        if self.loss is None:
            return None

        def printl(v):
            if v.creator is not None:
                print(v.rank, v.creator.label)
                printl(v.creator.inputs[0])
                if(v.creator.label == '_ + _'):
                    printl(v.creator.inputs[1])
                #elif(v.creator.label == 'Concat'):
                #    for l in v.creator.inputs:
                #        if (v.rank + 1 != l.rank) : printl(l)
        #printl(self.loss3)
        v = self.loss3
        while (v.creator is not None):
            if v.rank == rank:
                return v.creator.outputs[0]()
            v = v.creator.inputs[0]
        return None
