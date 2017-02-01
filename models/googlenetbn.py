
import chainer
import chainer.functions as F
import chainer.links as L
from .. import config

class GoogLeNetBN(chainer.Chain):

    insize = 224
    finetuned_model_path = './models/googlenet_bn.caffemodel'
    #mean_value = (104, 117, 123)
    mean_value = (104, 117, 124)

    layer_rank = {'conv1':1, 'relu1':2, 'pool1':3, 'norm1':4,
        'conv2_reduce':5, 'relu2_reduce':6,
        'conv2':7, 'relu2':8, 'norm2':9, 'pool2':10,
        'inception_3a':15, 'inception_3b':20, 'pool3':21,
        'inception_4a':26, 'inception_4b':31, 'inception_4c':36, 'inception_4d':41, 'inception_4e':46,
        'pool4':47, 'inception_5a':52, 'inception_5b':57,
        'pool5':58, 'loss3_classifier':59}

    def add_conv_bn_sc(self, name, in_channels, out_channels, ksize, stride=1, pad=0):
        super(GoogLeNetBN, self).add_link(name, L.Convolution2D(in_channels, out_channels, ksize, stride=stride, pad=pad, nobias=True))
        super(GoogLeNetBN, self).add_link(name + '/bn', L.BatchNormalization(out_channels, use_gamma=self.use_gamma, use_beta=self.use_beta))
        #super(GoogLeNetBN, self).add_link(name + '/bn/sc', L.Scale(W_shape = (out_channels), bias_term = True))

    def add_fc_bn_sc(self, name, in_size, out_size):
        super(GoogLeNetBN, self).add_link(name, L.Linear(in_size, out_size, nobias=True))
        super(GoogLeNetBN, self).add_link(name + '/bn', L.BatchNormalization(out_size, use_gamma=self.use_gamma, use_beta=self.use_beta))
        #super(GoogLeNetBN, self).add_link(name + '/bn/sc', L.Scale(W_shape = (out_size), bias_term = True))

    def call_conv_bn_sc(self, x, name, test=False, finetune=False):
        #return F.relu(self[name + '/bn/sc'](self[name + '/bn'](self[name](x), test=test, finetune=finetune)))
        return F.relu(self[name + '/bn'](self[name](x), test=test, finetune=finetune))

    def call_fc_bn_sc(self, x, name, test=False, finetune=False):
        #return F.relu(self[name + '/bn/sc'](self[name + '/bn'](self[name](x), test=test, finetune=finetune)))
        return F.relu(self[name + '/bn'](self[name](x), test=test, finetune=finetune))

    def call_inception_bn(self, x, name, test=False, finetune=False):
        outs = []
        if hasattr(self, name + '/1x1'):
            out1 = self.call_conv_bn_sc(x, name + '/1x1', test=test, finetune=finetune)
            outs.append(out1)

        out3 = self.call_conv_bn_sc(x, name + '/3x3_reduce', test=test, finetune=finetune)
        out3 = self.call_conv_bn_sc(out3, name + '/3x3', test=test, finetune=finetune)
        outs.append(out3)

        out33 = self.call_conv_bn_sc(x, name + '/double3x3_reduce', test=test, finetune=finetune)
        out33 = self.call_conv_bn_sc(out33, name + '/double3x3a', test=test, finetune=finetune)
        out33 = self.call_conv_bn_sc(out33, name + '/double3x3b', test=test, finetune=finetune)
        outs.append(out33)

        pool = self[name + '/pool'](x)
        if hasattr(self, name + '/pool_proj'):
            pool = self.call_conv_bn_sc(pool, name + '/pool_proj', test=test, finetune=finetune)
        outs.append(pool)
        y = F.concat(outs, axis=1)
        return y

    def add_inception_bn(self, name, in_channels, out1, proj3, out3, proj33, out33, pooltype, proj_pool=None, stride=1):
        # 1x1
        if out1 > 0:
            assert stride == 1
            assert proj_pool is not None
            self.add_conv_bn_sc(name + '/1x1', in_channels, out1, 1, stride=stride)

        # 3x3reduce -> 3x3
        self.add_conv_bn_sc(name + '/3x3_reduce', in_channels, proj3, 1)
        self.add_conv_bn_sc(name + '/3x3', proj3, out3, 3, pad=1, stride=stride)
        # 3x3reduce -> 3x3 -> 3x3
        self.add_conv_bn_sc(name + '/double3x3_reduce', in_channels, proj33, 1)
        self.add_conv_bn_sc(name + '/double3x3a', proj33, out33, 3, pad=1)
        self.add_conv_bn_sc(name + '/double3x3b', out33, out33, 3, pad=1, stride=stride)
        # pool -> 1x1
        if pooltype == 'max':
            setattr(self, name + '/pool', lambda x: F.max_pooling_2d(x, 3, stride=stride, pad=1))
        elif pooltype == 'avg':
            setattr(self, name + '/pool', lambda x: F.average_pooling_2d(x, 3, stride=stride, pad=1))
        else:
            raise NotImplementedError()

        if proj_pool is not None:
            self.add_conv_bn_sc(name + '/pool_proj', in_channels, proj_pool, 1)

    def __init__(self, labelsize=config.labelsize):
        self.labelsize = labelsize
        self.use_gamma = True
        self.use_beta = True
        super(GoogLeNetBN, self).__init__()
        self.add_conv_bn_sc('conv1/7x7_s2', 3, 64, 7, stride=2, pad=3)
        self.add_conv_bn_sc('conv2/3x3_reduce', 64, 64, 1)
        self.add_conv_bn_sc('conv2/3x3', 64, 192, 3, pad=1)
        self.add_inception_bn('inception_3a', 192,  64,  64,  64,  64,  96, 'avg', 32),
        self.add_inception_bn('inception_3b', 256,  64,  64,  96,  64,  96, 'avg', 64),
        self.add_inception_bn('inception_3c', 320,   0, 128, 160,  64,  96, 'max', stride=2),
        self.add_inception_bn('inception_4a', 576, 224,  64,  96,  96, 128, 'avg', 128),
        self.add_inception_bn('inception_4b', 576, 192,  96, 128,  96, 128, 'avg', 128),
        self.add_inception_bn('inception_4c', 576, 160, 128, 160, 128, 160, 'avg', 96),
        self.add_inception_bn('inception_4d', 576,  96, 128, 192, 160, 192, 'avg', 96),
        self.add_inception_bn('inception_4e', 576,   0, 128, 192, 192, 256, 'max', stride=2),
        self.add_inception_bn('inception_5a', 1024, 352, 192, 320, 160, 224, 'avg', 128),
        self.add_inception_bn('inception_5b', 1024, 352, 192, 320, 192, 224, 'max', 128),

        super(GoogLeNetBN, self).add_link('loss3/classifier', L.Linear(4096, self.labelsize))

        self.add_conv_bn_sc('loss1/conv', 576, 128, 1)
        self.add_fc_bn_sc('loss1/fc', 128*4*4, 1024)
        super(GoogLeNetBN, self).add_link('loss1/classifier', L.Linear(1024, self.labelsize))

        self.add_conv_bn_sc('loss2/conv', 1024, 128, 1)
        self.add_fc_bn_sc('loss2/fc', 128*2*2, 1024)
        super(GoogLeNetBN, self).add_link('loss2/classifier', L.Linear(1024, self.labelsize))

        self.train = True
        self.finetune = False

    def __call__(self, x, t):
        test = not self.train
        finetune = self.finetune

        h = self.call_conv_bn_sc(x, 'conv1/7x7_s2', test=test, finetune=finetune)
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = self.call_conv_bn_sc(h, 'conv2/3x3_reduce', test=test, finetune=finetune)
        h = self.call_conv_bn_sc(h, 'conv2/3x3', test=test)
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = self.call_inception_bn(h, 'inception_3a', test=test, finetune=finetune)
        h = self.call_inception_bn(h, 'inception_3b', test=test, finetune=finetune)
        h = self.call_inception_bn(h, 'inception_3c', test=test, finetune=finetune)

        a = F.average_pooling_2d(h, 5, stride=3)
        a = self.call_conv_bn_sc(a, 'loss1/conv', test=test, finetune=finetune)
        a = self.call_fc_bn_sc(a, 'loss1/fc', test=test, finetune=finetune)
        a = self['loss1/classifier'](a)
        loss1 = F.softmax_cross_entropy(a, t)

        h = self.call_inception_bn(h, 'inception_4a', test=test, finetune=finetune)
        h = self.call_inception_bn(h, 'inception_4b', test=test, finetune=finetune)
        h = self.call_inception_bn(h, 'inception_4c', test=test, finetune=finetune)
        h = self.call_inception_bn(h, 'inception_4d', test=test, finetune=finetune)
        h = self.call_inception_bn(h, 'inception_4e', test=test, finetune=finetune)

        b = F.average_pooling_2d(h, 5, stride=3)
        b = self.call_conv_bn_sc(b, 'loss2/conv', test=test, finetune=finetune)
        b = self.call_fc_bn_sc(b, 'loss2/fc', test=test, finetune=finetune)
        b = self['loss2/classifier'](b)
        loss2 = F.softmax_cross_entropy(b, t)

        h = self.call_inception_bn(h, 'inception_5a', test=test, finetune=finetune)
        h = self.call_inception_bn(h, 'inception_5b', test=test, finetune=finetune)

        h = F.average_pooling_2d(h, 7, stride=1)
        h = self['loss3/classifier'](h)
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
'''

import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class GoogLeNetBN(chainer.Chain):

    """New GoogLeNet of BatchNormalization version."""

    insize = 224
    mean_value = (104, 117, 124)

    def __init__(self):
        super(GoogLeNetBN, self).__init__(
            conv1=L.Convolution2D(None, 64, 7, stride=2, pad=3, nobias=True),
            norm1=L.BatchNormalization(64),
            conv2=L.Convolution2D(None, 192, 3, pad=1, nobias=True),
            norm2=L.BatchNormalization(192),
            inc3a=L.InceptionBN(None, 64, 64, 64, 64, 96, 'avg', 32),
            inc3b=L.InceptionBN(None, 64, 64, 96, 64, 96, 'avg', 64),
            inc3c=L.InceptionBN(None, 0, 128, 160, 64, 96, 'max', stride=2),
            inc4a=L.InceptionBN(None, 224, 64, 96, 96, 128, 'avg', 128),
            inc4b=L.InceptionBN(None, 192, 96, 128, 96, 128, 'avg', 128),
            inc4c=L.InceptionBN(None, 128, 128, 160, 128, 160, 'avg', 128),
            inc4d=L.InceptionBN(None, 64, 128, 192, 160, 192, 'avg', 128),
            inc4e=L.InceptionBN(None, 0, 128, 192, 192, 256, 'max', stride=2),
            inc5a=L.InceptionBN(None, 352, 192, 320, 160, 224, 'avg', 128),
            inc5b=L.InceptionBN(None, 352, 192, 320, 192, 224, 'max', 128),
            out=L.Linear(None, 1000),

            conva=L.Convolution2D(None, 128, 1, nobias=True),
            norma=L.BatchNormalization(128),
            lina=L.Linear(None, 1024, nobias=True),
            norma2=L.BatchNormalization(1024),
            outa=L.Linear(None, 1000),

            convb=L.Convolution2D(None, 128, 1, nobias=True),
            normb=L.BatchNormalization(128),
            linb=L.Linear(None, 1024, nobias=True),
            normb2=L.BatchNormalization(1024),
            outb=L.Linear(None, 1000),
        )
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
        self.inc3a.train = value
        self.inc3b.train = value
        self.inc3c.train = value
        self.inc4a.train = value
        self.inc4b.train = value
        self.inc4c.train = value
        self.inc4d.train = value
        self.inc4e.train = value
        self.inc5a.train = value
        self.inc5b.train = value

    def __call__(self, x, t):
        test = not self.train

        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(x), test=test)),  3, stride=2, pad=1)
        print(h.data.shape)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h), test=test)), 3, stride=2, pad=1)
        print(h.data.shape)
        h = self.inc3a(h)
        h = self.inc3b(h)
        print(h.data.shape)
        h = self.inc3c(h)
        print(h.data.shape)
        h = self.inc4a(h)

        a = F.average_pooling_2d(h, 5, stride=3)
        a = F.relu(self.norma(self.conva(a), test=test))
        a = F.relu(self.norma2(self.lina(a), test=test))
        a = self.outa(a)
        loss1 = F.softmax_cross_entropy(a, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        b = F.average_pooling_2d(h, 5, stride=3)
        b = F.relu(self.normb(self.convb(b), test=test))
        b = F.relu(self.normb2(self.linb(b), test=test))
        b = self.outb(b)
        loss2 = F.softmax_cross_entropy(b, t)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = self.out(h)
        loss3 = F.softmax_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        accuracy = F.accuracy(h, t)

        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy,
        }, self)
        return loss
#'''
