import chainer
import chainer.functions as F
import chainer.links as L
from .. import config
from chainer.functions import caffe
import types

class Caffemodel(chainer.links.CaffeFunction):

    """Caffemodel"""

    layer_rank = {'conv1':1, 'relu1':2, 'norm1':3, 'pool1':4,
            'conv2':5, 'relu2':6, 'norm2':7, 'pool2':8,
            'conv3':9, 'relu3':10, 'conv4':11, 'relu4':12,
            'conv5':13, 'relu5':14, 'pool5':15,
            'fc6':16, 'relu6':17, 'fc7':18, 'relu7':19, 'fc8':20}

    def __init__(self, finetuned_model_path, labelsize=config.labelsize):
        self.finetuned_model_path = finetuned_model_path
        super(Caffemodel, self).__init__(finetuned_model_path)
        self.train = True
        # gather loss layer and last fc layer
        self.losses = [l[0] for l in self.forwards
            if type(self.forwards[l[0]]) is types.FunctionType
             and self.forwards[l[0]].__name__ == 'softmax_cross_entropy']
        self.before_softmax = [l[1][0] for l in self.forwards
            if l[0] in self.losses]
        self.outputlayers = self.losses + self.before_softmax


    def __call__(self, x, t):
        outputs, = super(Caffemodel, self)(inputs={'data': x}, outputs=self.outputlayers)
        losses = [for x in zip(outputs, self.outputlayers) if x[1] in self.losses]
        before_softmax = [for x in zip(outputs, self.outputlayers) if x[1] in self.before_softmax]
        reports = dict(zip(self.losses, losses))
        # calculate accuracy from fc layer
        reports.update(dict(zip(
            ['accuracy/' + s for s in self.before_softmax],
            [F.accuracy(h, t) for h in before_softmax])))

        chainer.report(reports, self)
