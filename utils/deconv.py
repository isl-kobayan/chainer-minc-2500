import chainer
import chainer.functions as F
import chainer.links as L
import config

def deconv(variable):
    v = variable
    while (v.creator is not None):
        mapw = v.creator.inputs[0].data.shape[3]
        maph = v.creator.inputs[0].data.shape[2]
        if (v.creator.label == 'Convolution2DFunction'):
            kw = v.creator.inputs[1].data.shape[3]
            kh = v.creator.inputs[1].data.shape[2]
            sx = v.creator.sx; sy = v.creator.sy
            pw = v.creator.pw; ph = v.creator.ph
        elif (v.creator.label == 'MaxPooling2D'):
            kw = v.creator.kw; kh = v.creator.kh
            sx = v.creator.sx; sy = v.creator.sy
            pw = v.creator.pw; ph = v.creator.ph
        else:
            kw = 1; kh = 1; sx = 1; sy = 1; pw = 0; ph = 0

        left = sx * left - pw
        right = sx * right - pw + kw - 1
        top = sy * top - ph
        bottom = sy * bottom - ph + kh - 1
        if left < 0: left = 0
        if right >= mapw: right = mapw - 1
        if top < 0: top = 0
        if bottom >= maph: bottom = maph - 1

        v = v.creator.inputs[0]
    return (left, right + 1, top, bottom + 1)
class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227
    finetuned_model_path = './models/bvlc_alexnet.caffemodel'
    mean_value = (104, 117, 123)

    layer_rank = {'conv1':1, 'relu1':2, 'norm1':3, 'pool1':4,
            'conv2':5, 'relu2':6, 'norm2':7, 'pool2':8,
            'conv3':9, 'relu3':10, 'conv4':11, 'relu4':12,
            'conv5':13, 'relu5':14, 'pool5':15,
            'fc6':16, 'relu6':17, 'fc7':18, 'relu7':19, 'fc8':20}

    def __init__(self, labelsize=config.labelsize):
        self.labelsize = labelsize
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, self.labelsize),
        )
        self.train = True

    def __call__(self, x, t):
