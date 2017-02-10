import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class Inverter(chainer.Chain):

    """ Inverter """

    def __init__(self, model, label, initialImg=None, layers=[], beta=2, p=10, kambda_a=1, lambda_tv=10, lambda_lp=10):
        eval_model = model.copy()
        eval_model.train = False
        self.labelsize = model.labelsize
        self.label = label
        self.insize = model.insize
        if initialImg is None:
            initialImg = np.random.randn(3, self.insize, self.insize).astype(np.float32)[np.newaxis]
            initialImg = initialImg * 64
        self.model = eval_model

        super(Inverter, self).__init__(
            #model = eval_model,
            img=L.Parameter(initialImg)
        )
        self.beta = beta
        self.p = p
        self.lambda_a = lambda_a
        self.lambda_tv = lambda_tv
        self.lambda_lp = lambda_lp
        self.train = True
        self.add_persistent('Wh_data', np.array([[[[1],[-1]]]], dtype='f'))
        self.add_persistent('Ww_data', np.array([[[[1, -1]]]], dtype='f'))
        truth_probabiliry = np.zeros(self.labelsize)[np.newaxis].astype(np.float32)
        truth_probabiliry[0, label] = 1
        self.add_persistent('truth_probabiliry', truth_probabiliry)
        self.add_persistent('t_dummy', np.asarray([label]).astype(np.int32))

    def getbeforesoftmax(self, loss):
        v = loss
        before_softmax=None
        while (v.creator is not None):
            if v.creator.label == 'SoftmaxCrossEntropy':
                before_softmax = v.creator.inputs[0]
                break
            v = v.creator.inputs[0]
        return before_softmax

    def getprobability(self, loss):
        v = loss
        before_softmax=None
        while (v.creator is not None):
            if v.creator.label == 'SoftmaxCrossEntropy':
                before_softmax = v.creator.inputs[0]
                break
            v = v.creator.inputs[0]
        return F.softmax(before_softmax)

    def getCrossEntropy(self, loss):
        v = loss
        after_softmax=None
        while (v.creator is not None):
            if v.creator.label == 'SoftmaxCrossEntropy':
                after_softmax = v.creator.outputs[0]()
                break
            v = v.creator.inputs[0]
        return after_softmax

    def tv_norm(self, x, Wh, Ww):
        diffh = F.convolution_2d(F.reshape(x, (3, 1, self.insize, self.insize)), W=Wh)
        diffw = F.convolution_2d(F.reshape(x, (3, 1, self.insize, self.insize)), W=Ww)
        tv = (F.sum(diffh**2) + F.sum(diffw**2))**(self.beta / 2.)
        return tv

    def __call__(self, x):
        self.model.train = False
        Wh = chainer.Variable(self.Wh_data)
        Ww = chainer.Variable(self.Ww_data)
        tp = chainer.Variable(self.truth_probabiliry)
        t = chainer.Variable(self.t_dummy)
        model_loss = self.model(self.img(), t)
        #p = self.getprobability(model_loss)
        a = self.getbeforesoftmax(model_loss)
        #print(F.sum(p**2).data)
        #p = (1 / F.sum(p**2)) .* (p**2)
        #ce = self.getCrossEntropy(model_loss)
        #print(t.data, ce.data, p.data, tp.data)
        #class_mse = ce#F.mean_squared_error(p, tp)
        #class_mse = F.sum(-F.log(p**2 / F.sum(p**2).data) * tp)
        activation = -F.sum(a * tp)
        #class_mse = F.sum(-F.log(p) * tp)
        lp = (F.sum(self.img()**self.p) ** (1.0/self.p)) / np.prod(self.img().data.shape[1:])

        tv = self.tv_norm(self.img(), Wh, Ww) / np.prod(self.img().data.shape[1:])
        loss = self.lambda_a * activation + self.lambda_tv * tv + self.lambda_lp * lp
        chainer.report({'inv_loss': loss, 'activation': activation, 'tv': tv, 'lp': lp}, self)
        #print('inverter', x.data, class_mse.data, tv.data)
        return loss
