#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from .. import config

def deconv_old(variable):
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

def deconv(self, variable):
    v = variable
    if(v.creator is not None):
        # Convolution -> Deconvolutionに変換
        if (v.creator.label == 'Convolution2DFunction'):
            print(v.creator.label, v.rank)
            convW = v.creator.inputs[1].data
            in_cn, out_cn = convW.shape[0], convW.shape[1] # in/out channels
            kh, kw = convW.shape[2], convW.shape[3] # kernel size
            sx, sy = v.creator.sx, v.creator.sy # stride
            pw, ph = v.creator.pw, v.creator.ph # padding

            name = 'conv' + v.rank # temporal layer name
            super(DeconvNet, self).add_link(name, L.Deconvolution2D(
            in_cn, out_cn, (kh, kw), stride=(sy, sx), pad=(ph, pw),
            nobias=True, initialW=convW))
            self.forwards[name] = self[name]
            # もし畳み込み層にバイアスがある場合、それも登録
            if len(v.creator.inputs) == 3:
                F.bias(v)
                b = v.creator.inputs[2].data
                bname = 'convb' + v.rank
                super(DeconvNet, self).add_link(bname, L.Bias(shape=b.shape))
                self[bname].b.data = b
                self.depends[bname] = (parent)
                self.depends[name] = (bname)
                self.forwards[bname] = self[bname]
                self.layers.append((bname, [parent], name))
            else:
                self.depends[name] = (parent)

        elif (v.creator.label == 'ReLU'):
            name = parent
        elif (v.creator.label == 'MaxPooling2D'):
            kw, kh = v.creator.kw, v.creator.kh
            sx, sy = v.creator.sx, v.creator.sy
            pw, ph = v.creator.pw, v.creator.ph
            name = 'maxpool' + v.rank
            self.depends[name] = (parent)
            self.forwards[name] = lambda x: F.unpooling_2d(x, (kh, kw), stride=(sy, sx), pad=(ph, pw))

        self.register_inv_layer(v.creator.inputs[0], name)
    else:
        depends['output'] = parent


class DeconvNet(chainer.Chain):
    """ deconvnet made from variable """

    def register_inv_layer(self, variable, parent):
        v = variable
        if(v.creator is not None):
            # Convolution -> Deconvolutionに変換
            if (v.creator.label == 'Convolution2DFunction'):
                print(v.creator.label, v.rank)
                convW = v.creator.inputs[1].data
                in_cn, out_cn = convW.shape[0], convW.shape[1] # in/out channels
                kh, kw = convW.shape[2], convW.shape[3] # kernel size
                sx, sy = v.creator.sx, v.creator.sy # stride
                pw, ph = v.creator.pw, v.creator.ph # padding
                name = 'conv' + v.rank # temporal layer name
                super(DeconvNet, self).add_link(name, L.Deconvolution2D(
                in_cn, out_cn, (kh, kw), stride=(sy, sx), pad=(ph, pw),
                nobias=True, initialW=convW))
                self.forwards[name] = self[name]
                # もし畳み込み層にバイアスがある場合、それも登録
                if len(v.creator.inputs) == 3:
                    b = v.creator.inputs[2].data
                    bname = 'convb' + v.rank
                    super(DeconvNet, self).add_link(bname, L.Bias(shape=b.shape))
                    self[bname].b.data = b
                    self.depends[bname] = (parent)
                    self.depends[name] = (bname)
                    self.forwards[bname] = self[bname]
                    self.layers.append((bname, [parent], name))
                else:
                    self.depends[name] = (parent)

            elif (v.creator.label == 'ReLU'):
                name = parent
            elif (v.creator.label == 'MaxPooling2D'):
                kw, kh = v.creator.kw, v.creator.kh
                sx, sy = v.creator.sx, v.creator.sy
                pw, ph = v.creator.pw, v.creator.ph
                name = 'maxpool' + v.rank
                self.depends[name] = (parent)
                self.forwards[name] = lambda x: F.unpooling_2d(x, (kh, kw), stride=(sy, sx), pad=(ph, pw))

            self.register_inv_layer(v.creator.inputs[0], name)
        else:
            depends['output'] = parent

    def __init__(self, variable):
        super(DeconvNet, self).__init__()
        v = variable
        self.depends = {}
        self.forwards = {}
        self.layers = []
        self.register_inv_layer(variable, 'data')
        self.train = False
        print(self.depends.items())
        print(self.functions.items())

    def __call__(self, x):
        variables = {'data': x}
        for func_name, bottom, top in self.layers:
            if (func_name not in self.forwards or
                    any(blob not in variables for blob in bottom)):
                continue

            func = self.forwards[func_name]
            input_vars = tuple(variables[blob] for blob in bottom)
            output_vars = func(*input_vars)
            if not isinstance(output_vars, collections.Iterable):
                output_vars = output_vars,
            for var, name in zip(output_vars, top):
                variables[name] = var

        self.variables = variables
        return tuple(variables[blob] for blob in outputs)
