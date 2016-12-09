#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import six

import chainer
from chainer.training import extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import extension
from chainer import variable
from chainer import cuda
import numpy as np
import cupy
import os
import csv
import chainer.functions as F
import ioutil
import math

class DeconvAcquirer(extensions.Evaluator):
    lastname = 'validation/main/loss'
    layer_rank = None
    layer_name = None
    operation = 'max'
    top = None
    n_features = None
    mean = None
    switch_1stlayer = False


    '''trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, link.Link):
            target = {'main': target}
        self._targets = target

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func'''

    def printloss(self, loss):
        v = loss
        while (v.creator is not None):
            print(v.rank, v.creator.label)
            v = v.creator.inputs[0]

    def getVariableFromLoss(self, loss, rank):
        if loss is None:
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
        v = loss
        while (v.creator is not None):
            if v.rank == rank:
                return v.creator.outputs[0]()
            v = v.creator.inputs[0]
        return None

    def get_variable(self, loss, rank):
        v = loss
        while (v.creator is not None):
            if v.rank == rank:
                return v.creator.outputs[0]()
            v = v.creator.inputs[0]
        return None

    def get_patch_bounds(self, variable, x, y):
        left = x; right = x
        top = y; bottom = y
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

    def get_max_locs(self, loss, rank, indices):
        variable = self.get_variable(loss, rank)
        ax = (2,3) if len(variable.data.shape) == 4 else 1
        w = variable.data.shape[3]
        argmax = variable.data.argmax(axis=ax)
        locs=[]
        for (am, i) in zip(argmax, indices):
            loc = int(am[i])
            x = loc % w
            y = loc // w
            locs.append((x, y))
        return locs

    def get_max_info(self, loss, rank, indices):
        variable = self.get_variable(loss, rank)
        ax = (2,3) if len(variable.data.shape) == 4 else 1
        w = variable.data.shape[3]
        argmax = variable.data.argmax(axis=ax)
        maxval = variable.data.max(axis=ax)
        info=[]
        for (am, val, i) in zip(argmax, maxval, indices):
            loc = int(am[i])
            x = loc % w
            y = loc // w
            info.append((x, y, val[i]))
        return info

    def get_max_patch_bounds(self, loss, rank, indices):
        variable = self.get_variable(loss, rank)
        ax = (2,3) if len(variable.data.shape) == 4 else 1
        w = variable.data.shape[3]
        argmax = variable.data.argmax(axis=ax)
        bounds=[]
        for (am, i) in zip(argmax, indices):
            loc = int(am[i])
            x = loc % w
            y = loc // w
            bounds.append(self.get_patch_bounds(variable, x, y))
        return bounds

    def deconv(self, variable):
        v = variable
        fixed_RMS = 0.020
        if(v.creator is not None):
            bottom_blob = v.creator.inputs[0]
            print(v.creator.label, v.rank)
            # Convolution -> Deconvolutionに変換
            if (v.creator.label == 'Convolution2DFunction'):
                # 畳み込みフィルタをRMSがfixed_RMSになるように正規化
                convW = v.creator.inputs[1].data
                xp = cuda.get_array_module(convW)

                if xp == cupy:
                    rms = cupy.sqrt(cupy.sum(convW ** 2) / np.product(convW.shape))
                    #rms = cupy.sqrt(cupy.sum(convW ** 2, axis=(2, 3)) / np.product(convW.shape[2:]))
                else:
                    rms = np.linalg.norm(convW) ** 2 / np.product(convW.shape)
                    #rms = np.linalg.norm(convW, axis=(2, 3)) ** 2 / np.product(convW.shape[2:])
                scale = fixed_RMS / rms
                #if v.rank != 1:
                #    scale = 1
                print(rms, scale)
                #scale = 1
                #scale = min(scale, 1.5)
                convW = convW * scale

                # もし畳み込み層にバイアスがある場合、先にバイアス分を引いておく
                if len(v.creator.inputs) == 3:
                    in_data = F.bias(v, -v.creator.inputs[2] * scale)
                else:
                    in_data = v

                in_cn, out_cn = convW.shape[0], convW.shape[1] # in/out channels
                kh, kw = convW.shape[2], convW.shape[3] # kernel size
                sx, sy = v.creator.sx, v.creator.sy # stride
                pw, ph = v.creator.pw, v.creator.ph # padding
                outsize = (bottom_blob.data.shape[2], bottom_blob.data.shape[3])

                # Deconvolution （転置畳み込み）
                deconv_data = F.deconvolution_2d(
                    in_data, convW, stride=(sy, sx), pad=(ph, pw), outsize=outsize)
                # そもそも畳み込み前の値が0以下だったら伝搬させない
                switch = bottom_blob.data > 0
                if v.rank > 1 or switch_1stlayer:
                    deconv_data.data *= switch

                bottom_blob.data = deconv_data.data
                self.deconv(bottom_blob)
            # relu -> relu
            elif (v.creator.label == 'ReLU'):
                bottom_blob.data = F.relu(v).data
                self.deconv(bottom_blob)
            # Pooling -> UnPooling
            elif (v.creator.label == 'MaxPooling2D'):
                kw, kh = v.creator.kw, v.creator.kh
                sx, sy = v.creator.sx, v.creator.sy
                pw, ph = v.creator.pw, v.creator.ph
                outsize = (bottom_blob.data.shape[2], bottom_blob.data.shape[3])

                # UnPooling
                unpooled_data = F.unpooling_2d(
                    v, (kh, kw), stride=(sy, sx), pad=(ph, pw), outsize=outsize)

                # Max Location Switchesの作成（Maxの位置だけ1, それ以外は0）
                ## (1) Max pooling後のマップをNearest neighborでpooling前のサイズに拡大
                unpooled_max_map = F.unpooling_2d(
                    F.max_pooling_2d(bottom_blob, (kh, kw), stride=(sy, sx), pad=(ph, pw)),
                    (kh, kw), stride=(sy, sx), pad=(ph, pw), outsize=outsize)
                ## (2) 最大値と値が一致するところだけ1, それ以外は0 (Max Location Switches)
                pool_switch = unpooled_max_map.data == bottom_blob.data
                ## (3) そもそも最大値が0以下だったら伝搬させない
                lt0_switch = bottom_blob.data > 0
                pool_switch *= lt0_switch

                # Max Location Switchesが1のところだけUnPoolingの結果を伝搬
                bottom_blob.data = unpooled_data.data * pool_switch
                self.deconv(bottom_blob)

    def get_deconv(self, loss, rank, indices):
        variable = self.get_variable(loss, rank)
        # 1. 最も活性した場所以外を0にする
        maxinfo = self.get_max_info(loss, rank, indices)
        #maxbounds = self.get_max_patch_bounds(loss, rank, indices)
        variable.data.fill(0)
        for i, (c, info) in enumerate(zip(indices, maxinfo)):
            variable.data[i, c, info[1], info[0]] = info[2]

        # 2. 入力層まで逆操作を繰り返す
        #variable.backward(retain_grad=True)
        first_layer = self.get_variable(loss, 1)
        data_layer = first_layer.creator.inputs[0]

        xp = cuda.get_array_module(data_layer.data)

        fixed_RMS = 300
        if xp == cupy:
            rms = cupy.sqrt(cupy.sum(data_layer.data ** 2, axis=(1,2,3)) / np.product(data_layer.data.shape[1:]))
            #rms = cupy.sqrt(cupy.sum(convW ** 2, axis=(2, 3)) / np.product(convW.shape[2:]))
        else:
            rms = np.linalg.norm(data_layer.data, axis=(1,2,3)) ** 2 / np.product(data_layer.data.shape[1:])
            #rms = np.linalg.norm(convW, axis=(2, 3)) ** 2 / np.product(convW.shape[2:])
        scale = fixed_RMS / rms
        scale = scale.reshape(-1,1,1,1)
        #print(rms, scale)
        #data_layer.data *= scale

        #print(dir(first_layer))
        #print(first_layer.inputs)
        #print('before', data_layer.data[0])
        self.deconv(variable)
        #first_layer = self.get_variable(loss, 1)
        #print('after', data_layer.data[0])
        return data_layer.data


    def get_features(self, loss, rank, operation=None):
        variable = self.get_variable(loss, rank)
        if operation is None:
            return variable.data
        else:
            ax = (2,3) if len(variable.data.shape) == 4 else 1
            if operation == 'max':
                return variable.data.max(axis=ax)
            elif operation == 'argmax':
                return variable.data.argmax(axis=ax)
            elif operation == 'mean':
                return variable.data.mean(axis=ax)

    def savetxt(self, fname, X, fmt='%.18e', delimiter='', newline='\n', header='', footer='', comments='#'):
        xp = cuda.get_array_module(X)
        if xp is np:
            np.savetxt(fname, X, fmt, delimiter, newline, header, footer, comments)
        else:
            np.savetxt(fname, X.get(), fmt, delimiter, newline, header, footer, comments)

    def save_tuple_list(self, fname, data, delimiter='\t'):
        with open(fname,'w') as out:
            csv_out=csv.writer(out, delimiter=delimiter)
            for row in data:
                csv_out.writerow(row)

    def get_argmax_N(self, X, N):
        xp = cuda.get_array_module(X)
        if xp is np:
            return np.argsort(X, axis=0)[::-1][:N]
        else:
            return np.argsort(X.get(), axis=0)[::-1][:N]

    def __call__(self, trainer):
        """override method of extensions.Evaluator."""
        # set up a reporter
        reporter = reporter_module.Reporter()
        if hasattr(self, 'name'):
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        with reporter:
            self.deconv_image_dir = os.path.join(trainer.out, 'deconv_' + self.layer_name)
            if not os.path.exists(self.deconv_image_dir):
                os.makedirs(self.deconv_image_dir)
            result, locs, bounds = self.evaluate()
            if not os.path.exists(trainer.out):
                os.makedirs(trainer.out)
            #print(bounds)
            #self.savetxt(os.path.join(trainer.out, self.layer_name + '.txt'),
            #                features, delimiter='\t')
            #cupy.savez(os.path.join(trainer.out, self.layer_name + '.npz'),
            #                **{self.layer_name: features})
            '''self.save_tuple_list(os.path.join(trainer.out,
                        'maxloc_' + self.layer_name + '.txt'), locs)
            self.save_tuple_list(os.path.join(trainer.out,
                        'maxbounds_' + self.layer_name + '.txt'), bounds)'''
        reporter_module.report(result)
        return result

    def evaluate(self):
        """override method of extensions.Evaluator."""

        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()
        max_locs = []
        bounds = []
        n_processed = 0
        filter_idx = 0
        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                if isinstance(in_arrays, tuple):
                    in_vars = tuple(variable.Variable(x, volatile='off')
                                    for x in in_arrays)
                    eval_func(*in_vars)
                elif isinstance(in_arrays, dict):
                    in_vars = {key: variable.Variable(x, volatile='off')
                               for key, x in six.iteritems(in_arrays)}
                    eval_func(**in_vars)
                else:
                    in_var = variable.Variable(in_arrays, volatile='off')
                    eval_func(in_var)

            indices = np.arange(filter_idx, filter_idx + len(batch)) % self.n_features
            #print('x', in_vars[0].data[0])
            bounds = self.get_max_patch_bounds(
                observation[self.lastname], self.layer_rank, indices)
            deconv_data = self.get_deconv(
                observation[self.lastname], self.layer_rank, indices)

            topk = np.arange(n_processed, n_processed + len(batch)) // self.n_features

            for k, f, d, b in zip(topk, indices, deconv_data, bounds):
                #print(dir(d))
                img = ioutil.deprocess(d.get(), self.mean)
                img.crop((b[0], b[2], b[1], b[3])).save(
                os.path.join(self.deconv_image_dir,
                    "{0:0>4}_{1:0>2}.png".format(f, k)))

            '''max_locs.extend(self.get_max_locs(
                observation[self.lastname], self.layer_rank, indices))
            bounds.extend(self.get_max_patch_bounds(
                observation[self.lastname], self.layer_rank, indices))'''
            filter_idx = (filter_idx + len(batch)) % self.n_features
            n_processed += len(batch)
            '''if max_loc is None:
                max_loc = self.get_features(
                    observation[self.lastname], self.layer_rank, 'argmax')
                #print(batch)
                print(max_loc)
            else:
                xp = cuda.get_array_module(max_loc)
                max_loc = xp.vstack((max_loc, self.get_features(
                    observation[self.lastname], self.layer_rank, 'argmax')))'''

            #self.add_to_confmat(self.confmat, in_vars[1].data, self.getpred(observation[self.lastname]))
            summary.add(observation)
        #print(self.confmat)
        #print(np.diag(self.confmat))
        #print(1.0 * np.diag(self.confmat).sum() / self.confmat.sum())
        return summary.compute_mean(), max_locs, bounds
