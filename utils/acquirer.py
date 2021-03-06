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
import ioutil
import variableutil as Vutil
from tqdm import tqdm

class Acquirer(extensions.Evaluator):
    lastname = 'validation/main/loss'
    layer_rank = None
    layer_name = None
    operation = 'max'
    top = None
    n_features = None
    mean = None

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
            self.patch_image_dir = os.path.join(trainer.out, self.layer_name)
            if not os.path.exists(self.patch_image_dir):
                os.makedirs(self.patch_image_dir)
            result, locs, bounds = self.evaluate()

            outputdir = os.path.join(trainer.out, 'features')
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            #print(bounds)
            #self.savetxt(os.path.join(trainer.out, self.layer_name + '.txt'),
            #                features, delimiter='\t')
            #cupy.savez(os.path.join(trainer.out, self.layer_name + '.npz'),
            #                **{self.layer_name: features})
            if locs:
                self.save_tuple_list(os.path.join(outputdir,
                        'maxloc_' + self.layer_name + '.txt'), locs)
            if bounds:
                self.save_tuple_list(os.path.join(outputdir,
                        'maxbounds_' + self.layer_name + '.txt'), bounds)
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
        pbar = tqdm(total=len(iterator.dataset))
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
            pbar.update(len(batch))

            indices = np.arange(filter_idx, filter_idx + len(batch)) % self.n_features

            # deconv対象の層のVariableを取得
            layer_variable = Vutil.get_variable(
                observation[self.lastname], self.layer_rank)

            # 最大値の位置の計算に必要な入力層の領域を取得
            isfc = Vutil.has_fc_layer(layer_variable)
            if isfc:
                batch_bounds = Vutil.get_data_bounds(layer_variable)
            else:
                batch_bounds = Vutil.get_max_bounds(layer_variable, indices)

            topk = np.arange(n_processed, n_processed + len(batch)) // self.n_features
            input_data = Vutil.get_data_layer(layer_variable).data

            for k, f, d, b in zip(topk, indices, input_data, batch_bounds):
                #print(dir(d))
                # deconvされた入力層に平均画像を足して画像化
                img = ioutil.deprocess(d.get(), self.mean)
                # 最大値の位置の計算に必要な入力層の領域だけクロッピングして保存
                img.crop((b[0], b[2], b[1], b[3])).save(
                os.path.join(self.patch_image_dir,
                    "{0:0>4}_{1:0>2}.png".format(f, k)))

            if not isfc:
                max_locs.extend(Vutil.get_max_locs(layer_variable, indices))
                bounds.extend(batch_bounds)

            filter_idx = (filter_idx + len(batch)) % self.n_features
            n_processed += len(batch)
            #self.add_to_confmat(self.confmat, in_vars[1].data, self.getpred(observation[self.lastname]))
            summary.add(observation)
        pbar.close()
        #print(self.confmat)
        #print(np.diag(self.confmat))
        #print(1.0 * np.diag(self.confmat).sum() / self.confmat.sum())
        return summary.compute_mean(), max_locs, bounds
