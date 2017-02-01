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
import ioutil
import variableutil as Vutil
from tqdm import tqdm

class Extractor(extensions.Evaluator):
    lastname = 'validation/main/loss'
    layer_rank = None
    layer_name = None
    operation = 'max'
    top = None
    save_features = False

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
            result, features = self.evaluate()
            outputdir = os.path.join(trainer.out, 'features')
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            #ioutil.savetxt(os.path.join(trainer.out, self.layer_name + '.txt'),
            #                features, delimiter='\t')
            #cupy.savez(os.path.join(trainer.out, self.layer_name + '.npz'),
            #                **{self.layer_name: features})
            if self.save_features:
                cupy.save(os.path.join(outputdir, self.layer_name + '.npy'),
                        features)

            if self.top is not None:
                top_N_args = Vutil.get_argmax_N(features, self.top)
                #print(top_N_args)
                ioutil.savetxt(os.path.join(outputdir,
                            'top_' + self.layer_name + '.txt'), top_N_args,
                            fmt='%d', delimiter='\t')
                #np.savez(os.path.join(trainer.out,
                #            'top_' + self.layer_name + '.npz'),
                #            **{self.layer_name: top_N_args})
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
        features = None
        max_loc = None
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

            # deconv対象の層のVariableを取得
            layer_variable = Vutil.get_variable(
                observation[self.lastname], self.layer_rank)

            if features is None:
                features = Vutil.get_features(
                    layer_variable, self.operation)
            else:
                xp = cuda.get_array_module(features)
                features = xp.vstack((features, Vutil.get_features(
                    layer_variable, self.operation)))
            #self.add_to_confmat(self.confmat, in_vars[1].data, self.getpred(observation[self.lastname]))
            summary.add(observation)
        pbar.close()
        #print(self.confmat)
        #print(np.diag(self.confmat))
        #print(1.0 * np.diag(self.confmat).sum() / self.confmat.sum())
        return summary.compute_mean(), features
