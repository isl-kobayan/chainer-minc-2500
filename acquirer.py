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

class Acquirer(extensions.Evaluator):
    lastname = 'validation/main/loss'
    layer_rank = None
    layer_name = None
    operation = 'max'
    top = None
    n_features = None

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
            result, locs, bounds = self.evaluate()
            if not os.path.exists(trainer.out):
                os.makedirs(trainer.out)
            #print(bounds)
            #self.savetxt(os.path.join(trainer.out, self.layer_name + '.txt'),
            #                features, delimiter='\t')
            #cupy.savez(os.path.join(trainer.out, self.layer_name + '.npz'),
            #                **{self.layer_name: features})
            self.save_tuple_list(os.path.join(trainer.out,
                        'maxloc_' + self.layer_name + '.txt'), locs)
            self.save_tuple_list(os.path.join(trainer.out,
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
            
            max_locs.extend(self.get_max_locs(
                observation[self.lastname], self.layer_rank, indices))
            bounds.extend(self.get_max_patch_bounds(
                observation[self.lastname], self.layer_rank, indices))
            filter_idx = (filter_idx + len(batch)) % self.n_features
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
