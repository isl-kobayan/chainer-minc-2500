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
import os
from scipy.stats import rankdata
from tqdm import tqdm

class EvaluatorPlus(extensions.Evaluator):
    lastname = 'validation/main/loss'
    confmat = None

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

    def get_n_categories(self, loss):
        v = loss
        before_softmax=None
        while (v.creator is not None):
            if v.creator.label == 'SoftmaxCrossEntropy':
                before_softmax = v.creator.inputs[0]
                break
            v = v.creator.inputs[0]
        return before_softmax.data.shape[1]

    def add_to_confmat(self, confmat, truth, pred):
        for (t, p) in zip((int(x) for x in truth), (int(y) for y in pred)):
            confmat[t, p] += 1

    def getpred(self, loss):
        xp = cuda.get_array_module(loss.data)
        v = loss
        before_softmax=None
        while (v.creator is not None):
            if v.creator.label == 'SoftmaxCrossEntropy':
                before_softmax = v.creator.inputs[0]
                break
            v = v.creator.inputs[0]
        return xp.argmax(before_softmax.data, axis=1)

    def getranking(self, loss, t):
        xp = cuda.get_array_module(loss.data)
        v = loss
        before_softmax=None
        while (v.creator is not None):
            if v.creator.label == 'SoftmaxCrossEntropy':
                before_softmax = v.creator.inputs[0]
                break
            v = v.creator.inputs[0]
        if xp == np:
            data = before_softmax.data
        else:
            data = before_softmax.data.get()

        b = [rankdata(p[1], method='dense')[int(p[0])] for p in zip(t, -data)]

        return b

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

    def save_predictions(self, path, predictions):
        with open(path, 'w') as f:
            for batch in predictions:
                for p in batch:
                    f.write(str(p) + '\n')

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
            result, predictions, rankings = self.evaluate()
            #print(result)
            #print(predictions)
            self.save_predictions(os.path.join(trainer.out, 'pred.txt'), predictions)
            self.save_predictions(os.path.join(trainer.out, 'ranking.txt'), rankings)

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
        predictions = []
        rankings = []
        n_categories = 0
        pbar = tqdm(total=len(it.dataset))
        self.confmat = None
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

            if n_categories == 0:
                n_categories = self.get_n_categories(observation[self.lastname])
                self.confmat = np.zeros((n_categories, n_categories), dtype=np.int32)
                #self.printloss(observation[self.lastname])
                #print(self.getpred(observation[self.lastname]))

            predictions.append(self.getpred(observation[self.lastname]))
            rankings.append(self.getranking(observation[self.lastname], in_vars[1].data))
            self.add_to_confmat(self.confmat, in_vars[1].data, self.getpred(observation[self.lastname]))
            summary.add(observation)
            pbar.update(len(batch))
        #print(self.confmat)
        #print(np.diag(self.confmat))
        #print(1.0 * np.diag(self.confmat).sum() / self.confmat.sum())
        return summary.compute_mean(), predictions, rankings
