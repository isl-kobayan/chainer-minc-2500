import chainer
import chainer.link as link
import chainer.links as L
from chainer.functions import caffe
import cPickle as pickle
import os.path
import numpy as np

def copy_model(src, dst, ignore_layers=None):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        if ignore_layers is not None and child.name in ignore_layers: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        #if isinstance(child, link.Chain):
        #    print(child.name, 'recall'); copy_model(child, dst_child)
        if isinstance(child, link.Link):
            if isinstance(child, L.BatchNormalization):
                #dst_child.decay = child.decay
                #dst_child.eps = max(child.eps, 2e-5)
                #dst_child.start_finetuning()
                src_scale_layer = src[child.name + '/sc']
                dst_child.gamma.data = src_scale_layer.W.data
                dst_child.beta.data = src_scale_layer.bias.b.data
                #print ('copy batch normalization ' + child.name)

            # copy params
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False; print (a[1].data.shape, b[1].data.shape)
                    break
            if not match:
                print 'Ignore %s because of parameter mismatch' % child.name
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data; #print('copy param ' + a[0])
            # copy persistents
            match_persistents = True
            for a, b in zip(child._persistent, dst_child._persistent):
                if a[0] != b[0]:
                    match_persistents = False
                    break
                if isinstance(child.__dict__[a], np.ndarray) and child.__dict__[a].shape != dst_child.__dict__[b].shape:
                    match_persistents = False; #print (child.__dict__[a].shape, dst_child.__dict__[b].shape)
                    break
            if not match_persistents:
                print 'Ignore %s because of persistent mismatch' % child.name
                continue
            #for a, b in zip(child._persistent, dst_child._persistent):
            #    dst_child.__dict__[b] = child.__dict__[a]; #print('copy persistent ' + a)

            print("Copy {0}  (param:{1}  persistent:{2})".format(child.name, child._params, child._persistent))

def copy_model_old(src, dst, ignore_layers=None):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        if ignore_layers is not None and child.name in ignore_layers: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        #if isinstance(child, link.Chain):
        #    print(child.name, 'recall'); copy_model(child, dst_child)
        if isinstance(child, link.Link):
            if isinstance(child, L.BatchNormalization):
                dst_child.decay = child.decay
                dst_child.eps = max(child.eps, 2e-5)
                dst_child.start_finetuning()

            # copy params
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False; print (a[1].data.shape, b[1].data.shape)
                    break
            if not match:
                print 'Ignore %s because of parameter mismatch' % child.name
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data; #print('copy param ' + a[0])
            # copy persistents
            match_persistents = True
            for a, b in zip(child._persistent, dst_child._persistent):
                if a[0] != b[0]:
                    match_persistents = False
                    break
                if isinstance(child.__dict__[a], np.ndarray) and child.__dict__[a].shape != dst_child.__dict__[b].shape:
                    match_persistents = False; #print (child.__dict__[a].shape, dst_child.__dict__[b].shape)
                    break
            if not match_persistents:
                print 'Ignore %s because of persistent mismatch' % child.name
                continue
            #for a, b in zip(child._persistent, dst_child._persistent):
            #    dst_child.__dict__[b] = child.__dict__[a]; #print('copy persistent ' + a)

            print("Copy {0}  (param:{1}  persistent:{2})".format(child.name, child._params, child._persistent))

def change_ext(path, ext):
    dirname = os.path.dirname(path)
    filename = os.path.basename(path)
    fname, _ = os.path.splitext(filename)
    return os.path.join(dirname, fname + ext)

def makepkl(path, model):
    pklfilepath = change_ext(path, '.pkl')
    pickle.dump(model, open(pklfilepath, 'wb'))

def load_param(path, obj, ignore_layers=None):
    src = None

    # load .pkl if exists
    if os.path.isfile(change_ext(path, '.pkl')):
        with open(change_ext(path, '.pkl'), 'rb') as f:
            src = pickle.load(f)
    # load caffemodel and save pkl (if .pkl file doesn't exist)
    else:
        src = caffe.CaffeFunction(change_ext(path, '.caffemodel'))
        pickle.dump(src, open(change_ext(path, '.pkl'), 'wb'))

    copy_model(src, obj, ignore_layers)
