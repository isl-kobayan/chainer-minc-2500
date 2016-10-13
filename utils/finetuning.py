import chainer
import chainer.link as link
from chainer.functions import caffe
import cPickle as pickle
import os.path

def copy_model(src, dst, ignore_layers=None):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        if ignore_layers is not None and child.name in ignore_layers: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print 'Ignore %s because of parameter mismatch' % child.name
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print 'Copy %s' % child.name

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
