import os
import shutil
import tempfile
import ioutil

from chainer.serializers import npz
from chainer.training import extension
from chainer import cuda
'''
def image_snapshot(paramname='img', mean='None',
             filename='snapshot_iter_{.updater.iteration}',
             trigger=(1, 'epoch')):

    @extension.make_extension(trigger=trigger, priority=-100)
    def image_snapshot(trainer):
        _image_snapshot(trainer, trainer, filename.format(trainer), paramname, mean)

    return _image_snapshot


def _image_snapshot(trainer, target, filename, paramname, mean):'''



class ImageSnapshot(extension.Extension):
    def __init__(self, target, paramname='img', mean='None',
                 filename='snapshot_iter_{.updater.iteration}'):
        self.filename = filename
        self.mean = mean
        self.paramname = paramname
        self.target = target
    def __call__(self, trainer):
        fn = self.filename.format(trainer)
        img = ioutil.deprocess(cuda.to_cpu(self.target[self.paramname].W.data)[0], self.mean)
        img.save(os.path.join(trainer.out, fn))
        '''prefix = 'tmp' + fn
        fd, tmppath = tempfile.mkstemp(prefix=prefix, dir=trainer.out)
        try:

        except Exception:
            os.close(fd)
            os.remove(tmppath)
            raise
        os.close(fd)
        shutil.move(tmppath, )'''
