import chainer
import os
import numpy
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e
import six
import random

class LabeledBGRImageDataset(chainer.datasets.LabeledImageDataset):

    def __init__(self, pairs, root='.', dtype=numpy.float32,
                 label_dtype=numpy.int32, indices=None):
        _check_pillow_availability()
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split()
                    if len(pair) != 2:
                        raise ValueError(
                            'invalid format at line {} in file {}'.format(
                                i, pairs_path))
                    pairs.append((pair[0], int(pair[1])))
        if indices is not None:
            if isinstance(indices, six.string_types):
                indices_path = indices
                indices = numpy.loadtxt(indices_path, delimiter="\t", dtype='i')
            #print(indices)
            #print(indices.shape)
            self._indices = indices
            self._cols=indices.shape[1]
            self._rows=indices.shape[0]

            all_pairs = pairs
            pairs = [all_pairs[i] for i in indices.flatten()]
        self._pairs = pairs
        self._root = root
        self._dtype = dtype
        self._label_dtype = label_dtype

    """add ".convert('RGB')" to get_example() function."""

    def get_example(self, i):
        path, int_label = self._pairs[i]
        #print(i, path, int_label)
        full_path = os.path.join(self._root, path)
        with Image.open(full_path).convert('RGB') as f:
            image = numpy.asarray(f, dtype=self._dtype)
        label = numpy.array(int_label, dtype=self._label_dtype)
        return image.transpose(2, 0, 1)[::-1], label

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True, indices=None):
        self.base = LabeledBGRImageDataset(path, root, indices=indices)
        self.mean = mean
        self.crop_size = crop_size
        self.random = random
        if indices is not None:
            self.indices = self.base._indices
            self.cols = self.base._cols
            self.rows = self.base._rows

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        if self.mean is not None:
            image -= self.mean[:, top:bottom, left:right]
        #image /= 255
        return image, label

def _check_pillow_availability():
    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))
