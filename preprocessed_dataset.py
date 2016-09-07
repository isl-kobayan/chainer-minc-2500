import chainer
import os
from PIL import Image
import random
import numpy

class LabeledBGRImageDataset(chainer.datasets.LabeledImageDataset):

    """add ".convert('RGB')" to get_example() function."""

    def get_example(self, i):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)
        with Image.open(full_path).convert('RGB') as f:
            image = numpy.asarray(f, dtype=self._dtype)
        label = numpy.array(int_label, dtype=self._label_dtype)
        return image.transpose(2, 0, 1)[::-1], label

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = LabeledBGRImageDataset(path, root)
        self.mean = mean
        self.crop_size = crop_size
        self.random = random

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
