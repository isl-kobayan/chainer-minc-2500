# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import random
import os
import six
import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_image(path, insize, mean_image=None, center=False, flip=False):
    u"""Image reading function.
    This function reads image for Convolutional Neural Networls.
    Data is transposed to (channel, height, width).
    Args:
        path (string): Path of image.
        insize (int): Crop size.
        mean_image (numpy.ndarray): Mean image. Shape of this should be (3, 256, 256).
        center (bool): If ``True``, image is cropped from center. If ``False``,
            image is cropped randomly.
        flip (bool): If ``True``, image is flipped horizontaly with a 50% possibility.
    Returns:
        numpy.ndarray: Cropped image. Shape will be (3, insize, insize).
    """
    image = np.asarray(Image.open(path).convert('RGB')).transpose(2, 0, 1)[::-1]

    cropwidth = image.shape[1] - insize
    if center:
        top = left = cropwidth // 2
    else:
        top = random.randint(0, cropwidth - 1)
        left = random.randint(0, cropwidth - 1)
    bottom = insize + top
    right = insize + left

    image = image[:, top:bottom, left:right].astype(np.float32)

    if mean_image is not None:
        image -= mean_image[:, top:bottom, left:right]

    if flip and random.randint(0, 1) == 0:
        return image[:, :, ::-1]
    else:
        return image

def read_image_patch(path, insize, pl, pr, pt, pb):
    # Data loading routine
    image = np.asarray(Image.open(path).convert('RGB')).transpose(2, 0, 1)[::-1]

    cropwidth = image.shape[1] - insize
    top = left = cropwidth // 2
    bottom = insize + top
    right = insize + left

    image = image[:, top:bottom, left:right].astype(np.float32)
    image = image[:, int(pt):int(pb), int(pl):int(pr)]
    return image

def read_crop_image(path, insize, mean_image=None, flip=False):
    # get image data as np.float32
    image = np.asarray(Image.open(path).convert('RGB')).transpose(2, 0, 1)[::-1]

    cropwidth = image.shape[1] - insize
    #if center:
    #    top = left = cropwidth / 2
    #    else:
    top = random.randint(0, cropwidth - 1)
    left = random.randint(0, cropwidth - 1)
    bottom = insize + top
    right = insize + left

    image = image[:, top:bottom, left:right].astype(np.float32)
    if mean_image is not None:
        image -= mean_image[:, top:bottom, left:right]
    #image /= 255
    if flip and random.randint(0, 1) == 0:
        return image[:, :, ::-1]
    else:
        return image

def load_categories(path):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append(pair[0])
    return tuples

def load_image_list(path, root='.'):
    u"""Load image path and truth category.
    This function reads image path and correct label from space-separated text file.
    The first column is image path and second column is zero-origin label.
    Example of text file:
        cat01.jpg   0
        dog01.jpg   1
        bird01.jpg  2
        ...
    "0", "1", "2" indicates "cat", "dog", "bird" respectively.
    Args:
        path (string): Path of image.
        root (string): Root directory. image path becomes as "(root)/cat01.jpg".
    Returns:
        tuple<string, int>: Tuples of image path and label number.
    """
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append((os.path.join(root, pair[0]), np.int32(pair[1])))
    return tuples

def save_confmat_csv(path, matrix, labels):
    n_categories = matrix.shape[0]
    content = 'true\\pred'
    for labelname in labels:
        content = content + '\t' + labelname
    content = content + '\n'
    for true_idx in six.moves.range(n_categories):
        content = content + labels[true_idx]
        for pred_idx in six.moves.range(n_categories):
            content = content + '\t' + str(int(matrix[true_idx, pred_idx]))
        content = content + '\n'
    with open(path, 'w') as f:
        f.write(content)

def append_acts(path, imagepath, acts):
    content = imagepath + '\t'
    for i in six.moves.range(len(acts)):
       content = content + str(acts[i]) + '\t'
    content = content + '\n'
    with open(path, 'a') as f:
        f.write(content)

def get_act_table(path, val_list, acts, rank=10):
    maxargs=None
    for r in six.moves.range(rank):
        maxarg = np.argmax(acts, axis=0)
        if maxargs is None:
            maxargs=maxarg.reshape((1, -1))
        else:
            maxargs=np.r_[maxargs, maxarg.reshape((1, -1))]
        for i in six.moves.range(len(maxarg)):
            acts[maxarg[i],i]=0
    maxargs = maxargs.transpose()
    content=''
    for r in six.moves.range(maxargs.shape[0]):
        for i in six.moves.range(maxargs.shape[1]):
            content = content + val_list[maxargs[r,i]][0] + '\t'
        content = content + '\n'
    with open(path, 'a') as f:
        f.write(content)



# save confusion matrix as .png image
def save_confmat_fig0(matrix, savename, labels):
    norm_conf = []
    for i in matrix:
        #print(i)
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                interpolation='nearest')

    width = len(matrix)
    height = len(matrix[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(matrix[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])
    plt.savefig(savename, format='png')


def save_confmat_fig(savename, matrix, labels,
                     xlabel=None, ylabel=None, saveFormat="png",
                     title=None, clim=(None,None), mode="vote",
                     cmap=plt.cm.Blues, hide_zero=True):
    if mode=="rate":
        conf_rate = []
        for i in matrix:
            tmp_arr = []
            total = float(sum(i))
            for j in i:
                if total == 0:
                    tmp_arr.append(float(j))
                else:
                    tmp_arr.append(float(j)/total)
            conf_rate.append(tmp_arr)
        matrix = conf_rate
    norm_conf = []
    for i in matrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            if a == 0:
                tmp_arr.append(float(j))
            else:
                tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    fig = plt.figure(figsize=(8*2, 6*2))
    plt.clf()
    plt.subplots_adjust(top=0.85) # use a lower number to make more vertical space
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    if mode == "rate":
        res = plt.imshow(np.array(norm_conf)*100, cmap=cmap,
                        interpolation='nearest')
        plt.clim(0,100)
        threshold = 0.5
    else:
        res = plt.imshow(np.array(norm_conf), cmap=cmap,
                        interpolation='nearest')
        if clim!=(None,None):
            plt.clim(*clim)
        threshold = np.mean([np.max(norm_conf),np.min(norm_conf)])
    width = len(matrix)
    height = len(matrix[0])

    for x in xrange(width):
        for y in xrange(height):
            if norm_conf[x][y]>=threshold:
                textcolor = '1.0'
            else:
                textcolor = '0.0'
            if mode == "rate":
                ax.annotate("" if hide_zero and matrix[x][y] == 0 else "{0:.1f}".format(matrix[x][y]*100), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center', color=textcolor, fontsize=8)
            else:
                ax.annotate("{0}".format(matrix[x][y]) if matrix[x][y] == 0 else "", xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center', color=textcolor, fontsize=15)

    cb = fig.colorbar(res)
    cb.ax.tick_params(labelsize=15)
    if title != None:
        plt.text(0.5, 1.08, title,
                 horizontalalignment='center',
                 fontsize=20,
                 transform = ax.transAxes)
    ax.xaxis.tick_top()
    plt.xticks(range(width), labels[:width], rotation=45, rotation_mode='anchor', ha='left', fontsize=15)
    plt.yticks(range(height), labels[:height], fontsize=15)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
        ax.xaxis.set_label_position("top")
    plt.savefig(savename, format=saveFormat)
    plt.close(fig)

def save_pca_scatter_fig(data, savename, labels,
                     xlabel=None, ylabel=None, saveFormat="png",
                     title=None, cmap=plt.cm.Blues):
    import sklearn.decomposition
    import matplotlib.cm as cm
    markers=['x', '+']
    data=data.reshape((len(data), -1))
    pca = sklearn.decomposition.PCA(n_components = 2)
    pca.fit(data)
    result = pca.transform(data)
    np.savetxt(savename + ".result.csv", result, delimiter=",")
    print(result.shape)
    size1 = len(data)/len(labels)
    result = result.reshape((len(labels), size1, 2))
    fig = plt.figure()
    plt.clf()
    for c in six.moves.range(len(labels)):
        #print(result[c, :, 0].shape)
        #print(result[c, :, 0])
        plt.scatter(result[c, :, 0], result[c, :, 1], marker=markers[c%len(markers)], color=cm.jet(float(c) / len(labels)), label=labels[c])
    plt.legend(scatterpoints=1)
    plt.savefig(savename, format=saveFormat)
    plt.close()
