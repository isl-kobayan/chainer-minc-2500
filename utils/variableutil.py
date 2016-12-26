#!/usr/bin/env python
# -*- coding: utf-8 -*-
import six

from chainer import cuda
import numpy as np
import cupy
'''
import chainer
from chainer.training import extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import extension
from chainer import variable


import os
import csv
import chainer.functions as F
'''

""" chainer.Variableに関する関数群 """

def print_computation_history(variable):
    """ ~chainer.Variableの計算過程を出力する
    Args:
        variable (~chainer.Variable): chainerで何らかの計算を行ったあとのデータ
    """
    v = variable
    # 今の位置から最初の計算までたどりながら計算過程を出力
    while (v.creator is not None):
        # rank(入力層から見て何回処理を行ったか)と、施した関数名を出力
        print(v.rank, v.creator.label)
        v = v.creator.inputs[0] # 1つ前をたどる

def get_variable(variable, rank):
    """ ~chainer.Variableの計算過程を辿り、特定のrankにおける~chainer.Variableを取得する
    Args:
        variable (~chainer.Variable): chainerで何らかの計算を行ったあとのデータ
        rank (int): 取得対象のrank
    Returns:
        ~chainer.Variable: 特定のrankにおける~chainer.Variable
    """
    assert rank <= variable.rank, 'rank should be smaller than variable.rank !'
    v = variable
    # 今の位置から最初の計算までたどりながら、rankが一致する箇所を探し、それを返す
    while (v.creator is not None):
        if v.rank == rank:
            return v.creator.outputs[0]()
        v = v.creator.inputs[0]

    return None

def get_data_layer(variable):
    """ ~chainer.Variableの計算過程を辿り、入力層を取得する
    Args:
        variable (~chainer.Variable): chainerで何らかの計算を行ったあとのデータ
    Returns:
        ~chainer.Variable: 入力層の~chainer.Variable
    """
    first_layer = get_variable(variable, 1)
    return first_layer.creator.inputs[0]

def get_features(variable, operation=None):
    if operation is None:
        return variable.data
    else:
        ax = (2,3) if len(variable.data.shape) == 4 else 1
        if operation == 'max':
            return variable.data.max(axis=ax)
        elif operation == 'mean':
            return variable.data.mean(axis=ax)

def get_related_bounds(variable, x, y):
    """ 特徴マップのある位置の計算と関連がある入力層の範囲を計算する
    Args:
        variable (~chainer.Variable): 中間層 (chainer.Variable)
        x (int): 位置 (x座標)
        y (int): 位置 (y座標)

    Returns:
        tuple(int, int, int, int): 範囲(left, right, top, bottom), left <= x < right, top <= y < bottom
    """
    # 範囲の初期値は(x, x, y, y)とする
    left = x; right = x
    top = y; bottom = y
    v = variable
    # 今の層から入力層に辿り着くまで繰り返す
    while (v.creator is not None):
        # 特徴マップのサイズを取得
        mapw = v.creator.inputs[0].data.shape[3]
        maph = v.creator.inputs[0].data.shape[2]

        if (v.creator.label == 'Convolution2DFunction'):
            kw = v.creator.inputs[1].data.shape[3] # kernel size
            kh = v.creator.inputs[1].data.shape[2] # kernel size
            sx = v.creator.sx; sy = v.creator.sy # stride
            pw = v.creator.pw; ph = v.creator.ph # padding
        elif (v.creator.label == 'MaxPooling2D'):
            kw = v.creator.kw; kh = v.creator.kh # kernel size
            sx = v.creator.sx; sy = v.creator.sy # stride
            pw = v.creator.pw; ph = v.creator.ph # padding
        else:
            # ReLUやLRN等の場合、範囲は広がらない
            kw = 1; kh = 1; sx = 1; sy = 1; pw = 0; ph = 0
        # 1つ前の層の範囲を計算
        left = sx * left - pw
        right = sx * right - pw + kw - 1
        top = sy * top - ph
        bottom = sy * bottom - ph + kh - 1
        # 計算によってはマップの範囲外になってしまうので、それを抑制する
        if left < 0: left = 0
        if right >= mapw: right = mapw - 1
        if top < 0: top = 0
        if bottom >= maph: bottom = maph - 1
        # 1つ前の層をたどる
        v = v.creator.inputs[0]

    return (left, right + 1, top, bottom + 1)

def get_max_locs(variable, channels):
    """ 特徴マップ内の最大値の位置を取得する

    中間層の特徴マップのindicesで指定されたチャンネルで、最大値の位置を取得します。
    最大値の位置は2次元で表現されます。
    i番目のデータから、channels[i]番目のマップ内の最大値を取得します。

    Args:
        variable (~chainer.Variable): 中間層 (chainer.Variable)
        channels list(int): 最大値の位置を取得するチャンネル

    Returns:
        list(tuple(int, int)): 最大値の位置. list[i] = (x, y)
    """

    assert len(variable.data.shape) == 4, 'variable should be 4d array.'

    b, c, h, w = variable.data.shape
    argmax = variable.data.reshape(b, c, -1).argmax(axis=2)
    locs=[]
    for (am, i) in zip(argmax, channels):
        loc = int(am[i])
        x = loc % w
        y = loc // w
        locs.append((x, y))
    return locs

def get_max_info(variable, channels):
    """ 特徴マップ内の最大値とその位置を取得する

    中間層の特徴マップのindicesで指定されたチャンネルで、最大値とその位置を取得します。
    最大値の位置は2次元で表現されます。
    i番目のデータから、channels[i]番目のマップ内の最大値を取得します。

    Args:
        variable (~chainer.Variable): 中間層 (chainer.Variable)
        channels list(int): 最大値の位置を取得するチャンネル

    Returns:
        list(tuple(int, int, int)): 最大値とその位置。list[i] = (x, y, maxval)
    """

    assert len(variable.data.shape) == 4, 'variable should be 4d array.'

    b, c, h, w = variable.data.shape
    argmax = variable.data.reshape(b, c, -1).argmax(axis=2)
    maxval = variable.data.max(axis=(2, 3))
    info=[]
    for (am, val, i) in zip(argmax, maxval, channels):
        loc = int(am[i])
        x = loc % w
        y = loc // w
        info.append((x, y, val[i]))
    return info

def get_fc_info(variable, indices):
    """ 全結合層の指定された位置のニューロンの値を取得する

    全結合層のindicesで指定された位置のニューロンの値を取得します。
    i番目のデータから、indices[i]番目のニューロンの値を取得します。

    Args:
        variable (~chainer.Variable): 全結合層 (chainer.Variable)
        indices list(int): 最大値の位置を取得するチャンネル

    Returns:
        list(int): ニューロンの値
    """

    assert len(variable.data.shape) == 2, 'variable should be 2d array.'

    info=[]
    for d, i in zip(variable.data, indices):
        info.append(d[i])
    return info

def get_max_bounds(variable, channels):
    """ 特徴マップ内の最大値の位置の計算に必要な入力層の範囲を取得する

    中間層の特徴マップのindicesで指定されたチャンネルで、最大値の位置を取得し、
    その位置のユニットの計算に必要な入力層の範囲を取得します。
    i番目のデータから、channels[i]番目のマップ内の最大値の位置を求め、範囲を取得します。
    範囲を取得にはget_related_bounds関数を使用しています。

    Args:
        variable (~chainer.Variable): 中間層 (chainer.Variable)
        channels list(int): 最大値の位置を取得するチャンネル

    Returns:
        list(tuple(int, int, int, int)): 範囲のリスト。list[i] = (left, right, top, bottom)
    """
    assert len(variable.data.shape) == 4, 'variable should be 4th dimension.'

    b, c, h, w = variable.data.shape
    argmax = variable.data.reshape(b, c, -1).argmax(axis=2)
    bounds=[]
    for (am, i) in zip(argmax, channels):
        loc = int(am[i])
        x = loc % w
        y = loc // w
        bounds.append(get_related_bounds(variable, x, y))
    return bounds

def get_data_bounds(variable):
    """ 入力層の範囲を取得する

    Args:
        variable (~chainer.Variable): 中間層 (chainer.Variable)

    Returns:
        list(tuple(int, int, int, int)): 範囲のリスト。list[i] = (left, right, top, bottom)
    """

    data_layer = get_data_layer(variable)
    b, c, h, w = data_layer.data.shape
    bounds=[]
    for i in range(b):
        bounds.append((0, w, 0, h))
    return bounds

def has_fc_layer(variable):
    """ ~chainer.Variableの計算過程の中に全結合層が含まれているか判定する

    Args:
        variable (~chainer.Variable): chainerで何らかの計算を行ったあとのデータ

    Returns:
        bool: 全結合層が含まれている場合はTrue, 含まれていなければFalse
    """
    v = variable
    # 今の位置から最初の計算までたどりながら、rankが一致する箇所を探し、それを返す
    while (v.creator is not None):
        if (v.creator.label == 'LinearFunction'):
            return True
        v = v.creator.inputs[0]

    return False

def get_RMS(x, axis=None):
    """ RMSを計算する

    RMS (Root Mean Square) を計算します。
    RMS = \sqrt(1/N \sum _(a \in x) a^2)

    Args:
        x (numpy.ndarray or cupy.ndarray): 多次元配列

    Returns:
        rms: RMS
    """
    xp = cuda.get_array_module(x)
    '''if xp == cupy:
        rms = cupy.sqrt(cupy.sum(x**2, axis=axis) / np.product(x.shape))
    else:
        rms = np.linalg.norm(x, axis=axis) ** 2 / np.product(x.shape)'''
    N = np.product(x.shape)
    if axis is not None:
        N = N / x.shape[axis]
    rms = xp.sqrt(xp.sum(x**2, axis=axis) / N)
    return rms

def get_argmax_N(X, N):
    xp = cuda.get_array_module(X)
    if xp is np:
        return np.argsort(X, axis=0)[::-1][:N]
    else:
        return np.argsort(X.get(), axis=0)[::-1][:N]

def invert_convolution(variable, rms=0.02, guided=True, ignore_bias=True):
    """ 畳み込み後の~chainer.Variableから、畳み込み前の状態を復元する
    Args:
        variable (~chainer.Variable): 畳み込み後の中間層
        rms (float): 値が0以上の場合、畳み込みのフィルタ重みのRMSが指定された値になるように、フィルタ重みを正規化します。
        guided (bool): guided backpropagation を行う場合はTrue、行わない場合はFalse.
        ignore_bias: バイアス項を無視する場合はTrue、考慮する場合はFalse.
    Returns:
        data (ndarray): 復元された畳み込み前の中間層のデータ(返されるのは~chainer.Variableではないことに注意)
    """
    assert variable.creator is not None
    assert variable.creator.label == 'Convolution2DFunction', 'variable.creator should be Convolution2DFunction.'
    v = variable
    bottom_blob = v.creator.inputs[0]

    # 畳み込みフィルタをRMSがfixed_RMSになるように正規化
    convW = v.creator.inputs[1].data
    xp = cuda.get_array_module(convW)

    scale = Vutil.get_RMS(convW) / rms if rms > 0 else 1
    '''if rms > 0:
        rmsW = Vutil.get_RMS(convW)
        scale = rmsW / rms
        #print(rmsW, scale)
    else:
        scale = 1'''

    convW = convW * scale

    # もし畳み込み層のバイアスを考慮する場合、先にバイアス分を引いておく
    if not ignore_bias and len(v.creator.inputs) == 3:
        in_data = F.bias(v, -v.creator.inputs[2] * scale)
    else:
        in_data = v

    in_cn, out_cn = convW.shape[0], convW.shape[1] # in/out channels
    kh, kw = convW.shape[2], convW.shape[3] # kernel size
    sx, sy = v.creator.sx, v.creator.sy # stride
    pw, ph = v.creator.pw, v.creator.ph # padding
    outsize = (bottom_blob.data.shape[2], bottom_blob.data.shape[3])

    # Deconvolution （転置畳み込み）
    deconv_data = F.deconvolution_2d(
        in_data, convW, stride=(sy, sx), pad=(ph, pw), outsize=outsize)
    # guided backpropagation
    if guided and v.rank > 1:
        # そもそも畳み込み前の値が0以下だったら伝搬させない
        switch = bottom_blob.data > 0
        deconv_data.data *= switch

    return deconv_data.data

def invert_relu(variable):
    """ ReLUを通った後の~chainer.Variableから、通る前の状態を復元する
    Args:
        variable (~chainer.Variable): ReLUを通った後の中間層
    Returns:
        data (ndarray): 復元されたReLUを通る前の中間層のデータ(返されるのは~chainer.Variableではないことに注意)
    """
    assert variable.creator is not None
    assert variable.creator.label == 'ReLU', 'variable.creator should be ReLU.'
    return F.relu(variable).data

def invert_maxpooling(variable, guided=True):
    """ Max pooling後の~chainer.Variableから、Max pooling前の状態を復元する
    Args:
        variable (~chainer.Variable): Max pooling後の中間層
        guided (bool): guided backpropagation を行う場合はTrue、行わない場合はFalse.
    Returns:
        data (ndarray): 復元されたMax pooling前の中間層のデータ(返されるのは~chainer.Variableではないことに注意)
    """
    assert variable.creator is not None
    assert variable.creator.label == 'MaxPooling2D', 'variable.creator should be MaxPooling2D.'
    v = variable
    bottom_blob = v.creator.inputs[0]

    kw, kh = v.creator.kw, v.creator.kh
    sx, sy = v.creator.sx, v.creator.sy
    pw, ph = v.creator.pw, v.creator.ph
    outsize = (bottom_blob.data.shape[2], bottom_blob.data.shape[3])

    # UnPooling
    unpooled_data = F.unpooling_2d(
        v, (kh, kw), stride=(sy, sx), pad=(ph, pw), outsize=outsize)

    # Max Location Switchesの作成（Maxの位置だけ1, それ以外は0）
    ## (1) Max pooling後のマップをNearest neighborでpooling前のサイズに拡大
    unpooled_max_map = F.unpooling_2d(
        F.max_pooling_2d(bottom_blob, (kh, kw), stride=(sy, sx), pad=(ph, pw)),
        (kh, kw), stride=(sy, sx), pad=(ph, pw), outsize=outsize)
    ## (2) 最大値と値が一致するところだけ1, それ以外は0 (Max Location Switches)
    pool_switch = unpooled_max_map.data == bottom_blob.data
    ## (3) そもそも最大値が0以下だったら伝搬させない (guided backpropagation)
    if guided:
        guided_switch = bottom_blob.data > 0
        pool_switch *= guided_switch

    # Max Location Switchesが1のところだけUnPoolingの結果を伝搬、それ以外は0
    return unpooled_data.data * pool_switch

def invert_linear(variable, rms=0.02, guided=True, ignore_bias=True):
    """ 全結合層を通った後の~chainer.Variableから、通る前の状態を復元する
    Args:
        variable (~chainer.Variable): 全結合層を通った後の中間層
        rms (float): 値が0以上の場合、重みのRMSが指定された値になるように、重みを正規化します。
        guided (bool): guided backpropagation を行う場合はTrue、行わない場合はFalse.
        ignore_bias: バイアス項を無視する場合はTrue、考慮する場合はFalse.
    Returns:
        data (ndarray): 復元された全結合層を通る前の中間層のデータ(返されるのは~chainer.Variableではないことに注意)
    """
    assert variable.creator is not None
    assert variable.creator.label == 'LinearFunction', 'variable.creator should be LinearFunction.'
    v = variable
    bottom_blob = v.creator.inputs[0]

    bshape = bottom_blob.data.shape
    W = v.creator.inputs[1].data
    scale = Vutil.get_RMS(W) / rms if rms > 0 else 1
    scale = 1
    W = W * scale

    # もし全結合層のバイアスを考慮する場合、先にバイアス分を引いておく
    if not ignore_bias and len(v.creator.inputs) == 3:
        in_data = F.bias(v, -v.creator.inputs[2] * scale)
    else:
        in_data = v

    inv_data = F.linear(in_data, W.T)

    # guided backpropagation
    if self.guided:
        # そもそも順伝搬時の値が0以下だったら伝搬させない
        switch = bottom_blob.data > 0
        inv_data.data *= switch.reshape(inv_data.data.shape)

    return inv_data.data.reshape(bshape)
