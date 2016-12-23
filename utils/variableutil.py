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
        if (v.creator.label == 'Linear'):
            return True
        v = v.creator.inputs[0]

    return False

def get_RMS(x):
    """ RMSを計算する

    RMS (Root Mean Square) を計算します。
    RMS = \sqrt(1/N \sum _(a \in x) a^2)

    Args:
        x (numpy.ndarray or cupy.ndarray): 多次元配列

    Returns:
        rms: RMS
    """
    xp = cuda.get_array_module(x)
    if xp == cupy:
        rms = cupy.sqrt(cupy.sum(x**2) / np.product(x.shape))
    else:
        rms = np.linalg.norm(x) ** 2 / np.product(x.shape)
    return rms

def get_argmax_N(self, X, N):
    xp = cuda.get_array_module(X)
    if xp is np:
        return np.argsort(X, axis=0)[::-1][:N]
    else:
        return np.argsort(X.get(), axis=0)[::-1][:N]
