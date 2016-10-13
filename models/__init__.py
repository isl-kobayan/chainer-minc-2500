import nin
import alex
import alexbn
import googlenet
import googlenetbn
import vgg16
import vgg19
import squeezenet10
import squeezenet11
import minc_alex
import minc_googlenet
import minc_vgg16
import numpy as np

archs = {
    'alex': alex.Alex,
    'alexbn': alexbn.AlexBN,
    'googlenet': googlenet.GoogLeNet,
    'googlenetbn': googlenetbn.GoogLeNetBN,
    'vgg16': vgg16.VGG16,
    'vgg19': vgg19.VGG19,
    'nin': nin.NIN,
    'SqueezeNet10': squeezenet10.SqueezeNet10,
    'SqueezeNet11': squeezenet11.SqueezeNet11,
    'minc-alex': minc_alex.MINC_Alex,
    'minc-googlenet': minc_googlenet.MINC_GoogLeNet,
    'minc-vgg16': minc_vgg16.MINC_VGG16
}
