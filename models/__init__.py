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

'''NIN = nin.NIN
Alex = alex.Alex
AlexBN = alexbn.AlexBN
GoogLeNet = googlenet.GoogLeNet
GoogLeNetBN = googlenetbn.GoogLeNetBN
VGG16 = vgg16.VGG16
VGG19 = vgg19.VGG19
SqueezeNet10 = squeezenet10.SqueezeNet10
SqueezeNet11 = squeezenet11.SqueezeNet11
MINC_GoogLeNet = minc_googlenet.MINC_GoogLeNet,
MINC_VGG16 = minc_vgg16.MINC_VGG16

def getModel(arch):
    if arch == 'nin':
        return NIN()
    elif arch == 'alex':
        return Alex()
    elif arch == 'alexbn':
        return AlexBN()
    elif arch == 'vgg16':
        return VGG16()
    elif arch == 'vgg19':
        return VGG19()
    elif arch == 'googlenet':
        return GoogLeNet()
    elif arch == 'googlenetbn':
        return GoogLeNetBN()
    elif arch == 'squeezenet10':
        return SqueezeNet10()
    elif arch == 'squeezenet11':
        return SqueezeNet11()
    elif arch == 'minc-googlenet':
        return MINC_GoogLeNet()
    elif arch == 'minc-vgg16':
        return MINC_VGG16()
    else:
        raise ValueError('Invalid architecture name')
        return None
'''
