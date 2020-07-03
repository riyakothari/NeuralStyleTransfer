import os
import torch
import numpy as np
from PIL import Image
from vgg_neural_style import VGG
from torch.autograd import Variable
import torchfile
#from torch.utils.serialization import load_lua

def init_vgg16(model_folder):
    if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
            os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_folder, 'vgg16.t7'))
        # vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'))
        print(model_folder)
        vgglua = torchfile.load(os.path.join(model_folder, 'vgg16.t7'), force_8bytes_long=True)
        vgg = VGG()
        # for (src, dst) in zip(vgglua.parameters[0], vgg.parameters()):
        #     dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))

def tensor_load_rgbimage(fileName, size=None, scale=None):
    img = Image.open(fileName)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0]/scale), int(img.size[1]/scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2,0,1)
    img = torch.from_numpy(img).float()
    return img

def preprocess_batch(batch):
    batch = batch.transpose(0,1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0,1)
    return batch

def subtract_imagenet_mean_batch(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:,0,:,:] = 103.939
    mean[:,1,:,:] = 116.779
    mean[:,2,:,:] = 123.680
    return batch.sub(Variable(mean))

def gram_matrix(y):
    (b, c, h, w) = y.size()
    features = y.view(b, c, h*w)
    return features.bmm(features.transpose(1,2))/(c * h * w)

def tensor_save_rgbimage(tensor, fileName, cuda=False):
    if cuda:
        image = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        image = tensor.clone().clamp(0, 255).numpy()
    img = image.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    print(fileName)
    img.save(fileName)

def tensor_save_bgrimage(tensor, fileName, cuda=False):
    (b,g,r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r,g,b))
    tensor_save_rgbimage(tensor, fileName, cuda)
