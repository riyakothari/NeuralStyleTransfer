import torch
import torch.nn as nn
from torch import optim
import torchvision
import argparse
from torchvision import transforms
from vgg_neural_style import VGG
from PIL import Image
from collections import OrderedDict
from torch.autograd import Variable
from loss import StyleLoss, GramMatrix
from torch.hub import load_state_dict_from_url
import matplotlib
import matplotlib.pyplot as plt
# print(matplotlib.get_backend())
parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', type=str, default='images/')
parser.add_argument('--model_path', type=str, default='model/')
parser.add_argument('--style_image', type=str, default='style.jpg')
parser.add_argument('--content_image', type=str, default='content2.jpg')
parser.add_argument('--max_iter', type=int, default=500)
parser.add_argument('--show_iter', type=int, default=50)
parser.add_argument('--max_iter_hr', type=int, default=200)
opt = parser.parse_args()

image_dir = opt.dir_path
max_iter = opt.max_iter
show_iter = opt.show_iter
img_size = 512
prep = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor(), \
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), \
                            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),\
                            transforms.Lambda(lambda x: x.mul_(255))
                            ])

postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),\
                            transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1]),\
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])
                            ])

postpb = transforms.Compose([transforms.ToPILImage()])

def postp(tensor):
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img

model_url = 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
vgg = VGG()
# state_dict = load_state_dict_from_url(model_url, progress=True)
vgg.load_state_dict(torch.load(opt.model_path+'vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
# if torch.cuda.is_available():
#     vgg.cuda()

img_dirs = [image_dir]*2 #8
img_names = [opt.style_image, opt.content_image] #, 'style2.jpg', 'style3.jpg', 'style4.jpg', 'style5.jpg', 'style6.jpg', 'style7.jpg']
imgs = [Image.open(img_dirs[i]+name) for i,name in enumerate(img_names)]
imgs_torch = [prep(img) for img in imgs]
# if torch.cuda.is_available():
#     imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
# else:
imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
style_image, content_image = imgs_torch #, sty0, sty1, sty2, sty3, sty4, sty5
# sty = [sty0, sty1, sty2, sty3, sty4, sty5]
opt_image = Variable(content_image.data.clone(), requires_grad=True)

for img in imgs:
    # print("Huzzah")
    plt.imshow(img)
    plt.show()

style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
content_layers = ['r42']
loss_layers = style_layers + content_layers
loss_fns = [StyleLoss()]*len(style_layers) + [nn.MSELoss()]*len(content_layers)
# if torch.cuda.is_available():
#     loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
content_weights = [1e0]
weights = style_weights + content_weights

style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
content_targets = [A.detach() for A in vgg(content_image, content_layers)]
targets = style_targets + content_targets

###############################################
'''styWeights = [[1e3/n**2 for n in [64,128,256,512,512]]] * 6
print(styWeights, len(styWeights))
Ws = []

# Ws = [content_weights + styWeights[i] for i in range(len(styWeights))]
styTargets = []
Ts = []
Os = []
for i in range(0,6):
    styWeights = [1e3/n**2 for n in [64,128,256,512,512]]
    loss_fns = [StyleLoss()]*len(style_layers) + [nn.MSELoss()]*len(content_layers)
    content_weights = [1e0]
    Ws = content_weights + styWeights
    styTargets = [GramMatrix()(A).detach() for A in vgg(sty[i], style_layers)]
    # print(styTargets)
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    Ts = styTargets + content_targets
    Os = Variable(content_image.data.clone(), requires_grad=True)

    optimizer = optim.LBFGS([Os])
    n_iter = [0]

    while n_iter[0]<=max_iter:
        def closure():
            optimizer.zero_grad()
            out = vgg(Os, loss_layers)
            layer_losses = [Ws[a] * loss_fns[a](A, Ts[a]) for a,A in enumerate(out)]
            loss = sum(layer_losses)
            loss.backward()
            n_iter[0] += 1
            if n_iter[0]%show_iter == (show_iter-1):
                print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))
            return loss

        optimizer.step(closure)

    out_img = postp(Os.data[0].cpu().squeeze())
    plt.imshow(out_img)
    plt.show()'''
#################################################

optimizer = optim.LBFGS([opt_image])
n_iter = [0]

while n_iter[0]<=max_iter:
    def closure():
        optimizer.zero_grad()
        out = vgg(opt_image, loss_layers)
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter[0] += 1
        if n_iter[0]%show_iter == (show_iter-1):
            print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))
        return loss

    optimizer.step(closure)

out_img = postp(opt_image.data[0].cpu().squeeze())
plt.imshow(out_img)
plt.show()
# plt.savefig(out_img)
# gcf().set_size_inches(10,10)

image_size_hr = 800
prep_hr = transforms.Compose([transforms.Scale(image_size_hr), \
                            transforms.ToTensor(),\
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),\
                            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]), \
                            transforms.Lambda(lambda x: x.mul_(255)), \
                            ])

imgs_torch = [prep_hr(img) for img in imgs]
imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
style_image, content_image = imgs_torch
opt_img = prep_hr(out_img).unsqueeze(0)
opt_img = Variable(opt_img.type_as(content_image.data), requires_grad = True)

style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
content_targets = [A.detach() for A in vgg(content_image, content_layers)]
targets = style_targets + content_targets

max_iter_hr = opt.max_iter_hr
optimizer = optim.LBFGS([opt_img])
n_iter = [0]

while n_iter[0]<=max_iter_hr:
    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter[0] += 1
        if n_iter[0]%show_iter == (show_iter-1):
            print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))
        return loss

    optimizer.step(closure)

out_img_hr = postp(opt_img.data[0].cpu().squeeze())
plt.imshow(out_img_hr)
plt.show()
