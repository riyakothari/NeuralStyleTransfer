#python real_time_style_transfer.py train --dataset C:/Users/riyak/Downloads/Pytorch/MLP/NeuralStyleTransfer/NeuralStyleTransfer/train2014/train --style_image C:/Users/riyak/Downloads/Pytorch/MLP/NeuralStyleTransfer/NeuralStyleTransfer/images/style8.jpg --vgg_model_dir C:/Users/riyak/Downloads/Pytorch/MLP/NeuralStyleTransfer/NeuralStyleTransfer/vgg_model --save_model_dir C:/Users/riyak/Downloads/Pytorch/MLP/NeuralStyleTransfer/NeuralStyleTransfer/saved_model --cuda 0 --epochs 30 --log_interval 10

#python real_time_style_transfer.py eval --content_image C:/Users/riyak/Downloads/Pytorch/MLP/NeuralStyleTransfer/NeuralStyleTransfer/images/content2.jpg --model C:/Users/riyak/Downloads/Pytorch/MLP/NeuralStyleTransfer/NeuralStyleTransfer/saved_model/epoch_2_Wed_Jun_24_15_42_56_2020_1.0_5.0.model --output_image C:/Users/riyak/Downloads/Pytorch/MLP/NeuralStyleTransfer/NeuralStyleTransfer/images/real_time.jpg --cuda 0
#python real_time_style_transfer.py eval --content_image images/content2.jpg --model epoch_12_Tue_Jun_30_18_51_18_2020_1.0_5.0.model --output_image images/real_time.jpg --cuda 0
import os
import time
import argparse

import utils
import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from imageTransformationNetwork import ImageTransformationNetwork
from vgg_neural_style import VGG
from torchvision import transforms

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers':0, 'pin_memory':False}
    else:
        kwargs = {}

    transform = transforms.Compose([transforms.Scale(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])

    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    transformerNetwork = ImageTransformationNetwork()
    optimizer = Adam(transformerNetwork.parameters(), args.learning_rate)
    mse_loss = nn.MSELoss()

    vgg = VGG()
    # model_url = 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
    #vgg = VGG()
    # state_dict = load_state_dict_from_url(model_url, progress=True)
    # vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, 'vgg_conv.pth')))
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, 'vgg16.weight')))

    if args.cuda:
        transformerNetwork.cuda()
        vgg.cuda()

    style = utils.tensor_load_rgbimage(args.style_image, size = args.image_size)
    style = style.repeat(args.batch_size, 1,1,1)
    style = utils.preprocess_batch(style)
    if args.cuda:
        style.cuda()

    style_v = Variable(style, volatile=True)
    style_v = utils.subtract_imagenet_mean_batch(style_v)
    layers = ['r12', 'r22', 'r33', 'r43']#, 'r51']
    features_style = vgg(style_v, layers)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformerNetwork.train()
        agg_content_loss = 0
        agg_style_loss = 0
        count = 0

        for batch_id, (x, _) in enumerate(train_loader):
            count += len(x)
            optimizer.zero_grad()
            x = Variable(utils.preprocess_batch(x))
            if args.cuda:
                x.cuda()
            y = transformerNetwork(x)
            xc = Variable(x.data.clone(), volatile=True)
            y = utils.subtract_imagenet_mean_batch(y)
            xc = utils.subtract_imagenet_mean_batch(xc)

            features_y = vgg(y, layers)
            features_xc = vgg(xc, layers)

            f_xc_c = Variable(features_xc[1].data, requires_grad=False)
            content_loss = args.content_weight* mse_loss(features_y[1], f_xc_c)
            style_loss = 0

            for m in range(len(features_y)):
                gram_s = Variable(gram_style[m].data, requires_grad=False)
                gram_y = utils.gram_matrix(features_y[m])
                style_loss += args.style_weight*mse_loss(gram_y, gram_s[:len(x), :, :])

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()
            print("content loss ",content_loss.item())
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                msg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(msg)
        save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + "_" + str(args.content_weight) + "_" + str(args.style_weight) + ".model"
        save_model_path = os.path.join(args.save_model_dir, save_model_filename)
        torch.save(transformerNetwork.state_dict(), save_model_path)

    transformerNetwork.eval()
    transformerNetwork.cpu()
    save_model_filename = "epoch__" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformerNetwork.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)



def stylize(args):
    content_image = utils.tensor_load_rgbimage(args.content_image, scale = args.content_scale)
    content_image = content_image.unsqueeze(0)
    if args.cuda:
        content_image.cuda()
    content_image = Variable(utils.preprocess_batch(content_image), volatile=True)
    style_model = ImageTransformationNetwork()
    style_model.load_state_dict(torch.load(args.model))

    if args.cuda:
        style_model.cuda()

    output = style_model(content_image)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)

def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def main():
    main_parser = argparse.ArgumentParser(description='parser')
    subparsers = main_parser.add_subparsers(title='subcommands', dest='subcommand')
    train_arg_parser = subparsers.add_parser('train')
    eval_arg_parser = subparsers.add_parser('eval')

    train_arg_parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    train_arg_parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    train_arg_parser.add_argument('--cuda', type=int, required=True, help='1 for GPU, 0 for CPU')
    train_arg_parser.add_argument('--seed', type=int, default=42, help='sets the random seeds for training')
    train_arg_parser.add_argument('--image_size', type=int, default=256, help='set the training image size, 256 x 256')
    train_arg_parser.add_argument('--dataset', type=str, required=True, help='path to the folder containing training images')
    train_arg_parser.add_argument('--learning_rate', type=float, default=1e-03, help='set the learning rate for the optimizer')
    train_arg_parser.add_argument('--vgg_model_dir', type=str, required=True, help='if vgg model not present, download it')
    train_arg_parser.add_argument('--style_image', type=str, default='images/style/style1.jpg', help='style image file path')
    train_arg_parser.add_argument('--content_weight', type=float, default=1.0, help='weight for the content-loss')
    train_arg_parser.add_argument('--style_weight', type=float, default=5.0, help='weight for the style-loss')
    train_arg_parser.add_argument('--log_interval', type=int, default=100, help='logs the training loss after these many iterations')
    train_arg_parser.add_argument('--save_model_dir', type=str, required=True, help='path to save the trained model')

    eval_arg_parser.add_argument('--content_image', type=str, required=True, help='content image to be stylized')
    eval_arg_parser.add_argument('--content_scale', type=float, default=None, help='scale down the content image by this factor')
    eval_arg_parser.add_argument('--output_image', type=str, required=True, help='path for output to be stored')
    eval_arg_parser.add_argument('--model', type=str, required=True, help='saved model for stylizing the image')
    eval_arg_parser.add_argument('--cuda', type=int, required=True, help='1 for GPU, 0 for CPU')

    args = main_parser.parse_args()
    print(args)

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)



if __name__ == '__main__':
    main()
