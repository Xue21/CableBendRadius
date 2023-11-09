import numpy as np
from PIL import Image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#

def resize_image(image, size):
    iw, ih  = image.size#在Python中使用Pillow库读取图像后，可以使用image.size属性获取图像的尺寸（宽度和高度）
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128)) #创建了一个新的背景图片new_image，大小为size，背景色为(128, 128, 128)的灰色。size的大小与输入的图片大小一致。
    new_image.paste(image, ((w-nw)//2, (h-nh)//2)) #使用paste()函数将调整大小后的image图片粘贴到新图片的指定位置。

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] #每个参数组都是一个字典


def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    """该函数用于展示参数配置，输入为kwargs字典，该字典存储了键值对，其中键表示参数名称，值表示参数值。"""
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception'  : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    """load_state_dict_from_url()方法从给定的URL下载PyTorch状态字典，并返回相应的状态字典。
    在这里，您传递的URL是预训练权重的URL，model_dir是本地文件夹的路径，在其中下载文件。"""
    load_state_dict_from_url(url, model_dir)