import cv2
import numpy as np
import paddle
from paddleseg.core import infer
from paddleseg.utils import visualize
import PaddleVisualize
from PIL import Image as PILImage
from paddleseg.models import deeplab
from paddleseg.models.backbones import ResNet101_vd,ResNet50_vd
import paddleseg.transforms as T
import os
from PIL import Image
from paddleseg.models import AnatomyUNet


def get_voc_palette(num_classes):
    n = num_classes
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return palette


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def generate_img_with_choice_class(img, classes: list, num_classes: int):
    # 传入 图像路径 和 需要预测的类别  总共的类别
    # img = Image.open(img)#
    img = np.asarray(img)
    f_img = img.copy()
    for idx, c in enumerate(classes):
        f_img[np.where(img == c)] = 0  # 将对应位置置零
    f_img = colorize_mask(f_img, get_voc_palette(num_classes))  # 进行染色处理
    image = f_img.convert("RGB")
    # image.save('output/process_img.png')
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
    return img

def preprocess(im_path, transforms):
    data = {}
    data['img'] = im_path
    data = transforms(data)
    data['img'] = data['img'][np.newaxis, ...]
    data['img'] = paddle.to_tensor(data['img'])
    return data


def predict(model,
            model_path,
            transforms,
            image_list,
            aug_pred=True,
            scales = 2.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=True,
            stride=(1024,512),
            crop_size=(1024,512),
            custom_color=None
            ):
    # 加载模型权重
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    # 设置模型为评估模式
    model.eval()
    # 读取图像
    im = image_list.copy()
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        data = preprocess(im, transforms)
        # 是否开启多尺度翻转预测
        if aug_pred:
            pred, _ = infer.aug_inference(
                model,
                data['img'],
                trans_info=data['trans_info'],
                scales=scales,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical,
                is_slide=is_slide,
                stride=stride,
                crop_size=crop_size)
        else:
            pred, _ = infer.inference(
                model,
                data['img'],
                trans_info=data['trans_info'],
                is_slide=is_slide,
                stride=stride,
                crop_size=crop_size)
        # 将返回数据去除多余的通道，并转为uint8类型，方便保存为图片

        pred = paddle.squeeze(pred)
        pred = pred.numpy().astype('uint8')

        # 展示结果
        # added_image = visualize_myself.visualize(image=im, result=pred, color_map=color_map, weight=0.6)
        # cv2.imshow('image_predict', added_image)

        # save pseudo color prediction
        pred_mask = PaddleVisualize.get_pseudo_color_map(pred, color_map)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return  pred_mask

def predict_img(image, weightpath):
    transforms = T.Compose([
    # T.RandomPaddingCrop(crop_size=[1024, 512]),
    # T.RandomHorizontalFlip(),
    T.Normalize()
    ])
    backbone  = ResNet101_vd()
    # model = deeplab.DeepLabV3P(num_classes = 3,
    #              backbone = backbone,
    #              aspp_ratios=(1, 6, 12, 18),
    #              aspp_out_channels=256,
    #              align_corners=False,
    #              pretrained=None,
    #              data_format="NCHW")

    model = AnatomyUNet(num_classes = 3)

    model_path = weightpath
    pred_mask = predict(model, model_path=model_path, transforms=transforms, image_list=image)

    choice_list = [0, 3]#只展示第2个类型的分割结果
    img1 = generate_img_with_choice_class(pred_mask, [0, 1], 3)
    img2 = generate_img_with_choice_class(pred_mask, [2, 3], 3)
    # img3 = generate_img_with_choice_class(pred_mask,[2, 3], 3)

    return img1,img2

def DLsegBacth(filepath, weightpath, name, savepath):
    img = cv2.imread(filepath)

    if img.shape[0] > 5000 or img.shape[1] > 5000:
        crop_size = (1024, 512)
        # 进行padding处理，确保图片大小可以完整滑动裁剪
        padded_image = np.pad(img, ((0, crop_size[1]), (0, crop_size[0]), (0, 0)), mode='constant')
        print(padded_image.shape)
        # 使用滑动窗口裁剪图片
        cropped_images = sliding_window_crop(padded_image, crop_size)

        # 对图片进行预测, 输出有两张图片，输出成两个列表
        pred_img1 = [predict_img(crop_img, weightpath)[0] for crop_img in cropped_images]
        pred_img2 = [predict_img(crop_img, weightpath)[1] for crop_img in cropped_images]

        # 将预测结果拼接起来
        img1 = concatenate_images(pred_img1, [padded_image.shape[0], padded_image.shape[1]], crop_size)
        img2 = concatenate_images(pred_img2, [padded_image.shape[0], padded_image.shape[1]], crop_size)

    elif img.shape[0] < 1000 or img.shape[1] < 1000:
        if img.shape[0] > img.shape[1]:
            img = cv2.resize(img, (int(2000 * img.shape[1] / img.shape[0]), 2000), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (2000, int(2000 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
            # 旋转90°
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img1, img2 = predict_img(img, weightpath)

    else:
        img1, img2 = predict_img(img, weightpath)
    # img1,img2 = predict_img(img,weightpath)
    imglast = cv2.add(img1, img2)
    if os.path.exists(os.path.join(savepath, 'in')) is False:
        os.makedirs(os.path.join(savepath, 'in'))
    if os.path.exists(os.path.join(savepath, 'out')) is False:
        os.makedirs(os.path.join(savepath, 'out'))
    cv2.imwrite(os.path.join(savepath, 'in', name) , img2)
    # cv2.imwrite('G:\liqiang_root\data/ceshi_result/2_' + name, img2)
    cv2.imwrite(os.path.join(savepath, 'out', name), img1)
    # cv2.imwrite('G:\liqiang_root\data/' + name, imglast)
    return imglast

def sliding_window_crop(image, window_size):
    height, width = image.shape[:2]
    h, w = window_size

    cropped_images = []
    for y in range(0, height - h + 1, h):
        for x in range(0, width - w + 1, w):
            cropped_images.append(image[y:y+h, x:x+w])
            print(image[y:y+h, x:x+w].shape)

    return cropped_images

def concatenate_images(cropped_images, original_size, crop_size):
    result_image = np.zeros((original_size[0],original_size[1]), dtype=np.uint8)
    height, width = original_size[:2]
    crop_h, crop_w = crop_size

    idx = 0
    for y in range(0, height - crop_h + 1, crop_h):
        for x in range(0, width - crop_w + 1, crop_w):
            print(1)
            print(cropped_images[idx].shape)
            result_image[ y:y+crop_h, x:x+crop_w] = cropped_images[idx]
            idx += 1

    return result_image


def seg_img(filepath, weightpath,savepath):
    filename = os.listdir(filepath)
    for name in filename:
        img = cv2.imread(os.path.join(filepath, name))
        # 如果图片太大就进行滑动窗裁剪，裁剪大小为1024*512,裁剪步长为1024*512，使用padding
        # 裁剪大小
        if img.shape[0] > 10000 or img.shape[1] > 10000:
            crop_size = (1024, 512)
            # 进行padding处理，确保图片大小可以完整滑动裁剪
            padded_image = np.pad(img, ((0, crop_size[1]), (0, crop_size[0]), (0, 0)), mode='constant')
            # 使用滑动窗口裁剪图片
            cropped_images = sliding_window_crop(padded_image, crop_size)
            # 对图片进行预测, 输出有两张图片，输出成两个列表
            pred_img1 = [predict_img(crop_img, weightpath)[0] for crop_img in cropped_images]
            pred_img2 = [predict_img(crop_img, weightpath)[1] for crop_img in cropped_images]

            # 将预测结果拼接起来
            img1 = concatenate_images(pred_img1, padded_image.shape)
            img2 = concatenate_images(pred_img2, padded_image.shape)

        elif img.shape[0] < 1000 or img.shape[1] < 1000:
            print(1)
            if img.shape[0] > img.shape[1]:
                img = cv2.resize(img, (int(2000 * img.shape[1] / img.shape[0]), 2000), interpolation=cv2.INTER_AREA)
            else:
                img = cv2.resize(img, (2000, int(2000 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
                # 旋转90°
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img1, img2 = predict_img(img, weightpath)
    
        else:
            print(2)
            img1, img2 = predict_img(img, weightpath)

        if os.path.exists(os.path.join(savepath, 'in')) is False:
            os.makedirs(os.path.join(savepath, 'in'))
        if os.path.exists(os.path.join(savepath, 'out')) is False:
            os.makedirs(os.path.join(savepath, 'out'))
        cv2.imwrite(os.path.join(savepath, 'in',name) , img2)
        # cv2.imwrite('G:\liqiang_root\data/ceshi_result/2_' + name, img2)
        cv2.imwrite(os.path.join(savepath, 'out',name), img1)
        print(name)


# filepath1 = 'D:/2022_waterlogging_section_Total/0h'
# filepath2 = 'D:/2022_waterlogging_section_Total/24h'
# filepath3 = 'D:/2022_waterlogging_section_Total/48h'
#
# weightpath ='weight/anatomyunet.pdparams'
#
# savepath1 = 'output/0h'
# savepath2 = 'output/24h'
# savepath3 = 'output/48h'
#
# seg_img(filepath1, weightpath, savepath1)
# seg_img(filepath2, weightpath, savepath2)
# seg_img(filepath3, weightpath, savepath3)