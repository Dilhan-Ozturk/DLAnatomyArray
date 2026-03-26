import cv2
import numpy as np
import paddle
from paddleseg.core import infer
from paddleseg.utils import visualize
import PaddleVisualize
from PIL import Image
from paddleseg.models import deeplab
from paddleseg.models.backbones import ResNet50_vd
import paddleseg.transforms as T
import os
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
        while lab > 0:
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
    img = np.asarray(img)
    f_img = img.copy()
    for idx, c in enumerate(classes):
        f_img[np.where(img == c)] = 0
    f_img = colorize_mask(f_img, get_voc_palette(num_classes))
    image = f_img.convert("RGB")
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
            scales=2.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=True,
            stride=(1024, 512),
            crop_size=(1024, 512),
            custom_color=None
            ):
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    model.eval()
    im = image_list.copy()
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        data = preprocess(im, transforms)
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

        pred = paddle.squeeze(pred)
        pred = pred.numpy().astype('uint8')

        pred_mask = PaddleVisualize.get_pseudo_color_map(pred, color_map)
        return pred_mask


def predict_img(image, weightpath):
    transforms = T.Compose([
        T.Normalize()
    ])
    model = AnatomyUNet(num_classes=3)
    model_path = weightpath
    pred_mask = predict(model, model_path=model_path, transforms=transforms, image_list=image)

    img1 = generate_img_with_choice_class(pred_mask, [0, 1], 3)
    img2 = generate_img_with_choice_class(pred_mask, [2, 3], 3)

    return img1, img2


def DLsegBatch(filepath, weightpath, name, savepath):
    """Run DL segmentation on a single image and save the results."""
    img = cv2.imread(filepath)

    if img.shape[0] > 5000 or img.shape[1] > 5000:
        crop_size = (1024, 512)
        padded_image = np.pad(img, ((0, crop_size[1]), (0, crop_size[0]), (0, 0)), mode='constant')
        cropped_images = sliding_window_crop(padded_image, crop_size)

        pred_img1 = [predict_img(crop_img, weightpath)[0] for crop_img in cropped_images]
        pred_img2 = [predict_img(crop_img, weightpath)[1] for crop_img in cropped_images]

        img1 = concatenate_images(pred_img1, [padded_image.shape[0], padded_image.shape[1]], crop_size)
        img2 = concatenate_images(pred_img2, [padded_image.shape[0], padded_image.shape[1]], crop_size)

    elif img.shape[0] < 1000 or img.shape[1] < 1000:
        if img.shape[0] > img.shape[1]:
            img = cv2.resize(img, (int(2000 * img.shape[1] / img.shape[0]), 2000), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (2000, int(2000 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img1, img2 = predict_img(img, weightpath)

    else:
        img1, img2 = predict_img(img, weightpath)

    imglast = cv2.add(img1, img2)

    if not os.path.exists(os.path.join(savepath, 'in')):
        os.makedirs(os.path.join(savepath, 'in'))
    if not os.path.exists(os.path.join(savepath, 'out')):
        os.makedirs(os.path.join(savepath, 'out'))

    cv2.imwrite(os.path.join(savepath, 'in', name), img2)
    cv2.imwrite(os.path.join(savepath, 'out', name), img1)
    return imglast


def sliding_window_crop(image, window_size):
    height, width = image.shape[:2]
    h, w = window_size

    cropped_images = []
    for y in range(0, height - h + 1, h):
        for x in range(0, width - w + 1, w):
            cropped_images.append(image[y:y + h, x:x + w])

    return cropped_images


def concatenate_images(cropped_images, original_size, crop_size):
    result_image = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)
    height, width = original_size[:2]
    crop_h, crop_w = crop_size

    idx = 0
    for y in range(0, height - crop_h + 1, crop_h):
        for x in range(0, width - crop_w + 1, crop_w):
            result_image[y:y + crop_h, x:x + crop_w] = cropped_images[idx]
            idx += 1

    return result_image


def seg_img(filepath, weightpath, savepath):
    """Run DL segmentation on all images in a directory."""
    filename = os.listdir(filepath)
    for name in filename:
        img = cv2.imread(os.path.join(filepath, name))

        if img.shape[0] > 10000 or img.shape[1] > 10000:
            crop_size = (1024, 512)
            padded_image = np.pad(img, ((0, crop_size[1]), (0, crop_size[0]), (0, 0)), mode='constant')
            cropped_images = sliding_window_crop(padded_image, crop_size)
            pred_img1 = [predict_img(crop_img, weightpath)[0] for crop_img in cropped_images]
            pred_img2 = [predict_img(crop_img, weightpath)[1] for crop_img in cropped_images]
            img1 = concatenate_images(pred_img1, [padded_image.shape[0], padded_image.shape[1]], crop_size)
            img2 = concatenate_images(pred_img2, [padded_image.shape[0], padded_image.shape[1]], crop_size)

        elif img.shape[0] < 1000 or img.shape[1] < 1000:
            if img.shape[0] > img.shape[1]:
                img = cv2.resize(img, (int(2000 * img.shape[1] / img.shape[0]), 2000), interpolation=cv2.INTER_AREA)
            else:
                img = cv2.resize(img, (2000, int(2000 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img1, img2 = predict_img(img, weightpath)

        else:
            img1, img2 = predict_img(img, weightpath)

        if not os.path.exists(os.path.join(savepath, 'in')):
            os.makedirs(os.path.join(savepath, 'in'))
        if not os.path.exists(os.path.join(savepath, 'out')):
            os.makedirs(os.path.join(savepath, 'out'))

        cv2.imwrite(os.path.join(savepath, 'in', name), img2)
        cv2.imwrite(os.path.join(savepath, 'out', name), img1)
        print(f'{name} has been successfully segmented')