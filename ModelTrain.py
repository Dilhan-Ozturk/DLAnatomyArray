from paddleseg.models.backbones import ResNet101_vd,ResNet50_vd, HRNet_W64,xception_deeplab
from paddleseg.models import deeplab
from paddleseg.models import segformer
from paddleseg.models import attention_unet
from paddleseg.datasets import Dataset
from paddleseg import transforms as T
import paddle
from paddleseg.models.losses import CrossEntropyLoss
from paddleseg.core import train, evaluate


def train_model(datapath, savepath, LR, ITERS, BS, SI, LI, CLA):
    backbone  = ResNet101_vd()
    # backbone2  = ResNet50_vd()
    # backbone3 = SwinTransformer_base_patch4_window12_384()
    # backbone = HRNet_W64()

    model = deeplab.DeepLabV3P(num_classes = int(CLA),
                               backbone = backbone,
                               aspp_ratios=(1, 6, 12, 18),
                               aspp_out_channels=256,
                               align_corners=False,
                               pretrained=None,
                               data_format="NCHW")

    transforms_train = [
        T.ResizeStepScaling(min_scale_factor=0.5, max_scale_factor=2.0, scale_step_size=0.25),
        T.RandomPaddingCrop(crop_size=[1024, 512]),
        T.RandomHorizontalFlip(),
        T.RandomBlur(prob=0.1),
        T.RandomScaleAspect(min_scale = 0.5, aspect_ratio = 0.33),
        # T.RandomRotation(max_rotation = 15,
        #              im_padding_value = (127.5, 127.5, 127.5),
        #              label_padding_value = 255),
        T.RandomDistort(brightness_range = 0.5,
                     brightness_prob = 0.5,
                     contrast_range = 0.5,
                     contrast_prob = 0.5,
                     saturation_range = 0.5,
                     saturation_prob = 0.5,
                     hue_range = 18,
                     hue_prob = 0.5),
        T.Normalize()
    ]

    transforms_eval = [
        # T.RandomPaddingCrop(crop_size=[1024, 512]),
        # T.RandomHorizontalFlip(),
        T.Normalize()
    ]

    train_dataset = Dataset(
        transforms= transforms_train,
        dataset_root= datapath,
        num_classes = int(CLA),
        mode='train',
        train_path= datapath + '/train.txt',
        separator=' ',
        ignore_index=255)

    eval_dataset = Dataset(
        transforms= transforms_eval,
        num_classes = int(CLA),
        dataset_root= datapath,
        val_path= datapath + '/test.txt',
        mode='val'
    )

    base_lr = float(LR)
    lr = paddle.optimizer.lr.CosineAnnealingDecay(base_lr, int(ITERS))
    # optimizer = paddle.optimizer.Momentum(lr, parameters=model2.parameters(), momentum=0.9, weight_decay=4.0e-5)
    optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)
    losses = {}
    losses['types'] = [CrossEntropyLoss()] * 1
    losses['coef'] = [1] * 1

    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset= eval_dataset,
        optimizer=optimizer,
        save_dir= savepath,
        iters=int(ITERS),
        batch_size=int(BS),
        save_interval=int(SI),
        log_iters=10,
        num_workers=0,
        losses=losses,
        use_vdl=True)
