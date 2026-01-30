# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import deepcopy
from pathlib import Path

from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules.block import *
from ultralytics.nn.modules.conv import DSConv
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
import timm
import torch
import torch.nn as nn

from ultralytics.nn.modules import *
from ultralytics.nn.extra_modules import *
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights, intersect_dicts,
                                           make_divisible, model_info, scale_img, time_sync, smart_inference_mode)

from ultralytics.nn.backbone.convnextv2 import *
from ultralytics.nn.backbone.fasternet import *
from ultralytics.nn.backbone.efficientViT import *
from ultralytics.nn.backbone.EfficientFormerV2 import *
from ultralytics.nn.backbone.VanillaNet import *
# from ultralytics.nn.backbone.revcol import *
from ultralytics.nn.backbone.lsknet import *
from ultralytics.nn.backbone.SwinTransformer import *
from ultralytics.nn.backbone.repvit import *
from ultralytics.nn.backbone.CSwimTramsformer import *
from ultralytics.nn.backbone.UniRepLKNet import *
from ultralytics.nn.backbone.TransNext import *
from ultralytics.nn.backbone.rmt import *
from ultralytics.nn.backbone.pkinet import *
from ultralytics.nn.backbone.mobilenetv4 import *
from ultralytics.nn.backbone.starnet import *
import torch.nn.functional as F  # 导入functional模块

try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale. Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict_or(self, x, profile=False, visualize=False, augment=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize)
    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            augment (bool): Augment image during prediction.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)
    # ===================================================================================================================
    def _predict_once_or(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x
    # ====================================================================================================================================================================================
    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(f'WARNING ⚠️ {self.__class__.__name__} does not support augmented inference yet. '
                       f'Reverting to single-scale inference instead.')
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        if type(x) is tuple:
            x = list(x)
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        if type(x) is list:
            bs = x[0].size(0)
        else:
            bs = x.size(0)
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1E9 * 2 / bs if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, 'bn'):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvNormLayer):
                    m.conv = fuse_conv_and_bn(m.conv, m.norm)  # update conv
                    delattr(m, 'norm')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if hasattr(m, 'switch_to_deploy'):
                    m.switch_to_deploy()
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        preds = self.forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError('compute_loss() needs to be implemented by task heads')


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            try:
                self.forward(torch.zeros(2, ch, 640, 640))
            except (RuntimeError, ValueError) as e:
                if 'Not implemented on the CPU' in str(e) or 'Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)' in str(e) or \
                'CUDA tensor' in str(e) or 'is_cuda()' in str(e) or 'carafe_forward_impl' in str(e) or 'Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)' in str(e):
                    self.model.to(torch.device('cuda'))
            except Exception:
                pass
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self)


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg='yolov8n-seg.yaml', ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg='yolov8n-pose.yaml', ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg['kpt_shape']):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg['kpt_shape'] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self, cfg='yolov8n-cls.yaml', ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override YAML value
        elif not nc and not self.yaml.get('nc', None):
            raise ValueError('nc not specified. Must specify nc in model.yaml or function arguments.')
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = types.index(nn.Linear)  # nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = types.index(nn.Conv2d)  # nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        cfg (str): The configuration file path or preset string. Default is 'rtdetr-l.yaml'.
        ch (int): Number of input channels. Default is 3 (RGB).
        nc (int, optional): Number of classes for object detection. Default is None.
        verbose (bool): Specifies if summary statistics are shown during initialization. Default is True.

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a forward pass through the network and returns the output.
    """

    def __init__(self, cfg='rtdetr-l.yaml', ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True, use_sl=False, use_emasl=False, use_svfl=False, use_emasvfl=False)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        img = batch['img']
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch['batch_idx']
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            'cls': batch['cls'].to(img.device, dtype=torch.long).view(-1),
            'bboxes': batch['bboxes'].to(device=img.device),
            'batch_idx': batch_idx.to(img.device, dtype=torch.long).view(-1),
            'gt_groups': gt_groups}

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta['dn_num_split'], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion((dec_bboxes, dec_scores),
                              targets,
                              dn_bboxes=dn_bboxes,
                              dn_scores=dn_scores,
                              dn_meta=dn_meta)
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor([loss[k].detach() for k in ['loss_giou', 'loss_class', 'loss_bbox']],
                                                   device=img.device)

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt = [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if hasattr(m, 'backbone'):
                x = m(x)
                for _ in range(5 - len(x)):
                    x.insert(0, None)
                for i_idx, i in enumerate(x):
                    if i_idx in self.save:
                        y.append(i)
                    else:
                        y.append(None)
                # for i in x:
                #     if i is not None:
                #         print(i.size())
                x = x[-1]
            else:
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.

    Example:
        ```python
        with temporary_modules({'old.module.path': 'new.module.path'}):
            import old.module.path  # this will now import new.module.path
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if not modules:
        modules = {}

    import importlib
    import sys
    try:
        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = importlib.import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    it catches the error, logs a warning message, and attempts to install the missing module via the
    check_requirements() function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        (dict): The loaded PyTorch model.
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix='.pt')
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules({
                'ultralytics.yolo.utils': 'ultralytics.utils',
                'ultralytics.yolo.v8': 'ultralytics.models.yolo',
                'ultralytics.yolo.data': 'ultralytics.data'}):  # for legacy 8.0 Classify and Pose models
            return torch.load(file, map_location='cpu'), file  # load

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == 'models':
            raise TypeError(
                emojis(f'ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained '
                       f'with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with '
                       f'YOLOv8 at https://github.com/ultralytics/ultralytics.'
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")) from e
        LOGGER.warning(f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
                       f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")
        check_requirements(e.name)  # install missing module

        return torch.load(file, map_location='cpu'), file  # load


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""

    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt['train_args']} if 'train_args' in ckpt else None  # combined args
        model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
            m.inplace = inplace
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[torch.argmax(torch.tensor([m.stride.max() for m in ensemble])).int()].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f'Models differ in class counts {[m.nc for m in ensemble]}'
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get('train_args', {}))}  # combine model and default args, preferring model args
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])

    model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
            m.inplace = inplace
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True, warehouse_manager=None):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    is_backbone = False
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        try:
            if m == 'node_mode':
                m = d[m]
                if len(args) > 0:
                    if args[0] == 'head_channel':
                        args[0] = int(d[args[0]])
            t = m
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        except:
            pass
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    try:
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                    except:
                        args[j] = a

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, DSConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.Conv2d, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3,
                 ConvNormLayer, DWRC3, C3_DWR, C2f_DWR, C3_DCNv2_Dynamic, C2f_DCNv2_Dynamic, BasicBlock_DCNv2_Dynamic, BottleNeck_DCNv2_Dynamic,
                 C3_DCNv2, C2f_DCNv2, BasicBlock_DCNv2, BottleNeck_DCNv2, C3_DCNv3, C2f_DCNv3, BasicBlock_DCNv3, BottleNeck_DCNv3,
                 C3_iRMB, C2f_iRMB, C3_iRMB_Cascaded, C2f_iRMB_Cascaded, C3_Attention, C2f_Attention, C3_Ortho, C2f_Ortho,
                 C3_DySnakeConv, C2f_DySnakeConv, DySnakeConv,A2C2f,
                 C3_Faster, C2f_Faster, C3_Faster_EMA, C2f_Faster_EMA, C3_Faster_Rep, C2f_Faster_Rep, C3_Faster_Rep_EMA, C2f_Faster_Rep_EMA,
                 AKConv, C3_AKConv, C2f_AKConv, C3_RFAConv, C2f_RFAConv, C3_RFCAConv, C2f_RFCAConv, C3_RFCBAMConv, C2f_RFCBAMConv,
                 RFAConv, RFCAConv, RFCBAMConv, C3_Conv3XC, C2f_Conv3XC, C3_SPAB, C2f_SPAB, Conv3XCC3, DRBC3, DBBC3,
                 C3_UniRepLKNetBlock, C2f_UniRepLKNetBlock, C3_DRB, C2f_DRB, C3_DWR_DRB, C2f_DWR_DRB, DWRC3_DRB,C3k2,C2PSA,
                 C2f_DBB, C3_DBB, CSP_EDLAN, GSConv, VoVGSCSP, VoVGSCSPC,
                 C3_AggregatedAtt, C2f_AggregatedAtt, SPDConv,
                 C3_DCNv4, C2f_DCNv4, BasicBlock_DCNv4, BottleNeck_DCNv4, HWD,
                 C3_SWC, C2f_SWC, C3_iRMB_DRB, C2f_iRMB_DRB, C3_iRMB_SWC, C2f_iRMB_SWC,
                 C3_VSS, C2f_VSS, C3_LVMB, C2f_LVMB, RepNCSPELAN4, DBBNCSPELAN4, OREPANCSPELAN4, DRBNCSPELAN4, Conv3XCNCSPELAN4, ADown,
                 C3_ContextGuided, C2f_ContextGuided, CSP_PAC, DGCST, DGCST2, RetBlockC3, C3_RetBlock, C2f_RetBlock, RepNCSPELAN4_CAA,
                 C3_PKIModule, C2f_PKIModule, C3_FADC, C2f_FADC, C3_PPA, C2f_PPA, SRFD, DRFD, RGCSPELAN, C3_Faster_CGLU, C2f_Faster_CGLU,
                 C3_Star, C2f_Star, C3_Star_CAA, C2f_Star_CAA, C3_KAN, C2f_KAN, KANC3, C3_DEConv, C2f_DEConv, C3_SMPCGLU, C2f_SMPCGLU,
                 C3_Heat, C2f_Heat, CSP_PTB, SimpleStem, VisionClueMerge, VSSBlock_YOLO, XSSBlock, GLSA, WTConv2d, C2f_FMB, gConvC3, C2f_gConv,
                 LDConv, C2f_AdditiveBlock, C2f_AdditiveBlock_CGLU, CSP_MSCB, C2f_MSMHSA_CGLU, CSP_PMSFA, C2f_MogaBlock,
                 C2f_SHSA, C2f_SHSA_CGLU, C2f_SMAFB, C2f_SMAFB_CGLU, CSP_MutilScaleEdgeInformationEnhance, C2f_FFCM, C2f_SFHF, CSP_FreqSpatial,
                 C2f_MSM, CSP_MutilScaleEdgeInformationSelect, C2f_HDRAB, C2f_RAB, LFEC3, C2f_FCA, C2f_CAMixer, MANet, MANet_FasterBlock, MANet_FasterCGLU,
                 MANet_Star, C2f_HFERB, C2f_DTAB, C2f_JDPM, C2f_ETB, C2f_FDT, PSConv, C2f_AP, C2f_ELGCA, C2f_ELGCA_CGLU, C2f_Strip, C2f_StripCGLU,
                 C2f_KAT, C2f_Faster_KAN, C2f_DCMB, C2f_DCMB_KAN, C2f_GlobalFilter, C2f_DynamicFilter, RepHMS):
            if args[0] == 'head_channel':
                args[0] = d[args[0]]
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (DySnakeConv,):
                c2 = c2 * 3
            if m in (RepNCSPELAN4, DBBNCSPELAN4, OREPANCSPELAN4, DRBNCSPELAN4, Conv3XCNCSPELAN4, RepNCSPELAN4_CAA):
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                args[3] = make_divisible(min(args[3], max_channels) * width, 8)
            
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3, DWRC3, C3_DWR, C2f_DWR, C3_DCNv2_Dynamic, C2f_DCNv2_Dynamic,
                     C3_DCNv2, C2f_DCNv2, C3_DCNv3, C2f_DCNv3, C3_iRMB, C2f_iRMB, C3_iRMB_Cascaded, C2f_iRMB_Cascaded, 
                     C3_Attention, C2f_Attention, C3_Ortho, C2f_Ortho, C3_DySnakeConv, C2f_DySnakeConv,
                     C3_Faster, C2f_Faster, C3_Faster_EMA, C2f_Faster_EMA, C3_Faster_Rep, C2f_Faster_Rep, C3_Faster_Rep_EMA, C2f_Faster_Rep_EMA,
                     C3_AKConv, C2f_AKConv, C3_RFAConv, C2f_RFAConv, C3_RFCAConv, C2f_RFCAConv, C3_RFCBAMConv, C2f_RFCBAMConv,
                     C3_Conv3XC, C2f_Conv3XC, C3_SPAB, C2f_SPAB, C3_UniRepLKNetBlock, C2f_UniRepLKNetBlock, C3_DRB, C2f_DRB, C3_DWR_DRB, C2f_DWR_DRB, DWRC3_DRB,
                     Conv3XCC3, DRBC3, DBBC3, C2f_DBB, C3_DBB, CSP_EDLAN, VoVGSCSP, VoVGSCSPC,
                     C3_AggregatedAtt, C2f_AggregatedAtt, C3_DCNv4, C2f_DCNv4, C3_SWC, C2f_SWC, C3_iRMB_DRB, C2f_iRMB_DRB, C3_iRMB_SWC, C2f_iRMB_SWC,
                     C3_VSS, C2f_VSS, C3_LVMB, C2f_LVMB, C3_ContextGuided, C2f_ContextGuided, RetBlockC3, C3_RetBlock, C2f_RetBlock,
                     C3_PKIModule, C2f_PKIModule, C3_FADC, C2f_FADC, C3_PPA, C2f_PPA, RGCSPELAN, C3_Faster_CGLU, C2f_Faster_CGLU,
                     C3_Star, C2f_Star, C3_Star_CAA, C2f_Star_CAA, C3_KAN, C2f_KAN, KANC3, C3_DEConv, C2f_DEConv, C3_SMPCGLU, C2f_SMPCGLU, 
                     C3_Heat, C2f_Heat, CSP_PTB, XSSBlock, C2f_FMB, C2f_gConv, gConvC3, C2f_AdditiveBlock, C2f_AdditiveBlock_CGLU, CSP_MSCB,
                     C2f_MSMHSA_CGLU, CSP_PMSFA, C2f_MogaBlock, C2f_SHSA, C2f_SHSA_CGLU, C2f_SMAFB, C2f_SMAFB_CGLU, CSP_MutilScaleEdgeInformationEnhance,
                     C2f_FFCM, C2f_SFHF, CSP_FreqSpatial, C2f_MSM, CSP_MutilScaleEdgeInformationSelect, C2f_HDRAB, C2f_RAB, LFEC3, C2f_FCA, C2f_CAMixer, MANet,
                     MANet_FasterBlock, MANet_FasterCGLU, MANet_Star, C2f_HFERB, C2f_DTAB, C2f_JDPM, C2f_ETB, C2f_FDT, C2f_AP, C2f_ELGCA, C2f_ELGCA_CGLU, 
                     C2f_Strip, C2f_StripCGLU, C2f_KAT, C2f_Faster_KAN, C2f_DCMB, C2f_DCMB_KAN, C2f_GlobalFilter, C2f_DynamicFilter):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m in (AIFI, AIFI_LPE, TransformerEncoderLayer_LocalWindowAttention, TransformerEncoderLayer_DAttention, TransformerEncoderLayer_HiLo, 
                   TransformerEncoderLayer_EfficientAdditiveAttnetion, AIFI_RepBN, TransformerEncoderLayer_AdditiveTokenMixer,
                   TransformerEncoderLayer_MSMHSA, TransformerEncoderLayer_DHSA, TransformerEncoderLayer_DPB, DTAB, ETB, FDT,
                   TransformerEncoderLayer_Pola, TransformerEncoderLayer_TSSA):
            c2 = ch[f]
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock, Ghost_HGBlock, Rep_HGBlock, HGBlock_Attention):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m in (HGBlock, Ghost_HGBlock, Rep_HGBlock, HGBlock_Attention):
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m in {Concat}:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment, Pose):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is Fusion:
            args[0] = d[args[0]]
            c1, c2 = [ch[x] for x in f], (sum([ch[x] for x in f]) if args[0] == 'concat' else ch[f[0]])
            args = [c1, args[0]]
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif isinstance(m, str):
            t = m
            if len(args) == 2:        
                m = timm.create_model(m, pretrained=args[0], pretrained_cfg_overlay={'file':args[1]}, features_only=True)
            elif len(args) == 1:
                m = timm.create_model(m, pretrained=args[0], features_only=True)
            c2 = m.feature_info.channels()
        elif m in {convnextv2_atto, convnextv2_femto, convnextv2_pico, convnextv2_nano, convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge,
                   fasternet_t0, fasternet_t1, fasternet_t2, fasternet_s, fasternet_m, fasternet_l,
                   EfficientViT_M0, EfficientViT_M1, EfficientViT_M2, EfficientViT_M3, EfficientViT_M4, EfficientViT_M5,
                   efficientformerv2_s0, efficientformerv2_s1, efficientformerv2_s2, efficientformerv2_l,
                   vanillanet_5, vanillanet_6, vanillanet_7, vanillanet_8, vanillanet_9, vanillanet_10, vanillanet_11, vanillanet_12, vanillanet_13, vanillanet_13_x1_5, vanillanet_13_x1_5_ada_pool,
                #    RevCol,
                   lsknet_t, lsknet_s,
                   SwinTransformer_Tiny,
                   repvit_m0_9, repvit_m1_0, repvit_m1_1, repvit_m1_5, repvit_m2_3,
                   CSWin_tiny, CSWin_small, CSWin_base, CSWin_large,
                   unireplknet_a, unireplknet_f, unireplknet_p, unireplknet_n, unireplknet_t, unireplknet_s, unireplknet_b, unireplknet_l, unireplknet_xl,
                   transnext_micro, transnext_tiny, transnext_small, transnext_base,
                   RMT_T, RMT_S, RMT_B, RMT_L,
                   PKINET_T, PKINET_S, PKINET_B,
                   MobileNetV4ConvSmall, MobileNetV4ConvMedium, MobileNetV4ConvLarge, MobileNetV4HybridMedium, MobileNetV4HybridLarge,
                   starnet_s050, starnet_s100, starnet_s150, starnet_s1, starnet_s2, starnet_s3, starnet_s4
                   }:
            # if m is RevCol:
            #     args[1] = [make_divisible(min(k, max_channels) * width, 8) for k in args[1]]
            #     args[2] = [max(round(k * depth), 1) for k in args[2]]
            m = m(*args)
            c2 = m.channel
        elif m in {EMA, SpatialAttention, BiLevelRoutingAttention, BiLevelRoutingAttention_nchw,
                   TripletAttention, CoordAtt, CBAM, BAMBlock, LSKBlock, SEAttention, CPCA, EfficientAttention, 
                   MPCA, deformable_LKA, EffectiveSEModule, LSKA, SegNext_Attention, DAttention, MLCA,
                   FocusedLinearAttention, TransNeXt_AggregatedAttention, HiLo, ChannelAttention_HSFPN, ELA_HSFPN, CA_HSFPN, CAA_HSFPN,
                   DySample, CARAFE, ELA, CAA, CAFM, LocalWindowAttention, EfficientAdditiveAttnetion, AFGCAttention, EUCB, ContrastDrivenFeatureAggregation,
                   FSA
                #    ScConv, LAWDS, EMSConv, EMSConvP, Partial_conv3, FocalModulation
                   }:
            c2 = ch[f]
            args = [c2, *args]
            # print(args)
        elif m in {SimAM, SpatialGroupEnhance}:
            c2 = ch[f]
        elif m is ContextGuidedBlock_Down:
            c2 = ch[f] * 2
            args = [ch[f], c2, *args]
        # elif m is BiFusion:
        #     c1 = [ch[x] for x in f]
        #     c2 = make_divisible(min(args[0], max_channels) * width, 8)
        #     args = [c1, c2]
        # --------------GOLD-YOLO--------------
        elif m in {SimFusion_4in, AdvPoolFusion}:
            c2 = sum(ch[x] for x in f)
        elif m is SimFusion_3in:
            c2 = args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [[ch[f_] for f_ in f], c2]
        elif m is IFM:
            c1 = ch[f]
            c2 = sum(args[0])
            args = [c1, *args]
        elif m is InjectionMultiSum_Auto_pool:
            c1 = ch[f[0]]
            c2 = args[0]
            args = [c1, *args]
        elif m is PyramidPoolAgg:
            c2 = args[0]
            args = [sum([ch[f_] for f_ in f]), *args]
        elif m is TopBasicLayer:
            c2 = sum(args[1])
        # --------------GOLD-YOLO--------------
        # --------------ASF--------------
        elif m is Zoom_cat:
            c2 = sum(ch[x] for x in f)
        elif m is Add:
            c2 = ch[f[-1]]
        elif m in {ScalSeq, DynamicScalSeq}:
            c1 = [ch[x] for x in f]
            c2 = make_divisible(args[0] * width, 8)
            args = [c1, c2]
        elif m is asf_attention_model:
            args = [ch[f[-1]]]
        # --------------ASF--------------
        elif m is SDI:
            args = [[ch[x] for x in f]]
        elif m is Multiply:
            c2 = ch[f[0]]
        elif m in {AttentionUpsample, AttentionDownsample}:
            c2 = ch[f]
            args = [c2]
        elif m is FocusFeature:
            c1 = [ch[x] for x in f]
            c2 = int(c1[1] * 0.5 * 3)
            args = [c1, *args]
        elif m is DASI:
            c1 = [ch[x] for x in f]
            args = [c1, c2]
        elif m is CFC_CRB:
            c1 = ch[f]
            c2 = c1 // 2
            args = [c1, *args]
        elif m is SFC_G2:
            c1 = [ch[x] for x in f]
            c2 = c1[0]
            args = [c1]
        elif m in {CGAFusion, CAFMFusion, SDFM, PSFM}:
            c2 = ch[f[1]]
            args = [c2, *args]
        elif m in {ContextGuideFusionModule}:
            c1 = [ch[x] for x in f]
            c2 = 2 * c1[1]
            args = [c1]
        elif m in {PSA}:
            c2 = ch[f]
            args = [c2, *args]
        elif m in {SBA}:
            c1 = [ch[x] for x in f]
            c2 = c1[-1]
            args = [c1, c2]
        elif m in {WaveletPool}:
            c2 = ch[f] * 4
        elif m in {WaveletUnPool}:
            c2 = ch[f] // 4
        elif m in {CSPOmniKernel}:
            c2 = ch[f]
            args = [c2]
        elif m in {ChannelTransformer, PyramidContextExtraction}:
            c1 = [ch[x] for x in f]
            c2 = c1
            args = [c1]
        elif m in {GetIndexOutput}:
            c2 = ch[f][args[0]]
        elif m in {RCM}:
            c2 = ch[f]
            args = [c2, *args]
        elif m in {DynamicInterpolationFusion}:
            c2 = ch[f[0]]
            args = [[ch[x] for x in f]]
        elif m in {FuseBlockMulti}:
            c2 = ch[f[0]]
            args = [c2]
        elif m in {CrossLayerChannelAttention, CrossLayerSpatialAttention}:
            c2 = [ch[x] for x in f]
            args = [c2[0], *args]
        elif m in {FreqFusion}:
            c2 = ch[f[0]]
            args = [[ch[x] for x in f], *args]
        elif m in {DynamicAlignFusion, ConvEdgeFusion}:
            c2 = args[0]
            args = [[ch[x] for x in f], c2]
        elif m in {MutilScaleEdgeInfoGenetator}:
            c1 = ch[f]
            c2 = [make_divisible(min(i, max_channels) * width, 8) for i in args[0]]
            args = [c1, c2]
        elif m is HyperComputeModule:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m in {MultiScaleGatedAttn}:
            c1 = [ch[x] for x in f]
            c2 = min(c1)
            args = [c1]
        elif m in {WFU, MultiScalePCA, MultiScalePCA_Down}:
            c1 = [ch[x] for x in f]
            c2 = c1[0]
            args = [c1]
        elif m in {HAFB}:
            c1 = [ch[x] for x in f]
            c2 = args[0]
            args = [c1, c2, *args[1:]]
        elif m is Blocks:
            block_type = globals()[args[1]]
            c1, c2 = ch[f], args[0] * block_type.expansion
            args = [c1, args[0], block_type, *args[2:]]
        else:
            c2 = ch[f]

        if isinstance(c2, list) and m not in {ChannelTransformer, PyramidContextExtraction, CrossLayerChannelAttention, CrossLayerSpatialAttention, MutilScaleEdgeInfoGenetator}:
            is_backbone = True
            m_ = m
            m_.backbone = True
        else:
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
        
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i + 4 if is_backbone else i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % (i + 4 if is_backbone else i) for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        if isinstance(c2, list) and m not in {ChannelTransformer, PyramidContextExtraction, CrossLayerChannelAttention, CrossLayerSpatialAttention, MutilScaleEdgeInfoGenetator}:
            ch.extend(c2)
            for _ in range(5 - len(ch)):
                ch.insert(0, 0)
        else:
            ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub(r'(\d+)([nslmx])6(.+)?$', r'\1\2-p6\3', path.stem)
        LOGGER.warning(f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r'(\d+)([nslmx])(.+)?$', r'\1\3', str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r'yolov\d+([nslmx])', Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ''


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg['head'][-1][-2].lower()  # output module name
        if m in ('classify', 'classifier', 'cls', 'fc'):
            return 'classify'
        if m == 'detect':
            return 'detect'
        if m == 'segment':
            return 'segment'
        if m == 'pose':
            return 'pose'

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in 'model.args', 'model.model.args', 'model.model.model.args':
            with contextlib.suppress(Exception):
                return eval(x)['task']
        for x in 'model.yaml', 'model.model.yaml', 'model.model.model.yaml':
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, Detect):
                return 'detect'
            elif isinstance(m, Segment):
                return 'segment'
            elif isinstance(m, Classify):
                return 'classify'
            elif isinstance(m, Pose):
                return 'pose'

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if '-seg' in model.stem or 'segment' in model.parts:
            return 'segment'
        elif '-cls' in model.stem or 'classify' in model.parts:
            return 'classify'
        elif '-pose' in model.stem or 'pose' in model.parts:
            return 'pose'
        elif 'detect' in model.parts:
            return 'detect'

    # Unable to determine task from model
    LOGGER.warning("WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
                   "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.")
    return 'detect'  # assume detect
class OBBModel(DetectionModel):
    """YOLO Oriented Bounding Box (OBB) model."""

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLO OBB model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)

class WorldModel(DetectionModel):
    """YOLOv8 World Model."""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOv8 world model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (List[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
        """
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # for backwards compatibility of models lacking clip_model attribute
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            txt_feats (torch.Tensor, optional): The text features, use it if it's given.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x) or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)
class YOLOEModel(DetectionModel):
    """YOLOE detection model."""

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (List[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
            without_reprta (bool): Whether to return text embeddings cooperated with reprta module.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("mobileclip:blt", device=device)

        model = self.clip_model if cache_clip_model else build_text_model("mobileclip:blt", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        if without_reprta:
            return txt_feats

        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        return head.get_tpe(txt_feats)  # run axuiliary text head

    @smart_inference_mode()
    def get_visual_pe(self, img, visual):
        """
        Get visual embeddings.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features.

        Returns:
            (torch.Tensor): Visual positional embeddings.
        """
        return self(img, vpe=visual, return_vpe=True)

    def set_vocab(self, vocab, names):
        """
        Set vocabulary for the prompt-free model.

        Args:
            vocab (nn.ModuleList): List of vocabulary items.
            names (List[str]): List of class names.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)

        # Cache anchors for head
        device = next(self.parameters()).device
        self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # warmup

        # re-parameterization for prompt-free model
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2)
            for i, (cls, pf, loc) in enumerate(zip(vocab, head.cv3, head.cv2))
        )
        for loc_head, cls_head in zip(head.cv2, head.cv3):
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_vocab(self, names):
        """
        Get fused vocabulary layer from the model.

        Args:
            names (list): List of class names.

        Returns:
            (nn.ModuleList): List of vocabulary modules.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        assert not head.is_fused

        tpe = self.get_text_pe(names)
        self.set_classes(names, tpe)
        device = next(self.model.parameters()).device
        head.fuse(self.pe.to(device))  # fuse prompt embeddings to classify head

        vocab = nn.ModuleList()
        for cls_head in head.cv3:
            assert isinstance(cls_head, nn.Sequential)
            vocab.append(cls_head[-1])
        return vocab

    def set_classes(self, names, embeddings):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            names (List[str]): List of class names.
            embeddings (torch.Tensor): Embeddings tensor.
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
        )
        assert embeddings.ndim == 3
        self.pe = embeddings
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe, vpe):
        """
        Get class positional embeddings.

        Args:
            tpe (torch.Tensor, optional): Text positional embeddings.
            vpe (torch.Tensor, optional): Visual positional embeddings.

        Returns:
            (torch.Tensor): Class positional embeddings.
        """
        all_pe = []
        if tpe is not None:
            assert tpe.ndim == 3
            all_pe.append(tpe)
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(
        self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
    ):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            tpe (torch.Tensor, optional): Text positional embeddings.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.
            vpe (torch.Tensor, optional): Visual positional embeddings.
            return_vpe (bool): If True, return visual positional embeddings.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        b = x.shape[0]
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect):
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe:
                    assert vpe is not None
                    assert not self.training
                    return vpe
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x = m(x, cls_pe)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPDetectLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class FeatureAdapter(nn.Module):
    """适配FasterNet输出的特征列表"""

    def __init__(self, index=-1):
        super().__init__()
        self.index = index

    def forward(self, x):
        if isinstance(x, list):
            return x[self.index]  # 选择指定索引的特征图
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """通用自适应特征图融合模块，无需手动指定目标尺寸"""

    def __init__(self, mode='auto', target_size=None):
        """
        初始化自适应特征融合模块

        Args:
            mode: 特征图尺寸调整策略
                - 'auto': 自动选择最佳尺寸（默认）
                - 'min': 使用最小特征图尺寸
                - 'max': 使用最大特征图尺寸
                - 'mean': 使用平均特征图尺寸
            target_size: 可选的目标尺寸，如果提供则覆盖mode参数
        """
        super().__init__()
        self.mode = mode
        self.target_size = target_size
        self.convs = nn.ModuleDict()  # 动态创建的卷积层字典

    def forward(self, x):
        """前向传播，自动处理特征图尺寸"""
        if not isinstance(x, list):
            return x

        # 获取输入张量的设备和类型
        device = x[0].device
        dtype = x[0].dtype

        # 获取所有特征图的尺寸信息
        sizes = [(f.shape[2], f.shape[3]) for f in x]

        # 确定目标尺寸
        if self.target_size is not None:
            # 使用指定的目标尺寸
            target_size = self.target_size if isinstance(self.target_size, tuple) else (
            self.target_size, self.target_size)
        else:
            # 根据mode自动选择目标尺寸
            if self.mode == 'min' or self.mode == 'auto':
                # 对于小目标检测，通常使用最小尺寸更有效
                min_h = min([s[0] for s in sizes])
                min_w = min([s[1] for s in sizes])
                target_size = (min_h, min_w)
            elif self.mode == 'max':
                # 对于大目标检测，可能需要更高分辨率
                max_h = max([s[0] for s in sizes])
                max_w = max([s[1] for s in sizes])
                target_size = (max_h, max_w)
            elif self.mode == 'mean':
                # 使用平均尺寸作为折中方案
                mean_h = int(sum([s[0] for s in sizes]) / len(sizes))
                mean_w = int(sum([s[1] for s in sizes]) / len(sizes))
                target_size = (mean_h, mean_w)

        # 调整所有特征图到目标尺寸
        adjusted_features = []

        for i, feat in enumerate(x):
            # 获取特征图的形状
            b, c, h, w = feat.shape

            # 为每个不同尺寸/通道的特征图创建1x1卷积层（如果尚未创建）
            size_key = f"{c}_{h}_{w}"
            if size_key not in self.convs:
                # 创建1x1卷积并确保与输入有相同的设备和数据类型
                new_conv = Conv(c, c, 1)
                self.convs[size_key] = new_conv.to(device=device, dtype=dtype)

            # 应用1x1卷积
            feat = self.convs[size_key](feat)

            # 调整特征图尺寸（如果需要）
            if (h, w) != target_size:
                # 根据YOLO惯例，使用双线性插值进行上采样，自适应平均池化进行下采样
                if h * w < target_size[0] * target_size[1]:  # 上采样
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                else:  # 下采样
                    feat = F.adaptive_avg_pool2d(feat, target_size)

            adjusted_features.append(feat)

        # 在通道维度上连接所有特征图
        return torch.cat(adjusted_features, 1)
class ClassifyMultiChannel(nn.Module):
    """适用于多通道输入的分类头"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """
        初始化分类头
        Args:
            c1: 输入通道数（如果为-1，则在前向传播时自动检测）
            c2: 输出类别数
            k, s, p, g: 卷积参数
        """
        super().__init__()
        self.c1 = c1
        c_ = 1280  # 中间特征数
        self.conv = None  # 将在第一次前向传播时初始化（如果c1为-1）
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.2)
        self.linear = nn.Linear(c_, c2)

    def forward(self, x):
        # 自动处理列表输入
        if isinstance(x, list):
            x = torch.cat(x, 1)

        # 动态创建卷积层（如果需要）
        if self.conv is None:
            in_channels = x.shape[1] if self.c1 == -1 else self.c1
            self.conv = Conv(in_channels, 1280, 1).to(device=x.device, dtype=x.dtype)

        # 标准分类流程
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.drop(x)
        x = self.linear(x)

        if self.training:
            return x
        return x.softmax(1)

class FeatureSelector(nn.Module):
    """从特征列表中选择指定索引的特征"""

    def __init__(self, index=-1):
        super().__init__()
        self.index = index

    def forward(self, x):
        if isinstance(x, list):
            return x[self.index]
        return x

class YOLOESegModel(YOLOEModel, SegmentationModel):
    """YOLOE segmentation model."""

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPSegmentLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPSegmentLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Generate the YOLO network's final layer.

        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.

        Returns:
            (tuple): Tuple containing the concatenated predictions and None.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output