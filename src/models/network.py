import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

class AdaptedResNet(nn.Module):
    def __init__(self, res_name):
        super().__init__()
        self.backbone = ResBase(res_name)
        self.adapter_seq = VisualPromptAdapter()
        self.adapter_par = ConvAdapter()



class ConvAdapter(nn.Module):
    def __init__(self, inplanes, outplanes, width,
                 kernel_size=3, padding=1, stride=1, groups=1, dilation=1, norm_layer=None, act_layer=None,
                 adapt_scale=1.0):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        if act_layer is None:
            act_layer = nn.Identity

        # point-wise conv
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1)
        self.norm1 = norm_layer(width)

        # depth-wise conv
        self.conv2 = nn.Conv2d(width, width, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding,
                               dilation=int(dilation))
        self.norm2 = norm_layer(width)

        # poise-wise conv
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, stride=1)
        self.norm3 = norm_layer(outplanes)

        self.act = act_layer()

        self.adapt_scale = adapt_scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.act(out)

        return out * self.adapt_scale

class VisualPromptPadding(nn.Module):
    def __init__(self, in_channels=3, pad_width=30):
        super().__init__()
        self.pad_width = pad_width
        self.in_channels = in_channels

        # 创建四个边的可训练参数
        self.top_param = nn.Parameter(torch.Tensor(1, in_channels, pad_width, 1))
        self.bottom_param = nn.Parameter(torch.Tensor(1, in_channels, pad_width, 1))
        self.left_param = nn.Parameter(torch.Tensor(1, in_channels, 1, pad_width))
        self.right_param = nn.Parameter(torch.Tensor(1, in_channels, 1, pad_width))

        # 角部参数
        self.tl_corner = nn.Parameter(torch.Tensor(1, in_channels, pad_width, pad_width))
        self.tr_corner = nn.Parameter(torch.Tensor(1, in_channels, pad_width, pad_width))
        self.bl_corner = nn.Parameter(torch.Tensor(1, in_channels, pad_width, pad_width))
        self.br_corner = nn.Parameter(torch.Tensor(1, in_channels, pad_width, pad_width))

        self._init_weights()

    def _init_weights(self):
        for param in [self.top_param, self.bottom_param,
                      self.left_param, self.right_param,
                      self.tl_corner, self.tr_corner,
                      self.bl_corner, self.br_corner]:
            nn.init.zeros_(param)

    def forward(self, x):
        B, C, H, W = x.shape
        pw = self.pad_width

        # 创建空白画布
        padded = torch.zeros(B, C, H + 2 * pw, W + 2 * pw,
                             device=x.device, dtype=x.dtype)

        # 填充中心原始图像
        padded[:, :, pw:-pw, pw:-pw] = x

        # 生成各边参数
        top = self.top_param.expand(B, -1, -1, W + 2 * pw)
        bottom = self.bottom_param.expand(B, -1, -1, W + 2 * pw)
        left = self.left_param.expand(B, -1, H, -1)
        right = self.right_param.expand(B, -1, H, -1)

        # 填充各边
        padded[:, :, :pw, :] = top  # 上边
        padded[:, :, -pw:, :] = bottom  # 下边
        padded[:, :, pw:-pw, :pw] = left  # 左边
        padded[:, :, pw:-pw, -pw:] = right  # 右边

        # 填充四角
        padded[:, :, :pw, :pw] = self.tl_corner  # 左上
        padded[:, :, :pw, -pw:] = self.tr_corner  # 右上
        padded[:, :, -pw:, :pw] = self.bl_corner  # 左下
        padded[:, :, -pw:, -pw:] = self.br_corner  # 右下

        return padded

class VisualPromptOverlay(nn.Module):
    def __init__(self, in_channels=3, pad_width=30, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.pad_width = pad_width
        self.in_channels = in_channels

        # 创建四个边的可训练参数
        self.top_param = nn.Parameter(torch.Tensor(1, in_channels, pad_width, 1))
        self.bottom_param = nn.Parameter(torch.Tensor(1, in_channels, pad_width, 1))
        self.left_param = nn.Parameter(torch.Tensor(1, in_channels, 1, pad_width))
        self.right_param = nn.Parameter(torch.Tensor(1, in_channels, 1, pad_width))

        # 角部参数
        self.tl_corner = nn.Parameter(torch.Tensor(1, in_channels, pad_width, pad_width))
        self.tr_corner = nn.Parameter(torch.Tensor(1, in_channels, pad_width, pad_width))
        self.bl_corner = nn.Parameter(torch.Tensor(1, in_channels, pad_width, pad_width))
        self.br_corner = nn.Parameter(torch.Tensor(1, in_channels, pad_width, pad_width))

        self._init_weights()

    def _init_weights(self):
        for param in [self.top_param, self.bottom_param,
                      self.left_param, self.right_param,
                      self.tl_corner, self.tr_corner,
                      self.bl_corner, self.br_corner]:
            nn.init.zeros_(param)

    def forward(self, x):
        B, C, H, W = x.shape
        pw = self.pad_width
        alpha = self.alpha

        # 克隆原始图像
        padded = x.clone()

        # 生成各边参数
        top = self.top_param.expand(B, -1, -1, W)
        bottom = self.bottom_param.expand(B, -1, -1, W)
        left = self.left_param.expand(B, -1, H, -1)
        right = self.right_param.expand(B, -1, H, -1)

        # 按比例叠加各边
        padded[:, :, :pw, :] = alpha * top + (1 - alpha) * padded[:, :, :pw, :]  # 上边
        padded[:, :, -pw:, :] = alpha * bottom + (1 - alpha) * padded[:, :, -pw:, :]  # 下边
        padded[:, :, :, :pw] = alpha * left + (1 - alpha) * padded[:, :, :, :pw]  # 左边
        padded[:, :, :, -pw:] = alpha * right + (1 - alpha) * padded[:, :, :, -pw:]  # 右边

        # 按比例叠加四角
        padded[:, :, :pw, :pw] = alpha * self.tl_corner + (1 - alpha) * padded[:, :, :pw, :pw]  # 左上
        padded[:, :, :pw, -pw:] = alpha * self.tr_corner + (1 - alpha) * padded[:, :, :pw, -pw:]  # 右上
        padded[:, :, -pw:, :pw] = alpha * self.bl_corner + (1 - alpha) * padded[:, :, -pw:, :pw]  # 左下
        padded[:, :, -pw:, -pw:] = alpha * self.br_corner + (1 - alpha) * padded[:, :, -pw:, -pw:]  # 右下

        return padded

class VisualPromptAdapter(nn.Module):
    def __init__(self, input_channels=3, adapter_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, adapter_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(adapter_channels, eps=1e-5)
        self.attention_conv = nn.Conv2d(adapter_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.domain_align = nn.Sequential(
                nn.Conv2d(adapter_channels, adapter_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(adapter_channels, input_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(input_channels)
                )
        self.residual = nn.Identity()
        self.init_weights()

    def forward(self, x):
        features = F.relu(self.bn1(self.conv1(x)))

        # Compute attention map
        attention = self.sigmoid(self.attention_conv(features))

        # Apply attention
        attended_features = features * attention

        # Domain alignment
        domain_aligned = self.domain_align(attended_features)

        # Residual connection (modified input)
        output = self.residual(x) + domain_aligned

        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19, 
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 
class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, 
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name, conv_adapter=False):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](weights=ResNet50_Weights.DEFAULT)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResAnomaly(nn.Module):
    def __init__(self, res_name):
        super(ResAnomaly, self).__init__()
        model_resnet = res_dict[res_name](weights=ResNet50_Weights.DEFAULT)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.similarity_conv = nn.Conv2d(2048, 2, kernel_size=1)
        self.avgpool = model_resnet.avgpool
        self.classification_fc = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 2)  # 2个类别：正常和异常
                )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        similarity_map = self.similarity_conv(x)
        # print(y.shape)
        x = self.avgpool(x)
        # print(y.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        class_logits = self.classification_fc(x)
        return class_logits, similarity_map

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv(x)
        return x

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

class Net2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim,bias=True)

    def forward(self, x):
        out = self.linear(x)
        return out

class ResNet_FE(nn.Module):
    def __init__(self):
        super().__init__()
        model_resnet = torchvision.models.resnet50(True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu,
                                            self.maxpool, self.layer1,
                                            self.layer2, self.layer3,
                                            self.layer4, self.avgpool)
        self.bottle = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)

    def forward(self, x):
        out = self.feature_layers(x)
        out = out.view(out.size(0), -1)
        out = self.bn(self.bottle(out))
        return out


# if __name__ == '__main__':
#     vpt = VisualPrompt(pad_width=10, in_channels=3)
#     input_tensor = torch.randn(2, 3, 224, 224)  # 任意尺寸输入
#     output = vpt(input_tensor)  # 输出形状：[2, 3, 244, 244]
#     print(output.shape)
