import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from . import model_utils
from . import cbam_saca as cbam

def gradient_order(x, h_x=None, w_x=None):
    x0 = x[:, 3:6, :, :]
    x = x[:, 0:3, :, :]
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2)
    xgrad = xgrad * 2
    xgrad = torch.cat((xgrad, x0), 1)
    return xgrad


class HighFeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(HighFeatExtractor, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128, 256, k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256, 256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        x = gradient_order(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.view(-1)
        return out_feat, [n, c, h, w]


class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128, 256, k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256, 256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.view(-1)
        return out_feat, [n, c, h, w]


class MultistageRegression(nn.Module):
    def __init__(self, batchNorm=False):
        super(MultistageRegression, self).__init__()
        self.mconv1 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.mconv2 = model_utils.conv(batchNorm, 128, 128, stride=1, pad=1)
        self.mconv3 = model_utils.conv(batchNorm, 128, 256, stride=2, pad=1)
        self.mconv4 = model_utils.conv(batchNorm, 256, 256, stride=1, pad=1)
        self.mconv5 = model_utils.deconv(256, 128)
        self.mconv6 = model_utils.deconv(128, 64)
        self.red1 = model_utils.conv1_1(batchNorm, 64, 64, stride=1, pad=0)
        self.red2 = model_utils.conv1_1(batchNorm, 128, 128, stride=1, pad=0)

    def forward(self, x):
        red_1 = self.red1(x)  # shortcut1
        out = self.mconv1(x)
        out = self.mconv2(out)  # shortcut2
        red_2 = self.red2(out)
        out = self.mconv3(out)
        out = self.mconv4(out)
        out = self.mconv5(out) + red_2
        out = self.mconv6(out)
        normal_s = torch.nn.functional.normalize(out + red_1, 2, 1)
        return normal_s


class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other = other
        self.deconv0 = model_utils.conv(batchNorm, 512, 256, k=3, stride=1, pad=1)
        self.deconv1 = model_utils.conv(batchNorm, 256, 128, k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.deconv4 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other = other

        self.multi_reg2 = MultistageRegression(batchNorm)
        self.est_normal2 = self._make_output(64, 3, k=3, stride=1, pad=1)

        self.multi_reg3 = MultistageRegression(batchNorm)
        self.est_normal3 = self._make_output(64, 3, k=3, stride=1, pad=1)

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x1, shape1, x2, shape2, f, g):
        x1 = x1.view(shape1[0], shape1[1], shape1[2], shape1[3])
        x2 = x2.view(shape2[0], shape2[1], shape2[2], shape2[3])
        f = f.view(shape1[0], shape1[1], shape1[2], shape1[3])
        g = g.view(shape2[0], shape2[1], shape2[2], shape2[3])
        x = torch.cat((x1, x2, f, g), 1)
        out = self.deconv0(x)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)

        # stage-2
        out2 = self.multi_reg2(out)
        normal2 = self.est_normal2(out2)
        normal2_ = torch.nn.functional.normalize(normal2, 2, 1)

        # stage-3
        out3 = self.multi_reg3(out2)
        normal3 = self.est_normal2(out3)
        normal3_ = torch.nn.functional.normalize(normal3, 2, 1)
        return normal, normal2_, normal3_ 


class IGA_PSN(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(IGA_PSN, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.high_extractor = HighFeatExtractor(batchNorm, c_in, other)
        self.attentionfusion = cbam.CBAM(128)
        self.regressor = Regressor(batchNorm, other)
        self.c_in = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        img = x[0]
        img_split = torch.split(img, 3, 1)
        if len(x) > 1:
            light = x[1]
            light_split = torch.split(light, 3, 1)

        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            grad, gshape = self.high_extractor(net_in)
            feat, shape = self.extractor(net_in)

            featt1 = self.attentionfusion(grad.view(gshape[0], gshape[1], gshape[2], gshape[3]),
                                          feat.view(shape[0], shape[1], shape[2], shape[3]))
            featt2 = self.attentionfusion(feat.view(shape[0], shape[1], shape[2], shape[3]),
                                          grad.view(gshape[0], gshape[1], gshape[2], gshape[3]))
            featss = featt2.view(-1)
            grads = featt1.view(-1)
            feats1 = grad
            feats2 = feat

            if i == 0:
                feat_fused = featss
                grads_fused = grads
                feat1_fused = feats2
                grads1_fused = feats1
            else:
                if self.fuse_type == 'mean':
                    feat_fused = torch.stack([feat_fused, featss], 1).sum(1)
                    grads_fused = torch.stack([grads_fused, grads], 1).sum(1)
                    feat1_fused = torch.stack([feat1_fused, feats2], 1).sum(1)
                    grads1_fused = torch.stack([grads1_fused, feats1], 1).sum(1)
                elif self.fuse_type == 'max':
                    feat_fused, _ = torch.stack([feat_fused, featss], 1).max(1)
                    grads_fused, _ = torch.stack([grads_fused, grads], 1).max(1)
                    feat1_fused, _ = torch.stack([feat1_fused, feats2], 1).max(1)
                    grads1_fused, _ = torch.stack([grads1_fused, feats1], 1).max(1)

        if self.fuse_type == 'mean':
            feat_fused = feat_fused / len(img_split)
            grads_fused = grads_fused / len(img_split)
            feat1_fused = feat1_fused / len(img_split)
            grads1_fused = grads1_fused / len(img_split)

        normal1, normal2, normal3 = self.regressor(feat_fused, shape, grads_fused, gshape, feat1_fused, grads1_fused)

        return normal1, normal2, normal3
