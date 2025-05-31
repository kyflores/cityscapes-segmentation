import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBasic(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            cin, cout, kernel_size=3, stride=1, padding=1, padding_mode="replicate"
        )
        self.bn_1 = nn.BatchNorm2d(cout)

        self.conv_2 = nn.Conv2d(
            cout, cout, kernel_size=3, stride=1, padding=1, padding_mode="replicate"
        )
        self.bn_2 = nn.BatchNorm2d(cout)

        self.skip_proj = nn.Conv2d(cin, cout, kernel_size=1, stride=1)
        self.skip_bn = nn.BatchNorm2d(cout)

        self.act = nn.ReLU()

    def forward(self, x):
        xid = self.skip_proj(x)
        xid = self.skip_bn(xid)

        x = self.conv_1(x)
        x = self.act(x)
        x = self.bn_1(x)

        x = self.conv_2(x)
        x = self.act(x)
        x = self.bn_2(x)

        return x + xid


class ResnetDilated(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            cin, cout, kernel_size=3, stride=1, padding=1, padding_mode="replicate"
        )
        self.bn_1 = nn.BatchNorm2d(cout)
        self.conv_2 = nn.Conv2d(
            cout,
            cout,
            kernel_size=3,
            stride=1,
            dilation=2,
            padding=2,
            padding_mode="replicate",
        )
        self.bn_2 = nn.BatchNorm2d(cout)

        self.conv_3 = nn.Conv2d(
            cin,
            cout,
            kernel_size=3,
            stride=1,
            dilation=2,
            padding=2,
            padding_mode="replicate",
        )
        self.bn_3 = nn.BatchNorm2d(cout)
        self.conv_4 = nn.Conv2d(
            cout,
            cout,
            kernel_size=3,
            stride=1,
            dilation=4,
            padding=4,
            padding_mode="replicate",
        )
        self.bn_4 = nn.BatchNorm2d(cout)

        self.skip_proj = nn.Conv2d(cin, cout, kernel_size=1, stride=1)
        self.skip_bn = nn.BatchNorm2d(cout)

        self.act = nn.ReLU()

    def forward(self, x):
        xid = self.skip_proj(x)
        xid = self.skip_bn(xid)

        x_l = self.conv_1(x)
        x_l = self.act(x_l)
        x_l = self.bn_1(x_l)
        x_l = self.conv_2(x_l)
        x_l = self.act(x_l)
        x_l = self.bn_2(x_l)

        x_r = self.conv_1(x)
        x_r = self.act(x_r)
        x_r = self.bn_1(x_r)
        x_r = self.conv_2(x_r)
        x_r = self.act(x_r)
        x_r = self.bn_2(x_r)

        return x_l + x_r + xid


class ResnetBottleneck(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        c_scale = 4
        low_c = cin // c_scale

        self.conv_1 = nn.Conv2d(cin, low_c, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(low_c)

        self.conv_2 = nn.Conv2d(low_c, low_c, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(low_c)

        self.conv_3 = nn.Conv2d(low_c, cout, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(cout)

        self.skip_proj = nn.Conv2d(cin, cout, kernel_size=1, stride=1)
        self.skip_bn = nn.BatchNorm2d(cout)

        self.act = nn.ReLU()

    def forward(self, x):
        xid = self.skip_proj(x)
        xid = self.skip_bn(xid)

        x = self.conv_1(x)
        x = self.act(x)
        x = self.bn_1(x)

        x = self.conv_2(x)
        x = self.act(x)
        x = self.bn_2(x)

        x = self.conv_3(x)
        x = self.act(x)
        x = self.bn_3(x)

        return x + xid


class UpConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.upconv_1 = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        self.bn_1 = nn.BatchNorm2d(cout)

        # Cat with through connection happens here, which doubles the number of channels.

        self.conv_1 = ResnetBasic(cin, cout)
        self.conv_2 = ResnetBasic(cout, cout)

    def forward(self, x, xcat):
        # Cat on channel dim
        x = self.upconv_1(x)
        x = self.bn_1(x)

        tmp = torch.cat((x, xcat), dim=1)

        x = self.conv_1(tmp)
        x = self.conv_2(x)

        return x


class UnetSeg(nn.Module):
    def __init__(self, cin, cout, filts=32):
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_in = nn.Conv2d(cin, filts, kernel_size=7, stride=2, padding=3)
        self.bn_in = nn.BatchNorm2d(filts)

        self.l1 = ResnetBasic(filts, filts)
        self.l2 = ResnetBasic(filts, filts * 2)
        self.l3 = ResnetBasic(filts * 2, filts * 4)
        self.l4 = ResnetBasic(filts * 4, filts * 8)
        self.l5 = ResnetBasic(filts * 8, filts * 16)

        self.u1 = UpConv(filts * 16, filts * 8)
        self.u2 = UpConv(filts * 8, filts * 4)
        self.u3 = UpConv(filts * 4, filts * 2)
        self.u4 = UpConv(filts * 2, filts)

        # Final upsample is to get the image back to input size
        self.conv_out = nn.Conv2d(filts, cout, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)

        # Down
        x0 = self.l1(x)
        x1 = self.l2(self.downsample(x0))
        x2 = self.l3(self.downsample(x1))
        x3 = self.l4(self.downsample(x2))

        x4 = self.l5(self.downsample(x3))

        # Up
        xu3 = self.u1(x4, x3)
        xu4 = self.u2(xu3, x2)
        xu5 = self.u3(xu4, x1)
        xu6 = self.u4(xu5, x0)

        # Output
        xout = self.conv_out(xu6)
        xout = self.upsample(xout)

        return xout


if __name__ == "__main__":
    model = UnetSeg(3, 20)

    out = model(torch.randn(1, 3, 256, 256))
    print(out.shape)

    basic = ResnetBasic(64, 64)
    print(basic(torch.randn(7, 64, 128, 128)).shape)

    bottleneck = ResnetBottleneck(64, 64)
    print(bottleneck(torch.randn(7, 64, 128, 128)).shape)
