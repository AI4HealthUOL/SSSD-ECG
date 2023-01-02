import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.cond_batchnorm import ConditionalBatchNorm1d


class Transpose1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample

        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        if self.upsample:
            #x = torch.cat((x, in_feature), 1)
            return self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        else:
            return self.Conv1dTrans(x)

class Transpose1dLayer_multi_input(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1):
        super(Transpose1dLayer_multi_input, self).__init__()
        self.upsample = upsample

        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x, in_feature):
        if self.upsample:
            x = torch.cat((x, in_feature), 1)                           
            return self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        else:
            return self.Conv1dTrans(x)


class P2PCondGenerator(nn.Module):
    def __init__(self, model_size=50, ngpus=1, num_channels=8,
                 post_proc_filt_len=512,
                 verbose=False, upsample=True):
        super(WaveGANGenerator, self).__init__()
        self.ngpus = ngpus
        self.model_size = model_size  # d
        self.num_channels = num_channels  # c
        self.post_proc_filt_len = post_proc_filt_len
        self.verbose = verbose

        stride = 4
        if upsample:
            stride = 1
            upsample = 5
        self.deconv_2 = Transpose1dLayer_multi_input(5 * model_size * 2, 3 * model_size, 25, stride, upsample=upsample)
        self.deconv_3 = Transpose1dLayer_multi_input(3 * model_size * 2,  model_size, 25, stride, upsample=upsample)
        self.deconv_5 = Transpose1dLayer_multi_input( model_size * 2, int(model_size / 2), 25, stride, upsample=2)
        self.deconv_6 = Transpose1dLayer_multi_input(  int(model_size / 2) * 2, int(model_size / 5), 25, stride, upsample=upsample)
        self.deconv_7 = Transpose1dLayer(  int(model_size / 5), num_channels, 25, stride, upsample=2)

        self.conv_1 = nn.Conv1d(num_channels, int(model_size / 5), 25, stride=2, padding=25 // 2)
        self.conv_2 = nn.Conv1d(model_size // 5, model_size // 2, 25, stride=5, padding= 25 // 2)
        self.conv_3 = nn.Conv1d(model_size // 2, model_size , 25, stride=2, padding= 25 // 2)
        self.conv_4 = nn.Conv1d(model_size, model_size * 3 , 25, stride=5, padding= 25 // 2)
        self.conv_5 = nn.Conv1d(model_size * 3, model_size * 5 , 25, stride=5, padding= 25 // 2)
        
        self.bn_conv1 = ConditionalBatchNorm1d(int(model_size / 5), 71)
        self.bn_conv2 = ConditionalBatchNorm1d(model_size // 2, 71)
        self.bn_conv3 = ConditionalBatchNorm1d(model_size, 71)
        self.bn_conv4 = ConditionalBatchNorm1d(model_size * 3, 71)
        self.bn_conv5 = ConditionalBatchNorm1d(model_size * 5, 71)
        
        self.bn_deconv2 = ConditionalBatchNorm1d(3 * model_size, 71)
        self.bn_deconv3 = ConditionalBatchNorm1d(model_size, 71)
        self.bn_deconv5 = ConditionalBatchNorm1d(int(model_size / 2), 71)
        self.bn_deconv6 = ConditionalBatchNorm1d(int(model_size / 5), 71)
        self.bn_deconv7 = ConditionalBatchNorm1d(num_channels, 71)
        

        if post_proc_filt_len:
            self.ppfilter1 = nn.Conv1d(num_channels, num_channels, post_proc_filt_len)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, y):

        conv_1_out = F.leaky_relu(self.bn_conv1( self.conv_1(x), y)) # x = (bs, 8, 1000)
        conv_2_out = F.leaky_relu(self.bn_conv2(self.conv_2(conv_1_out), y))
        conv_3_out = F.leaky_relu(self.bn_conv3(self.conv_3(conv_2_out), y))
        conv_4_out = F.leaky_relu(self.bn_conv4(self.conv_4(conv_3_out), y))
        conv_5_out = F.leaky_relu(self.bn_conv5(self.conv_5(conv_4_out), y))
        x = conv_5_out.clone()

        x = F.relu(self.bn_deconv2(self.deconv_2(x, conv_5_out), y))  # prob
        if self.verbose:
            print(x.shape)

        x = F.relu(self.bn_deconv3(self.deconv_3(x, conv_4_out), y))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.bn_deconv5(self.deconv_5(x, conv_3_out), y))
        if self.verbose:
            print(x.shape)
        
        x = F.relu(self.bn_deconv6(self.deconv_6(x, conv_2_out), y))
        if self.verbose:
            print(x.shape)

        output = torch.tanh(self.bn_deconv7(self.deconv_7(x), y))

        if self.verbose:
            print(output.shape)
        return output


    
    
class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """
    
    # Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                       x.shape)
        return x_shuffle


class PhaseRemove(nn.Module):
    def __init__(self):
        super(PhaseRemove, self).__init__()

    def forward(self, x):
        pass


class P2PCondDiscriminator(nn.Module):
    def __init__(self, model_size=64, ngpus=1, num_channels=8, shift_factor=2,
                 alpha=0.2, verbose=False):
        super(WaveGANDiscriminator, self).__init__()
        self.model_size = model_size  # d
        self.ngpus = ngpus
        self.num_channels = num_channels  # c
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose

        self.conv1 = nn.Conv1d(num_channels,  model_size, 25, stride=2, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=2, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 5 * model_size, 25, stride=2, padding=11)
        self.conv4 = nn.Conv1d(5 * model_size, 10 * model_size, 25, stride=2, padding=11)
        self.conv5 = nn.Conv1d(10 * model_size, 20 * model_size, 25, stride=4, padding=11)
        self.conv6 = nn.Conv1d(20 * model_size, 25 * model_size, 25, stride=4, padding=11)

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)
        self.ps5 = PhaseShuffle(shift_factor)

        self.fc1 = nn.LazyLinear(1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps4(x)

        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps5(x)

        x = F.leaky_relu(self.conv6(x), negative_slope=self.alpha)
        
        x = x.view(-1, x.shape[1] * x.shape[2])
        if self.verbose:
            print(x.shape)

        return self.fc1(x)
