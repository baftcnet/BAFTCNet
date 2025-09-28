import sys
import torch
import torch.nn as nn
import torch
from torchinfo import summary
from torchstat import stat


class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        # if self.bias:
        #     self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

    def __call__(self, *input, **kwargs):
        return super()._call_impl(*input, **kwargs)


class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)
        if self.bias:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)

class Conv1dWithConstraint1(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def forward(self, x):
        return super(Conv1dWithConstraint1, self).forward(x)



# %%
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class LinearWithConstraint1(nn.Linear):
    def forward(self, x):
        return super(LinearWithConstraint1, self).forward(x)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

#%%
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, bias=False, WeightNorm=False, group=True, max_norm=1.):
        super(TemporalBlock, self).__init__()
        if group:
            if n_inputs >= n_outputs:
                self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                                  dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_outputs)
            else:
                self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                                  dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_inputs)
            self.conv1_point = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size=1, stride=1, bias=bias)
        else:
            self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu1 = nn.ELU() # inplace=True
        self.dropout1 = nn.Dropout(dropout)

        if group:
            self.conv2 = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_outputs)
            self.conv2_point = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size=1, stride=1, bias=bias)
        else:
            self.conv2 = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        if group:
            self.net = nn.Sequential(self.conv1, self.conv1_point, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                     self.conv2, self.conv2_point, self.chomp2, self.bn2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = out+res
        out = self.relu(out)
        return out



def relative_pos_dis(height=1, weight=32):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    if height > 1:
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2 # Wh*Ww, Wh*Ww
    else:
        relative_coords = coords_flatten[1, :, None] - coords_flatten[1, None, :]  # Wh*Ww, Wh*Ww
        relative_coords = relative_coords.contiguous()  # Wh*Ww, Wh*Ww
        dis = relative_coords # Wh*Ww, Wh*Ww
    return  dis

import torch
import torch.nn as nn
from torch.autograd import Function



#%%
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

#%%
class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        # if self.bias:
        #     self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
    
    def __call__(self, *input, **kwargs):
        return super()._call_impl(*input, **kwargs)

class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)
        if self.bias:
            self.bias.data.fill_(0.0)
            
    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)

#%%
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

#%%

#%%
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

#%%
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, bias=False, WeightNorm=False, group=True, max_norm=1.):
        super(TemporalBlock, self).__init__()
        if group:
            if n_inputs >= n_outputs:
                self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                                  dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_outputs)
            else:
                self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                                  dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_inputs)
            self.conv1_point = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size=1, stride=1, bias=bias)
        else:
            self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu1 = nn.ELU() # inplace=True
        self.dropout1 = nn.Dropout(dropout)

        if group:
            self.conv2 = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_outputs)
            self.conv2_point = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size=1, stride=1, bias=bias)
        else:
            self.conv2 = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        if group:
            self.net = nn.Sequential(self.conv1, self.conv1_point, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                     self.conv2, self.conv2_point, self.chomp2, self.bn2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = out+res
        out = self.relu(out)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, bias=False, WeightNorm=False, group=True, max_norm=1.):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, 
                                     dropout=dropout, bias=bias, WeightNorm=WeightNorm, group=group, max_norm=max_norm)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class TemporalInception(nn.Module):
    def __init__(self, in_chan=1, kerSize_1=(1, 3), kerSize_2=(1, 5), kerSize_3=(1, 7),
                 kerStr=1, out_chan=4, pool_ker=(1, 3), pool_str=1, bias=False, max_norm=1., point_fusion=True):
        '''
        Inception模块的实现代码,
        '''
        super(TemporalInception, self).__init__()
        self.point_fusion = point_fusion
        self.conv1 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kerSize_1,
            stride=kerStr,
            padding='same',
            groups=out_chan,
            bias=bias,
            max_norm=max_norm
        )
        self.conv2 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kerSize_2,
            stride=kerStr,
            padding='same',
            groups=out_chan,
            bias=bias,
            max_norm=max_norm
        )
        self.conv3 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kerSize_3,
            stride=kerStr,
            padding='same',
            groups=out_chan,
            bias=bias,
            max_norm=max_norm
        )
        self.pool4 = nn.MaxPool2d(
            kernel_size=pool_ker,
            stride=pool_str,
            padding=(round(pool_ker[0] / 2 + 0.1) - 1, round(pool_ker[1] / 2 + 0.1) - 1)
        )
        self.conv4 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=1,
            stride=1,
            bias=bias,
            max_norm=max_norm
        )
        self.ce = ECABlock()
        self.ta = TimeSeriesModel(window_size=50)
    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(x)
        p4 = self.conv4(self.pool4(x))
        out = torch.cat((p1, p2, p3, p4), dim=1)
        x = self.ce(p1, p2, p3, p4)
        out = self.ta(x)
        return out


class TriInputAttention(nn.Module):
    def __init__(self):
        super(TriInputAttention, self).__init__()
        self.attention_qector = nn.Parameter(torch.tensor(1.0))
        self.attention_cector = nn.Parameter(torch.tensor(1.0))
        self.attention_vector = nn.Parameter(torch.tensor(1.0))
        self.out = Conv2dWithConstraint(in_channels=96,out_channels=32,kernel_size=1,stride=1,bias=False)
    def normalize_attention_weights(self):
        # 计算当前参数的总和
        total_sum = self.attention_vector + self.attention_qector + self.attention_cector
        # 归一化，使总和为3
        with torch.no_grad():
            self.attention_vector.data = (self.attention_vector / total_sum) * 3
            self.attention_qector.data = (self.attention_qector / total_sum) * 3
            self.attention_cector.data = (self.attention_cector / total_sum) * 3
    def forward(self, x1, x2, x3):
        # 计算每个输入的线性变换
        self.normalize_attention_weights()
        h1 = self.attention_vector*x1
        h2 = self.attention_qector*x2
        h3 = self.attention_cector*x3
        out = torch.cat((h1, h2, h3), dim=1)
        out = self.out(out)
        return out      

class ECABlock(nn.Module):
    def __init__(self, k_size=3):
        super(ECABlock, self).__init__()
        self.global_avg_pool12 = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，将空间维度变为 1x1
        self.conv = Conv1dWithConstraint1(1, 1, kernel_size=k_size,bias=False,padding=1)
        self.sigmoid = nn.Sigmoid()  # 输出通道权重，范围在 0 到 1 之间
    def forward(self, x1, x2, x3, x4):
        batch_size, channels, _, _ = x1.size()
        # 处理 x1
        y1 = self.global_avg_pool12(x1).view(batch_size, 1, channels)  # 调整为 (batch_size, 1, channels) 以适应1D卷积
        y1 = self.conv(y1)  # 1D卷积
        y1 = self.sigmoid(y1).view(batch_size, channels, 1, 1)  # 应用Sigmoid激活并调整回 (batch_size, channels, 1, 1)
        y1 = x1 * y1.expand_as(x1)  # 权重应用于输入 x1
        # 处理 x2
        y2 = self.global_avg_pool12(x2).view(batch_size, 1, channels)
        y2 = self.conv(y2)
        y2 = self.sigmoid(y2).view(batch_size, channels, 1, 1)
        y2 = x2 * y2.expand_as(x2)
        # 处理 x3
        y3 = self.global_avg_pool12(x3).view(batch_size, 1, channels)
        y3 = self.conv(y3)
        y3 = self.sigmoid(y3).view(batch_size, channels, 1, 1)
        y3 = x3 * y3.expand_as(x3)
        # 处理 x4
        y4 = self.global_avg_pool12(x4).view(batch_size, 1, channels)
        y4 = self.conv(y4)
        y4 = self.sigmoid(y4).view(batch_size, channels, 1, 1)
        y4 = x4 * y4.expand_as(x4)
        # 将处理后的特征拼接在一起
        out = torch.cat((y1, y2, y3, y4), dim=1)
        return out

class SEBlock(nn.Module):
    def __init__(self, time_steps=50, reduction=50):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            LinearWithConstraint1(50, 2),
            nn.ReLU(),
            LinearWithConstraint1(2, 50),
            nn.Sigmoid()
        )
    def forward(self, x):
        # 这里x的形状为 (batch_size, time_steps)
        se_weights = self.fc(x)
        # 扩展权重以匹配输入的形状
        return se_weights.unsqueeze(1)  # (batch_size, 1, 1, time_steps)

class SEAlock(nn.Module):
    def __init__(self, time_steps=25, reduction=25):
        super(SEAlock, self).__init__()
        self.fc = nn.Sequential(
            LinearWithConstraint1(25, 1),
            nn.ReLU(),
            LinearWithConstraint1(1, 25),
            nn.Sigmoid()
        )
    def forward(self, x):
        # 这里x的形状为 (batch_size, time_steps)
        se_weights = self.fc(x)
        # 扩展权重以匹配输入的形状
        return se_weights.unsqueeze(1)  # (batch_size, 1, 1, time_steps)

class TimeSeriesModel(nn.Module):
    def __init__(self, window_size, num_pipelines=32, num_time_steps=250, reduction=50, overlap=25 ):
        super(TimeSeriesModel, self).__init__()
        self.window_size = window_size
        self.overlap = overlap
        self.se_block = SEBlock(50, 25)
        self.se_block_for_25 = SEAlock(25,25)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((None, 1))  
    def forward(self, x):
        # 去掉第三维度（通道维度）
        x = x.squeeze(2)  # x的形状变为 (batch_size, num_pipelines, time_steps
        b, num_pipelines, t = x.shape
        # 1. 对数据进行平均处理（每个时间步上的管道平均）
        x_mean = x.permute(0, 2, 1)
        x_mean = self.global_avg_pool(x_mean)
        x_mean = x_mean.permute(0, 2, 1)
        windowed = []
        se_weights = []
        weights = torch.zeros(b, 1, 1, 250).to(x.device)
        for i in range(0, t - self.window_size + self.overlap, self.overlap):
            end = min(i + self.window_size, t)
            window = x_mean[:, :, i:end]
            windowed.append(window)
        while len(windowed) < 10:
            last_start = t - self.window_size - (len(windowed) * self.overlap)
            last_window = x_mean[:, :, last_start:t]
            windowed.append(last_window)
            # print(f"Window shape at iteration {len(windowed) - 1}: {last_window.shape}")
        # 3. 对于每份数据应用SE块计算权重
        for i, window in enumerate(windowed):
            if window.shape[2] == 25:
                se_weight = self.se_block_for_25(window)  # 假设有一个专门处理25大小的SE块
            else:
                se_weight = self.se_block(window)
            se_weights.append(se_weight)
        for i, weight in enumerate(se_weights):
            start = i * self.overlap
            end = start + self.window_size
            if i == 0:
                weights[:, :, :, start:end] = weight[:, :, :, :self.window_size]
            else:
                overlap_start = start + self.overlap
                # 确保重叠部分的索引正确
                weights[:, :, :, start:overlap_start] += weight[:, :, :, :self.overlap]
                # 调整重叠部分的切片以匹配长度
                weights[:, :, :, overlap_start:end] += weight[:, :, :, self.overlap:]  # 重叠部分的末尾元素
                weights[:, :, :, start:overlap_start] /= 2  # 如果需要取平均
        return x.unsqueeze(2) * weights



class EEGNet(nn.Module):
    def __init__(self, eeg_chans=22, samples=1000, dropoutRate=0.2, kerSize=32, kerSize_Tem=4,F1=16,D=2, 
                 poolSize1=8,dropout_atten=0.3,bias=False,n_classes=4,poolSize2=8,dropout_temp=0.3,tcn_filters=32, tcn_kernelSize=4, tcn_dropout=0.3,dropout_zhojian = 0.35):
        super(EEGNet, self).__init__()
        F2 = F1*D
        F3 = 32
        self.temp_conv = Conv2dWithConstraint(  # Conv2dWithConstraint( # sincConv
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kerSize),
            stride=1,
            padding='same',
            bias=False,
            max_norm=.5
        )
        self.bn = nn.BatchNorm2d(num_features=F1)  # bn_sinc
        self.incept_temp = TemporalInception(
            in_chan=F2,
            kerSize_1=(1, kerSize_Tem * 4),
            kerSize_2=(1, kerSize_Tem * 2),
            kerSize_3=(1, kerSize_Tem),
            kerStr=1,
            out_chan=F2//4,
            pool_ker=(1, 3),
            pool_str=1,
            bias=False,
            max_norm=.5
        )
        self.point_conv = Conv2dWithConstraint(
                in_channels=96,
                out_channels=32,
                kernel_size=1,
                stride=1,
                bias=False
        )
        self.drop_temp = nn.Dropout(p=dropout_temp)        
        self.region_channels = {
            'Frontal Region': [0,1,2,3,4,5,6,7,8,9,10,11,12],  
            'Central Region': [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
            'zong': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        }
        self.num_regions = len(self.region_channels)
        # 动态卷积层
        self.depthwise_convs = nn.ModuleDict(
            {
                region: nn.Sequential(
                    Conv2dWithConstraint(
                        in_channels=F1,
                        out_channels=F1*D,
                        kernel_size=(len(channels), 1),
                        bias=bias, 
                        groups=F1,
                        max_norm=0.5                 
                    ),
                    nn.BatchNorm2d(num_features=F1*D),
                    nn.ELU(),
                    nn.AvgPool2d(
                        kernel_size=(1, 4),
                        stride=(1, 4)
                    ),
                    nn.Dropout(p=dropoutRate)
                )
                for region, channels in self.region_channels.items()
            }
        )
        self.bn_temp = nn.BatchNorm2d(num_features=32)
        self.act_temp = nn.ELU()
        self.avgpool_temp = nn.AvgPool2d(
            kernel_size=(1, poolSize2),
            stride=(1, poolSize2)
        )
        self.avgpool1_temp = nn.AvgPool2d(
            kernel_size=(1, 2),
            stride=(1, 2)
        )
        self.drop_temp = nn.Dropout(p=dropout_temp)        
        self.drop_zhojiantemp = nn.Dropout(p=dropout_zhojian)  
        self.tcn_block = TemporalConvNet(
            num_inputs  =32,
            num_channels=[tcn_filters, tcn_filters],
            kernel_size =tcn_kernelSize,
            dropout     =tcn_dropout
        )
        self.flatten_eeg = nn.Flatten()
        self.liner_eeg = LinearWithConstraint(
            in_features  = 992,
            out_features = n_classes,
            max_norm     = .5,
            bias         = True
        )
        self.class_head = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(
                in_features=32,
                out_features=n_classes,
                max_norm=.5,
                bias = True,
            ),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.beta = nn.Parameter(torch.randn(1, requires_grad=True))
        self.beta_sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 打印输入初始形状
        print("Input x shape: (batch_size, channels, height, width) = (%d, %d, %d, %d)" % 
            (x.shape[0], x.shape[1], x.shape[2], x.shape[3]) if len(x.shape) == 4 else 
            "Input x shape: (batch_size, height, width) = (%d, %d, %d)" % (x.shape[0], x.shape[1], x.shape[2]))
        
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, 1)
            print("After unsqueeze: x shape = (%d, %d, %d, %d)" % (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        
        x = self.temp_conv(x)
        print("After temp_conv: x shape = (%d, %d, %d, %d)" % (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        
        x = self.bn(x)
        print("After bn: x shape = (%d, %d, %d, %d)" % (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        
        region_outputs = []
        for region, channels in self.region_channels.items():
            region_x = x[:, :, channels, :]
            print("Region %s after slice: shape = (%d, %d, %d, %d)" % 
                (region, region_x.shape[0], region_x.shape[1], region_x.shape[2], region_x.shape[3]))
            
            region_x = self.depthwise_convs[region](region_x)
            print("Region %s after depthwise_conv: shape = (%d, %d, %d, %d)" % 
                (region, region_x.shape[0], region_x.shape[1], region_x.shape[2], region_x.shape[3]))
            
            region_x = self.incept_temp(region_x)
            print("Region %s after incept_temp: shape = (%d, %d, %d, %d)" % 
                (region, region_x.shape[0], region_x.shape[1], region_x.shape[2], region_x.shape[3]))
            
            region_outputs.append(region_x)
        
        x = torch.cat(region_outputs, dim=1)
        print("After concatenation: x shape = (%d, %d, %d, %d)" % (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        
        x = self.point_conv(x)
        print("After point_conv: x shape = (%d, %d, %d, %d)" % (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        
        x = self.avgpool_temp(self.act_temp(self.bn_temp(x)))
        print("After avgpool_temp: x shape = (%d, %d, %d, %d)" % (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        
        eeg = self.drop_zhojiantemp(x)
        print("eeg shape after drop_zhojiantemp: (%d, %d, %d, %d)" % (eeg.shape[0], eeg.shape[1], eeg.shape[2], eeg.shape[3]))
        
        eeg_out = self.liner_eeg(self.flatten_eeg(eeg))
        print("eeg_out shape: (%d, %d)" % (eeg_out.shape[0], eeg_out.shape[1]))
        
        x = self.drop_temp(x)
        print("x shape after drop_temp: (%d, %d, %d, %d)" % (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        
        x = torch.squeeze(x, dim=2)
        print("After squeeze: x shape (NCW) = (%d, %d, %d)" % (x.shape[0], x.shape[1], x.shape[2]))
        
        x = self.tcn_block(x)
        print("After tcn_block: x shape (NWC) = (%d, %d, %d)" % (x.shape[0], x.shape[1], x.shape[2]))
        
        x = x[:, :, -1]
        print("After taking last time step: x shape = (%d, %d)" % (x.shape[0], x.shape[1]))
        
        x = self.class_head(x)
        print("After class_head: x shape = (%d, %d)" % (x.shape[0], x.shape[1]))
        
        fusionDecision = self.beta_sigmoid(self.beta)*eeg_out + (1-self.beta_sigmoid(self.beta))*x
        print("fusionDecision shape: (%d, %d)" % (fusionDecision.shape[0], fusionDecision.shape[1]))
        
        out = self.softmax(fusionDecision)
        print("Final output shape: (%d, %d)" % (out.shape[0], out.shape[1]))
        
        return out
    
from thop import profile

#%%
###============================ Initialization parameters ============================###
def main():
    channels = 22  # 假设22个EEG通道
    samples = 1000  # 假设每个通道1000个样本
    input_data = torch.randn(32, channels, samples)  # 模拟批次大小为32的输入

    # 检查是否有 GPU 可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 创建模型并将其移动到 GPU 或 CPU
    model = EEGNet().to(device)
    # 将输入数据移到 GPU 或 CPU
    input_data = input_data.to(device)
    print('===============================================================')
    print('Model summary:')
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))


if __name__ == "__main__":
    main()
