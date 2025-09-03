import torch
import torch.nn as nn
from torch.autograd import Function

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

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

#%%
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, bias=False, WeightNorm=False, max_norm=1.):
        super(TemporalBlock, self).__init__()
        self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                          dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                          dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

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

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, bias=False, WeightNorm=False, max_norm=1.):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout, bias=bias, WeightNorm=WeightNorm, max_norm=max_norm)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SEBlock(nn.Module):
    def __init__(self, time_steps=10, reduction=10):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(time_steps, time_steps // reduction),
            nn.ReLU(),
            nn.Linear(time_steps // reduction, time_steps),
            nn.Sigmoid()
        )
    def forward(self, x):
        # 这里x的形状为 (batch_size, time_steps)
        se_weights = self.fc(x)
        # 扩展权重以匹配输入的形状
        return se_weights.unsqueeze(1)  # (batch_size, 1, 1, time_steps)

class SEAlock(nn.Module):
    def __init__(self, time_steps=5, reduction=5):
        super(SEAlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(time_steps, time_steps // reduction),
            nn.ReLU(),
            nn.Linear(time_steps // reduction, time_steps),
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
        self.num_time_steps = num_time_steps
        self.num_pipelines = num_pipelines
        self.se_block = SEBlock(50, 25)
        self.se_block_for_25 = SEAlock(25,25)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((None, 1))  
    def forward(self, x):
        # 去掉第三维度（通道维度）
        x = x.squeeze(2)  # x的形状变为 (batch_size, num_pipelines, time_steps)
        b, num_pipelines, t = x.shape
        # 1. 对数据进行平均处理（每个时间步上的管道平均）
        x_mean = x.permute(0, 2, 1)
        x_mean = self.global_avg_pool(x_mean)
        x_mean = x_mean.permute(0, 2, 1)
        weights = torch.zeros(b, 1, 1, 250).to(x.device)
        index = 0
        num_windows_added = 0
        while num_windows_added < 10:
            if index < t - self.window_size + self.overlap:
                start = index
                end = min(start + self.window_size, t)
            else:
                last_start = t - self.overlap
                start = last_start
                end = t
            window = x_mean[:, :, start:end]
            if window.shape[2] == 25:
                se_weight = self.se_block_for_25(window)
            else:
                se_weight = self.se_block(window)
            if num_windows_added == 0:
                weights[:, :, :, start:end] = se_weight[:, :, :, :self.window_size]
            else:
                weights[:, :, :, start:end] += se_weight[:, :, :, :end]
                # weights[:, :, :, overlap_start:end] += se_weight[:, :, :, self.overlap:]  # 重叠部分的末尾元素
            # 更新 num_windows_added 和 index
            num_windows_added += 1
            index += self.overlap
        weights[:, :, :, self.overlap:] /= 2  # 如果需要取平均
        return x.unsqueeze(2) * weights


class ECABlock(nn.Module):
    def __init__(self, k_size=3):
        super(ECABlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，将空间维度变为 1x1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,bias=False,padding=1,)  # 1D卷积
        self.sigmoid = nn.Sigmoid()  # 输出通道权重，范围在 0 到 1 之间
    def forward(self, x1):
        batch_size, channels, _, _ = x1.size()
        # 处理 x1
        y1 = self.global_avg_pool(x1).view(batch_size, 1, channels)  # 调整为 (batch_size, 1, channels) 以适应1D卷积
        y1 = self.conv(y1)  # 1D卷积
        y1 = self.sigmoid(y1).view(batch_size, channels, 1, 1)  # 应用Sigmoid激活并调整回 (batch_size, channels, 1, 1)
        y1 = x1 * y1.expand_as(x1)  # 权重应用于输入 x1
        return y1

class EEGNet1(nn.Module):
    def __init__(self, eeg_chans=22, samples=1000, dropoutRate=0.25, kerSize=32, kerSize_Tem=32, F1=16, D=2, bias=False, n_classes=4):
        super(EEGNet1, self).__init__()
        F2 = F1*D

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=F1, 
                kernel_size=(1, kerSize), 
                stride=1,
                padding='same',
                bias=bias
            ), 
            nn.BatchNorm2d(num_features=F1) 
        )

        self.depthwiseConv = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=F1, 
                out_channels=F1*D,
                kernel_size=(eeg_chans, 1),
                groups=F1,
                bias=bias, 
                max_norm=1.
            ), 
            nn.BatchNorm2d(num_features=F1*D),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1,4),
                stride=(1,4)
            ),
            nn.Dropout(p=dropoutRate)
        )

        self.seqarableConv = nn.Sequential(
            ECABlock(),
            TimeSeriesModel(
                window_size=50, 
                num_pipelines=32, 
                num_time_steps=250, 
                reduction=50,
                overlap=25),
            nn.Conv2d(
                in_channels=F2, 
                out_channels=F2,
                kernel_size=(1,kerSize_Tem),
                stride=1,
                padding='same',
                groups=F2,
                bias=bias
            ),
            nn.Conv2d(
                in_channels=F2,
                out_channels=F2,
                kernel_size=(1,1),
                stride=1,
                bias=bias
            ),     
            nn.BatchNorm2d(num_features=F2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1,8),
                stride=(1,8)
            ),
            nn.AvgPool2d(
                kernel_size=(1,2),
                stride=(1,2)
            ),
            nn.Dropout(p=dropoutRate)
        )
        self.class_head = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(
                in_features=480,
                out_features=n_classes,
                max_norm=.5
            ),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)
        x = self.block1(x)
        x = self.depthwiseConv(x)
        x = self.seqarableConv(x)
        x = self.class_head(x)
        return x
    
model = EEGNet1(n_classes=4,eeg_chans=22)

print(model)