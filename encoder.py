import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    '''像素空间编码器'''
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,output_logits=False):
        '''
        param output_logits: 在实际使用时均为True
        '''
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        # 卷积层，没有激活函数？没有归一化层
        # 在forward中加入了relu
        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        # 接着就是线性层 + 层归一化
        out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers] 
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        # outputs存储中间结果，比如obs，conv1，conv2，fc，ln，tanh
        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        # 对环境进行归一化预处理
        obs = obs / 255.
        self.outputs['obs'] = obs

        # 进行卷积操作
        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        # 返回编码特征向量
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        '''
        params obs: 环境编码器
        params detach: 是否需要detach todo 作用 ，在eval时为false，没必要断开，因为不计算梯度
        params output_logits: 是否需要输出logits，决定输出结果是否经过tanh函数
        '''
        h = self.forward_conv(obs)

        if detach:
            # detach原因：在CURL中的应用场景Actor网络更新时不需要更新编码器
            # 目标编码器计算时需要固定参数
            # 防止编码器被过度更新
            # 稳定训练过程
            # 允许不同组件独立学习
            # todo 那为啥组合网络，一边仅优化actor 线性层 一边优化所有
            # todo 实现actor和critic合并
            h = h.detach() # 断开梯度传播

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            '''
            todo 实际使用时都是true，可能一般不使用
            直接输出层归一化(LayerNorm)后的值
            值域不受限制
            适用于需要原始特征分布的情况
            在CURL中用于对比学习
            '''
            out = h_norm
        else:
            '''
            todo
            对层归一化后的值进行tanh激活
            将值压缩到[-1, 1]区间
            适用于需要有界输出的情况
            可以防止特征值过大
            '''
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    '''
    线性空间编码器，啥也不做
    '''
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )
