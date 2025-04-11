import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf. todo
    函数实现了 SAC 算法中的 tanh 压缩操作，其作用是将无界的动作值压缩到有界区间。让我分析一下这个函数
    """
    # 压缩到[-1, 1]区间
    mu = torch.tanh(mu)
    # 对采样的动作进行压缩
    if pi is not None:
        pi = torch.tanh(pi)
    # 对数概率进行修正
    if log_pi is not None:
        # 根据变量替换法则修正对数概率
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super().__init__()

        # 创建编码器
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 创建header了，线性空间，有激活函数
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        # 存储中间值，有均值、方差、
        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        '''
        obs: 观察值

        环境的观察输入(状态)
        通常是图像数据
        compute_pi=True: 是否计算策略动作,

        True: 计算随机策略动作(用于训练和探索)，在eval时为True
        False: 只返回确定性动作(用于评估)
        compute_log_pi=True: 是否计算对数概率

        True: 计算动作的对数概率(用于计算策略损失)
        False: 不计算对数概率,在eval时为false
        detach_encoder=False: 是否分离编码器梯度

        True: 停止梯度传播到编码器
        False: 允许梯度传播到编码器 在eval时为false
        '''
        # 环境编码器得到的时均值和方差
        obs = self.encoder(obs, detach=detach_encoder)

        # 拆分预测值分别为均值和方差
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        # 限制方差的大小，可能是为防止抖动把 todo
        '''
        是的，这段代码是为了限制方差的大小。让我解析一下这段代码：

        ```python
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)  # 首先压缩到(-1,1)范围
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        ```

        这个限制操作分两步：

        1. **首先使用tanh函数压缩**:
        - `torch.tanh(log_std)` 将输入压缩到(-1,1)范围
        - tanh函数可以防止数值过大或过小

        2. **然后线性映射到指定范围**:
        - `log_std_min` 默认为-10
        - `log_std_max` 默认为2
        - `(log_std + 1)` 将范围调整到(0,2)
        - `0.5 * (self.log_std_max - self.log_std_min)` 计算目标范围的一半
        - 最终将值映射到[log_std_min, log_std_max]范围内

        限制方差的目的：
        1. 防止动作探索过度发散
        2. 保持策略的稳定性
        3. 控制随机性的程度
        4. 避免数值不稳定

        这样做可以让策略网络输出的动作分布保持在一个合理的范围内，既不会太确定(方差太小)也不会太随机(方差太大)。
        '''
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            # 对动作进行采样，使用的是高斯分布
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        # 在eval时不计算对数概率
        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        # 返回预测的均值、采样的动作、对数概率和标准差
        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()

        # 环境编码器
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        # 两个线性编码器，作用是啥？对比我现在已有的SAC看看
        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class CURL(nn.Module):
    """
    CURL 
    """

    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type="continuous"):
        '''
        obs_shape: 观察空间的形状

        用于处理环境状态的输入维度
        通常是图像数据,形状为 (channels, height, width)
        z_dim: 隐空间维度

        编码器输出的特征向量维度
        决定了表征学习的维度大小
        在这个实现中等于 encoder_feature_dim (默认50)
        batch_size: 批量大小

        训练时每批数据的样本数
        用于计算对比损失时的批次维度
        critic: Critic 网络实例

        包含主要的编码器网络
        CURL 会使用 critic.encoder 作为其编码器
        critic_target: 目标 Critic 网络实例

        包含目标编码器网络
        CURL 使用 critic_target.encoder 作为目标编码器
        用于计算对比学习的正样本编码
        output_type: 输出类型

        默认为 "continuous"
        指定输出空间类型
        这个实现中仅支持连续动作空间
        '''

        super(CURL, self).__init__()
        self.batch_size = batch_size

        # 评价网络的环境编码器
        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder 

        # todo W的参数
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        # 输出的动作空间类型，todo 是否支持离散动作空间
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        todo 这里计算了CURL损失，了解原理
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

class CurlSacAgent(object):
    """CURL representation learning with SAC."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        curl_latent_dim=128
    ):
        '''
        # 基础参数
        obs_shape             # 观察空间的形状，用于确定输入维度
        action_shape         # 动作空间的形状，用于确定输出维度
        device              # 运行设备(CPU/GPU)
        hidden_dim=256      # 隐藏层维度
        discount=0.99       # 折扣因子，用于计算未来奖励的衰减率
        
        # 温度参数
        init_temperature=0.01  # SAC算法中温度参数α的初始值
        alpha_lr=1e-3         # 温度参数α的学习率
        alpha_beta=0.9        # 温度参数优化器的beta参数(Adam优化器)
        
        # Actor网络参数
        actor_lr=1e-3         # Actor网络的学习率
        actor_beta=0.9        # Actor优化器的beta参数
        actor_log_std_min=-10 # 动作分布标准差的最小对数值
        actor_log_std_max=2   # 动作分布标准差的最大对数值
        actor_update_freq=2   # Actor网络更新频率
        
        # Critic网络参数
        critic_lr=1e-3        # Critic网络的学习率
        critic_beta=0.9       # Critic优化器的beta参数
        critic_tau=0.005      # 目标网络软更新系数
        critic_target_update_freq=2  # Critic目标网络更新频率
        
        # 编码器参数
        encoder_type='pixel'   # 编码器类型，可以是'pixel'或其他
        encoder_feature_dim=50 # 编码器输出特征维度
        encoder_lr=1e-3       # 编码器学习率
        encoder_tau=0.005     # 编码器目标网络软更新系数
        num_layers=4          # 编码器网络层数
        num_filters=32        # 编码器卷积层滤波器数量
        
        # 其他参数
        cpc_update_freq=1     # CPC(对比预测编码)更新频率
        log_interval=100      # 日志记录间隔
        detach_encoder=False  # 是否分离编码器梯度
        curl_latent_dim=128   # CURL隐空间维度
        '''
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type

        # 动作预测
        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        # 评价预测
        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        # 目标评价网络
        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        # 这里是将动作空间的编码器和评价空间的编码器进行同步
        # todo 会不会对中断持续训练产生影响？
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        
        # todo 这里是如何起作用的
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        # 如果编码器类型是像素
        if self.encoder_type == 'pixel':
            # create CURL encoder (the 128 batch size is probably unnecessary)
            # 则创建CURL编码器
            self.CURL = CURL(obs_shape, encoder_feature_dim,
                        self.curl_latent_dim, self.critic,self.critic_target, output_type='continuous').to(self.device)

            # optimizer for critic encoder for reconstruction loss
            # 单独对评价网络的环境编码器进行优化
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            # 创建curl的优化器
            self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    # 设置为训练模式
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixel':
            self.CURL.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        '''
        选择动作会不进行观察空间的裁剪
        '''
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            # 返回预测的动作，不使用随机动作采样
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        '''
        动作随机采集
        # 随机采样动作会对观察进行裁剪
        '''
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        '''
        更新评价网络
        '''
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            # 目标评价网络通过下一个状态和下一个状态的动作评价q值
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            # 从中选择较小的q值
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            # 计算当前obs的q值
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        # 评价网络计算
        # detach_encoder这里False，也就是不分离梯度
        # 得到预测的q值
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
        # 两个q值需要都接近真实的q值
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        # 优化
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        '''
        param obs: 观察
        param L: 日志
        param step： 步数
        '''
        # detach encoder, so we don't update it with the actor loss
        # 在训练actor时分离梯度，不更新动作的编码器
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        # 得到预测的小梯度
        actor_Q = torch.min(actor_Q1, actor_Q2)
        # todo 预测的对数概率和评价网络的Q值要接近？
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            # 记录日志损失、熵
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        #得到熵 todo 这里的熵是如何计算的
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        # 优化动作网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 记录日志，没有其他操作
        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        # todo 这里的损失是什么？
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs, L, step):
        
        # 通过CURL计算两个不同随机采样的观察之间的差异
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)
        
        logits = self.CURL.compute_logits(z_a, z_pos)
        # 也是和另一个CURL是一个顺序的labels todo 为啥
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
        
        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/curl_loss', loss, step)


    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':
            obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            # 隔指定的轮数更新actor
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            # 同步评价网络和目标评价网络的参数
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
        
        # 看起来像素空间有额外的处理
        if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
            # 观察空间的采样数据cpc_kwargs["obs_anchor"]
            # 观察空间的采样数据cpc_kwargs["obs_pos"]，两者之间的差异采用的
            # 是不同的随机裁剪采样
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            self.update_cpc(obs_anchor, obs_pos,cpc_kwargs, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def save_curl(self, model_dir, step):
        torch.save(
            self.CURL.state_dict(), '%s/curl_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
 