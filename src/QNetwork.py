from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.nn import LeakyReLU

class QNetwork(nn.Block):
    def __init__(self, n_actions, vae_absent, **kwargs):
        super(QNetwork, self).__init__(**kwargs)
        self.vae_absent = vae_absent
        if(vae_absent):
            self.conv1 = nn.Conv2D(channels=20, kernel_size=10, layout='NCHW')
            self.conv2 = nn.Conv2D(channels=30, kernel_size=5, layout='NCHW')
        self.leaky = nn.LeakyReLU(0.2)
        self.dense1 = nn.Dense(1024)
        self.dense2 = nn.Dense(512)
        self.dense3 = nn.Dense(n_actions)
        self.flatten = nn.Flatten()

    def forward(self, x):
        if(self.vae_absent):
            x = self.conv1(x)
            x = self.leaky(x)
            x = nd.Pooling(x, kernel=(3,3), stride=(2,2),layout='NHWC', pool_type='lp', p_value=2)
            x = self.conv2(x)
            x = self.leaky(x)
            x = nd.Pooling(x, kernel=(3,3), stride=(2,2),layout='NHWC', pool_type='lp', p_value=2)
            x = self.flatten(x)
        x = self.dense1(x)
        x = self.leaky(x)
        x = self.dense2(x)
        x = self.leaky(x)
        y = self.dense3(x)
        return y