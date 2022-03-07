import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm

ndim_total = 28 * 28

def one_hot(labels, out=None):
    '''
    Convert LongTensor labels (contains labels 0-9), to a one hot vector.
    Can be done in-place using the out-argument (faster, re-use of GPU memory)
    '''
    if out is None:
        out = torch.zeros(labels.shape[0], 10).to(labels.device)
    else:
        out.zeros_()

    out.scatter_(dim=1, index=labels.view(-1,1), value=1.)
    return out


class FashionMNIST_cINN1(nn.Module):
    '''cINN for class-conditional FashionMNIST generation'''
    def __init__(self, lr):
        super().__init__()

        self.cinn = self.build_inn()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self):

        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, ch_out))

        cond = Ff.ConditionNode(10)
        nodes = [Ff.InputNode(1, 28, 28)]

        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(20):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, l):
        z, jac = self.cinn(x, c=one_hot(l))
        return z, jac

    def reverse_sample(self, z, l):
        return self.cinn(z, c=one_hot(l), rev=True)


class FashionMNIST_cINN2(nn.Module):
    '''cINN for class-conditional FashionMNIST generation'''
    def __init__(self, lr):
        super().__init__()

        self.cinn = self.build_inn()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self):

        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 512),
                                 nn.LeakyReLU(),
                                 nn.Linear(512, ch_out))

        cond = Ff.ConditionNode(10)
        nodes = [Ff.InputNode(1, 28, 28)]

        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(20):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, l):
        z, jac = self.cinn(x, c=one_hot(l))
        return z, jac

    def reverse_sample(self, z, l):
        return self.cinn(z, c=one_hot(l), rev=True)


class FashionMNIST_cINN3(nn.Module):
    '''cINN for class-conditional FashionMNIST generation'''
    def __init__(self, lr):
        super().__init__()

        self.cinn = self.build_inn()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self):

        def conv_subnet(ch_hidden, kernel):
            pad = kernel // 2
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

        def fcn_subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 512),
                                 nn.LeakyReLU(),
                                 nn.Linear(512, ch_out))

        cond = Ff.ConditionNode(10)
        nodes = [Ff.InputNode(1, 28, 28)]

        # Downsample, because we can't split GS image (1 channel) into two parts
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

        subnet = conv_subnet(32, 3)
        for k in range(2):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0}))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))

        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

        subnet = conv_subnet(64, 3)
        for k in range(4):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0}))

        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(10):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':fcn_subnet, 'clamp':1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, l):
        z, jac = self.cinn(x, c=one_hot(l))
        return z, jac

    def reverse_sample(self, z, l):
        return self.cinn(z, c=one_hot(l), rev=True)
