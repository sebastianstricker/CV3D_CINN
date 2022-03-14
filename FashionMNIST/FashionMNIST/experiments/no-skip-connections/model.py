import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from FashionMNIST.data import IM_DIMS

"""
Code adapted from: https://github.com/VLL-HD/conditional_INNs/blob/master/mnist_minimal_example/model.py
"""


def one_hot_encode(y):
    """
    We create one-hot-encodings for 4 stages:
    Stage 1: (N, 1, 28, 28), where N is the length of y array
             and the feature map at index i is full of values
             y[i] + 1 / 10. We assume 10 possible class values
             of FashionMNIST from 0 to 9.
    Stage 2: (N, 1, 14, 14), analogous to stage 1
    Stage 3: (N, 1, 7, 7), analogous to stage 1
    Stage 4: (N, 10), a generic OHC, where the vector at index i
                      has all elements equals to 0, except for
                      the index at position y[i].
    """

    out_stage1 = y.reshape(-1, 1, 1, 1).repeat(1, 1, 28, 28).to(y.device)
    out_stage1 = ((out_stage1 + 1) / 10).float()

    out_stage2 = y.reshape(-1, 1, 1, 1).repeat(1, 1, 14, 14).to(y.device)
    out_stage2 = ((out_stage2 + 1) / 10).float()

    out_stage3 = y.reshape(-1, 1, 1, 1).repeat(1, 1, 7, 7).to(y.device)
    out_stage3 = ((out_stage3 + 1) / 10).float()

    out_stage4 = torch.zeros(y.shape[0], 10).to(y.device)
    out_stage4.scatter_(dim=1, index=y.view(-1, 1), value=1.)

    return [out_stage1, out_stage2, out_stage3, out_stage4]


class FashionMNIST_cINN(nn.Module):

    def __init__(self, lr):
        super().__init__()

        self.cinn = self.build_inn()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self):

        def sub_conv(ch_hidden, kernel):
            pad = kernel // 2
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                            nn.ReLU(),
                                            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

        def sub_fc(ch_hidden):
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Linear(ch_in, ch_hidden),
                                            nn.ReLU(),
                                            nn.Linear(ch_hidden, ch_out))

        conditions = [Ff.ConditionNode(1, 28, 28),
                      Ff.ConditionNode(1, 14, 14),
                      Ff.ConditionNode(1, 7, 7),
                      Ff.ConditionNode(10)]

        nodes = [Ff.InputNode(*IM_DIMS)]

        # Stage 1
        subnet = sub_conv(32, 3)
        for k in range(5):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {"subnet_constructor": subnet, "clamp": 1.0},
                                 conditions=conditions[0]))

        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {"rebalance": 0.5}))

        # Stage 2
        subnet = sub_conv(64, 3)
        for k in range(5):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {"subnet_constructor": subnet, "clamp": 1.0},
                                 conditions=conditions[1]))

        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {"rebalance": 0.5}))

        # Stage 3
        subnet = sub_conv(128, 3)
        for k in range(10):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {"subnet_constructor": subnet, "clamp": 1.0},
                                 conditions=conditions[2]))

        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        # Stage 4
        subnet = sub_fc(512)
        for k in range(15):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {"subnet_constructor": subnet, "clamp": 1.0},
                                 conditions=conditions[3]))

        nodes.append(Ff.OutputNode(nodes[-1]))

        return Ff.ReversibleGraphNet(nodes + conditions, verbose=False)

    def forward(self, x, y):
        z, jac = self.cinn(x, c=one_hot_encode(y))
        return z, jac

    def reverse_sample(self, z, y):
        return self.cinn(z, c=one_hot_encode(y), rev=True)
