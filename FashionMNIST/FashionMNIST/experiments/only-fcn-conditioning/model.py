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
    out = torch.zeros(y.shape[0], 10).to(y.device)
    out.scatter_(dim=1, index=y.view(-1, 1), value=1.)
    return out


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

        cond = Ff.ConditionNode(10)

        nodes = [Ff.InputNode(*IM_DIMS)]

        split_nodes = []

        # Stage 1
        subnet = sub_conv(32, 3)
        for k in range(5):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {"subnet_constructor": subnet, "clamp": 1.0}))

        nodes.append(Ff.Node(nodes[-1], Fm.Split,
                             {"section_sizes": [1, 1], "dim": 0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))

        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {"rebalance": 0.5}))

        # Stage 2
        subnet = sub_conv(64, 3)
        for k in range(5):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {"subnet_constructor": subnet, "clamp": 1.0}))

        nodes.append(Ff.Node(nodes[-1], Fm.Split,
                             {"section_sizes": [2, 2], "dim": 0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))

        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, {"rebalance": 0.5}))

        # Stage 3
        subnet = sub_conv(128, 3)
        for k in range(10):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {"subnet_constructor": subnet, "clamp": 1.0}))

        nodes.append(Ff.Node(nodes[-1], Fm.Split,
                             {"section_sizes": [4, 4], "dim": 0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
        nodes.append(Ff.Node(nodes[-1].out0, Fm.Flatten, {}))

        # Stage 4
        subnet = sub_fc(512)
        for k in range(15):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {"subnet_constructor": subnet, "clamp": 1.0},
                                 conditions=cond))

        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat, {"dim": 0}))
        nodes.append(Ff.OutputNode(nodes[-1]))

        return Ff.ReversibleGraphNet(nodes + split_nodes + [cond], verbose=False)

    def forward(self, x, y):
        z, jac = self.cinn(x, c=one_hot_encode(y))
        return z, jac

    def reverse_sample(self, z, y):
        return self.cinn(z, c=one_hot_encode(y), rev=True)
