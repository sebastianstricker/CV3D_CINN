import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm

def one_hot_sequenceINN(labels, nr_conditions, out=None):
    '''
    Framework is currently bugged. For SequenceINN, this needs to be passed as a tuple.
    '''

    out = one_hot_GraphINN(labels, nr_conditions, out)
    return (out,)

"""
Function from: https://github.com/VLL-HD/conditional_INNs/blob/master/mnist_minimal_example
"""
def one_hot_GraphINN(labels, nr_conditions, out=None):
    '''
    Convert LongTensor labels (contains labels 0-9), to a one hot vector.
    Can be done in-place using the out-argument (faster, re-use of GPU memory)
    '''
    if out is None:
        out = torch.zeros(labels.shape[0], nr_conditions).to(labels.device)
    else:
        out.zeros_()

    out.scatter_(dim=1, index=labels.view(-1,1), value=1.)
    return out

"""
Function adapted from: https://github.com/VLL-HD/conditional_INNs/blob/master/mnist_minimal_example
"""
class Toy_cINN(nn.Module):
    '''cINN for class-conditional MNISt generation'''
    def __init__(self, dims, nr_conditions):
        super().__init__()

        self.cinn = self.build_inn(dims, nr_conditions)
        self.dims = dims
        self.nr_conditions = nr_conditions

    def build_inn(self, dims, nr_conditions):
        cond_shape = (nr_conditions,)

        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, 256), nn.ReLU(),
                                nn.Linear(256,  256), nn.ReLU(),
                                nn.Linear(256,  c_out))

        inn = Ff.SequenceINN(dims)

        for _ in range(8):
            inn.append(Fm.AllInOneBlock, cond=0, cond_shape=cond_shape, subnet_constructor=subnet_fc, permute_soft=False)

        return inn

    def forward(self, x, l, jac=True):
        z, jac = self.cinn(x, c=one_hot_sequenceINN(l, self.nr_conditions), jac=jac)
        return z, jac

    def reverse_sample(self, z, l):
        return self.cinn(z, c=one_hot_sequenceINN(l, self.nr_conditions), rev=True)
