from functools import partial
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class Reward:
    def __init__(self, property, reward, weight=1.0, preprocess=None, batched_mode=False):
        self.property = property
        self.reward = reward
        self.weight = weight
        self.preprocess = preprocess
        self.batched_mode = batched_mode

    def __call__(self, input):
        if self.preprocess:
            input = self.preprocess(input)
        property = self.property(input)
        reward = self.reward(property)
        if self.batched_mode:
            reward = [self.weight * r for r in reward]
            return list(zip(reward, property))
        else:
            reward = self.weight * reward
            return reward, property


def identity(x):
    return x


def ReLU(x):
    return max(x, 0)


def HSF(x):
    return float(x > 0)


class OutOfRange:
    def __init__(self, lower=None, upper=None, hard=True):
        self.lower = lower
        self.upper = upper
        self.func = HSF if hard else ReLU

    def __call__(self, x):
        y, u, l, f = 0, self.upper, self.lower, self.func
        if u is not None:
            y += f(x - u)
        if l is not None:
            y += f(l - x)
        return y


class PatternFilter:
    def __init__(self, patterns):
        self.structures = list(filter(None, map(Chem.MolFromSmarts, patterns)))

    def __call__(self, molecule):
        return int(any(molecule.HasSubstructMatch(struct) for struct in self.structures))


def MolLogP(m):
    return rdMolDescriptors.CalcCrippenDescriptors(m)[0]
