# -*- coding: utf-8 -*-

# @Author: xyq



import numpy as np

class Node:
    def __init__(self, inputs=[]):
        self.inputs = inputs
        self.outputs = []

        for n in self.inputs:
            n.outputs.append(self)
        self.value = None
        self.gradients = {}
    def forward(self):
        raise NotImplemented
    def backward(self):
        raise NotImplemented


class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self,value=None):
        self.value = value

    def backward(self):
        for n in self.outputs:
            self.gradients[self] = n.gradients[self] * 1


class Add(Node):
    def __init__(self, nodes):
        Node.__init__(self, nodes)

    def forward(self):
        self.value = sum([n.value for n in self.inputs])


class Linear(Node):
    def __init__(self, nodes, weights, bias):
        Node.__init__(self, [nodes,weights,bias])


    def forward(self):
        inputs = self.inputs[0].value
        weights = self.inputs[1].value
        bias = self.inputs[2].value

        self.value = np.dot(inputs, weights) + bias

    def backward(self):
        # self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.inputs[0]] = np.dot(grad_cost, self.inputs[1].value.T)
            self.gradients[self.inputs[1]] = np.dot(self.inputs[0].value.T, grad_cost)
            self.gradients[self.inputs[2]] = np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _issigmod(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def forward(self):
        self.x = self.inputs[0].value
        self.value = self._issigmod(self.x)

    def backward(self):
        self.partial = self._issigmod(self.x) * (1 - self._issigmod(self.x))
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self.inputs[0]] = grad_cost * self.partial

class MSE(Node):
    def __init__(self, y_true, y_hat):
        Node.__init__(self,[y_true,y_hat])

    def forward(self):
        y_true = self.inputs[0].value.reshape(-1,1)
        y_hat = self.inputs[1].value.reshape(-1,1)
        assert (y_true.shape == y_hat.shape)

        self.m = self.inputs[0].value.shape[0]
        self.diff = y_true - y_hat
        self.value = np.mean(self.diff ** 2)

    def backward(self):
        self.gradients[self.inputs[0]] = (2 / self.m) * self.diff
        self.gradients[self.inputs[1]] = (-2 / self.m) * self.diff

def forward_and_backward(outputnode, graph):
    for n in graph:
        n.forward()
    for n in graph[::-1]:
        n.backward()

def topological_sort(feed_dict):
    input_nodes = [n for n in feed_dict.keys()]
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outputs:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outputs:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value += -1 * learning_rate * t.gradents[t]






