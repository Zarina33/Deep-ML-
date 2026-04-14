"""
Implementing Basic Autograd Operations
Medium
Deep Learning


Inspired by Andrej Karpathy's micrograd â check out his excellent video: https://youtu.be/VMj-3S1tku0
Implement a Value class that wraps a scalar number and supports automatic differentiation. Your class should:
Support + and * operations that work with other Value objects or plain numbers, returning a new Value with the correct result.
Support a relu() method that applies the ReLU activation function (outputs the value if positive, 0 otherwise).
Support a backward() method that computes gradients for all upstream Value objects in the computation graph using backpropagation.
Each operation should track its inputs and know how to locally compute its gradient contribution. The backward() method should process nodes in the correct order so that gradients accumulate properly from output back to inputs.
Hints:
Each operation creates a new Value that remembers which values produced it (_children).
Think about what order nodes need to be processed during the backward pass.
Gradients should be accumulated (+=), not replaced.
Example:
Input:
a = Value(2)
        b = Value(-3)
        c = Value(10)
        d = a + b * c
        e = d.relu()
        e.backward()
        print(a, b, c, d, e)
Output:
Value(data=2, grad=0) Value(data=-3, grad=0) Value(data=10, grad=0)
Reasoning:
The output reflects the forward computation and gradients after backpropagation. The ReLU on 'd' zeros out its output and gradient due to the negative data value.
"""

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad  += 1 * out.grad   # d(x+y)/dx = 1
            other.grad += 1 * out.grad   # d(x+y)/dy = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += other.data * out.grad  # d(x*y)/dx = y
            other.grad += self.data * out.grad   # d(x*y)/dy = x
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad  # 1 если >0, иначе 0
        out._backward = _backward
        return out

    def backward(self):
        # 1. строим топологический порядок (от выхода к входам)
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)  # сначала дети, потом родитель

        build_topo(self)

        # 2. grad выходного узла = 1 (dL/dL = 1)
        self.grad = 1

        # 3. идём в обратном порядке и вызываем _backward
        for v in reversed(topo):
            v._backward()