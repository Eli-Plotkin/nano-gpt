import math
import random

'''
1) Initialize nodes, data points, forward and backward pass to get one full pass
2) Convert to neurons with weights and constant formula
3) Add Layers
'''



class Value:
    def __init__(self, data, children, label):
        self.data = data
        self.grad = 0
        self.label = label
        self._backward = lambda: None
        self.children = tuple(children)

    def __repr__(self):
        return f"{self.data}"
    
    def __add__(self, other):
        new = Value(self.data + other.data, (self, other), '+') \
        if isinstance(other, Value) else Value(self.data + other, (self, ), '+')
        

        def _backward():
            self.grad += new.grad
            if isinstance(other, Value): other.grad += new.grad
            print(f"Updating {self.label} grad to {self.grad}")
            if isinstance(other, Value): print(f"Updating {other.label} grad to {other.grad}")


        new._backward = _backward

        return new
    
    def __mul__(self, other):
        new = Value(self.data * other.data, (self, other), '*') \
        if isinstance(other, Value) else Value(self.data * other, (self, ), '*')
        
        def _backward():
            self.grad += other.data * new.grad
            if isinstance(other, Value): other.grad += self.data * new.grad 
            print(f"Updating {self.label} grad to {self.grad}")
            if isinstance(other, Value): print(f"Updating {other.label} grad to {other.grad}")

        
        new._backward = _backward

        return new

    def tanh(self):
        x = self.data
        val = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        new = Value(val, {self, }, 'tanh')

        def _backward():
            self.grad += (1 - val**2) * new.grad
            print(f"Updating {self.label} grad to {self.grad}")

        new._backward = _backward

        return new
    

    def backward(self):
        self.grad = 1.0
        
        top_list = []
        top_list.append(self)
        visited = set()
        ind = 0 

        while True:
            curr = top_list[ind]
            for child in curr.children:
                if child not in visited:
                    top_list.append(child)
            visited.add(curr)

            ind += 1
            if ind >= len(top_list):
                break

        print(top_list)

        for item in top_list:
            item._backward()




class Neuron:
    def __init__(self, num_inputs):
        self.weights = [Value(random.uniform(-1, 1), (), f"w{i}") for i in len(num_inputs)]
        self.bias = Value(random.uniform(-1, 1), (), "b")

    def __call__(self, inputs):
        pairs = zip(self.weights, inputs)

        total = self.bias

        for w, x in pairs:
            total += w * x

    
        return total.tanh()
        

class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in len(num_outputs)]

    def __call__(self, inputs):
        out = []
        for neuron in self.neurons:
            out.append(neuron(inputs))
            
        return out
    
class CNN:
    def __init__(self, num_inputs, num_out_per_layer):
        in_size = [num_inputs].extend(num_out_per_layer)
        self.layers = [Layer(in_size[i], num_out_per_layer[i]) for i in len(num_out_per_layer)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



curr_cnn = CNN(6, [8, 8, 1])

training = [[0.4, 0.8, 0.223, -0.9, 3.67, 9.33],
            [0.23, 0.865, 2.346, 8.321, 985.1, 1.2],
            [34.5, 29.9, -34.5, -3.0, 0.113, 0.1]]

targets = [0.3, -3.1, 0.94]

predictions = [curr_cnn(x) for x in training]

loss = sum([(targ - pred)**2 for targ, pred in zip(targets, predictions)])
print(f"Loss: {loss}")
loss.backward
