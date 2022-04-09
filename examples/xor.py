import numpy as np
import neurogenome as ng

# 2-input XOR inputs and expected outputs.
xor_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
xor_outputs = [0.0, 1.0, 1.0, 0.0]

# gene localization scheme
schema_0 = [
	[
		["i0","i1"]
	],
	[
		["h0"],
		["h1"]
	],
	[
		["o0"]
	]
]

# hyperparameters
SETTINGS = {
	"population size": 20,
	"number of leaders": 5,
	"select by": "min",
	"number of input nodes per gene": 2,
	"number of hidden nodes per gene": 4,
	"number of output nodes per gene": 1,
	"schema": schema_0,
}

MAX_EPOCH = 100
SUCCESS = 0.001

genome = ng

nn_matrix_list = ng.builders.neuro_builder(genome)



def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def nn(data, wM):
	return sigmoid(np.dot(data, wM))

for gen in range(MAX_EPOCH):
	for idx, in_data in enumerate(xor_inputs):
		pass
