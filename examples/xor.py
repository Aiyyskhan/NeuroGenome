from cgi import test
import numpy as np
import neurogenome as ng

# 2-input XOR inputs and expected outputs.
xor_inputs = [
	[0.0, 0.0], 
	[0.0, 1.0], 
	[1.0, 0.0], 
	[1.0, 1.0]
]
xor_targets = [
	0.0, 
	1.0, 
	1.0, 
	0.0
]

# gene localization scheme
schema_0 = [
	[
		["i0"]
	],
	[
		["h0"]
	],
	[
		["o0"]
	]
]

# hyperparameters
SETTINGS = {
	"population size": 70,
	"number of leaders": 5,
	"select by": "min",
	"number of input nodes per gene": 2,
	"number of hidden nodes per gene": 7,
	"number of output nodes per gene": 1,
	"schema": schema_0,
}

# VAL_SEQ = [-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0]
# VAL_SEQ = [-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0]
# VAL_SEQ = [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]
VAL_SEQ = [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]

MAX_EPOCH = 500
SUCCESS = 0.001

genome = ng.builders.genome_builder(SETTINGS, VAL_SEQ)

w_matrix_list = ng.builders.neuro_builder(genome)

# def sigmoid(x):
# 	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(x, 0)

def nn(data, wMs):
	for wM in wMs:
		# data = sigmoid(np.matmul(data, wM))
		data = np.tanh(relu(np.matmul(data, wM)))
	return data

for gen in range(MAX_EPOCH):

	in_data = np.array(xor_inputs)[None,:,:]

	resp = nn(in_data, w_matrix_list)

	err_arr = np.sum(((np.array(xor_targets)[:,None] - resp)**2), axis=1).ravel()

	if gen % 10 == 0:
		print(f"Gen: {gen} - Best result {err_arr.min()}")

	if err_arr.min() <= SUCCESS:
		break

	lead_genome = ng.evolution.selection(genome, err_arr)
	genome = ng.evolution.crossover(lead_genome)
	ng.evolution.mutation(genome)
	w_matrix_list = ng.builders.neuro_builder(genome)

print("\nTrain end")

champion_index = np.argsort(np.array(err_arr))[0]

champ_w = [tensor[champion_index] for tensor in w_matrix_list]

while True:
	test_arr = input("Enter test data (or 'stop'): ")
	if test_arr == "stop":
		break
	
	resp = nn(np.array(test_arr.split()).astype("int"), champ_w)
	print(f"Response: {resp}")


