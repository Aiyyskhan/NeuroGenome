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
xor_outputs = [
	0.0, 
	1.0, 
	1.0, 
	0.0
]

# gene localization scheme
schema_0 = [
	[
		["i0", "i1", "i2", "i3"]
	],
	[
		["h0", "h1", "h2", "h12"],
		["h3", "h4", "h5", "h13"],
		["h6", "h7", "h8", "h14"],
		["h9", "h10", "h11", "h15"]
	],
	[
		["o0"],
		["o1"],
		["o2"],
		["o3"]
	]
]

# schema_1 = [
# 	[
# 		["i0", "i1", "i2", "i3", "i4"]
# 	],
# 	[
# 		["h0", "h1", "h2"],
# 		["h3", "h4", "h5"],
# 		["h6", "h7", "h8"],
# 		["h9", "h10", "h11"],
# 		["h12", "h13", "h14"]
# 	],
# 	[
# 		["o0"],
# 		["o1"],
# 		["o2"]
# 	]
# ]

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

VAL_SEQ = [-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

MAX_EPOCH = 1000
SUCCESS = 0.001

genome = ng.builders.genome_builder(SETTINGS, VAL_SEQ)

w_matrix_list = ng.builders.neuro_builder(genome)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def nn(data, wMs):
	for wM in wMs:
		data = sigmoid(np.matmul(data, wM))
	return data

for gen in range(MAX_EPOCH):

	in_data = np.array(xor_inputs)[None,:,:]

	resp = nn(in_data, w_matrix_list)

	err_arr = np.sum(((np.array(xor_outputs)[:,None] - resp)**2), axis=1).ravel()

	if gen % 10 == 0:
		print(f"Gen: {gen} - Best result {err_arr.min()}")
		# print(f"iGenes[0, 0]: \n{genome.iGenes[0, 0]}")

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


