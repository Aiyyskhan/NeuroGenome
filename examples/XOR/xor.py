import numpy as np
import neurogenome as ng

# 2-input XOR inputs and expected outputs (targets).
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
	"population size": 50,
	"number of leaders": 5,
	"select by": "min",
	"number of input nodes per gene": 2,
	"number of hidden nodes per gene": 3,
	"number of output nodes per gene": 1,
	"schema": schema_0,
}

# VAL_SEQ = [-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0]
# VAL_SEQ = [-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0]
VAL_SEQ = [-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.]

MAX_EPOCH = 500

# expected accuracy
SUCCESS = 0.001

# activation functions
# def sigmoid(x):
# 	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(x, 0)

# neural network
def nn(data, wMs):
	for wM in wMs:
		# data = sigmoid(np.matmul(data, wM))
		data = np.tanh(relu(np.matmul(data, wM)))
	return data

# genome creation
genome = ng.builders.genome_builder(SETTINGS, VAL_SEQ)
# generation of weight matrices from the genome
w_matrix_list = ng.builders.neuro_builder(genome)

print("\n*** Evolution started ***\n")

# evolution loop
for gen in range(MAX_EPOCH):
	# add axis
	in_data = np.array(xor_inputs)[None,:,:]
	# neural network operation
	resp = nn(in_data, w_matrix_list)
	# error calculation (calculation of L2 norm)
	err_arr = np.sqrt(np.sum(((np.array(xor_targets)[:,None] - resp)**2), axis=1)).ravel()

	# if gen % 10 == 0:
	# 	print(f"Gen: {gen} - Best result {err_arr.min()}")
	print(f"Gen: {gen} - Best result {err_arr.min()}")

	if err_arr.min() < SUCCESS:
		break
	
	# leader selection
	lead_genome = ng.evolution.selection(genome, err_arr)
	# crossover - combining the genetic information of two parents to produce new offspring
	genome = ng.evolution.crossover(lead_genome)
	# genome mutation of new generations
	ng.evolution.mutation(genome)
	# generation of weight matrices from a new genome
	w_matrix_list = ng.builders.neuro_builder(genome)

print("\n*** Evolution is finished ***")
print("\n-----------------------------")
print("\n*** Testing started ***\n")

# extract the champion genome
champion_index = np.argsort(np.array(err_arr))[0]
champ_w = [tensor[champion_index] for tensor in w_matrix_list]

while True:
	test_arr = input("Enter data (example: 1 1) or 'stop' to exit: ")
	if test_arr == "stop":
		break
	try:
		resp = nn(np.array(test_arr.split()).astype("float"), champ_w)
		print(f"Response: {resp}")
	except:
		print("Entered incorrect data")


