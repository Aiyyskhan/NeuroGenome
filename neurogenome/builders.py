from typing import List, Dict, Any
import numpy as np

from neurogenome.genome import Genome, I_DTYPE, F_DTYPE, MAIN_VALUE_SEQUENCE


def genome_builder(settings: Dict[str, Any]) -> Genome:

	population_size = settings["population size"]
	num_inputs = settings["number of input nodes per gene"]
	num_hiddens = settings["number of hidden nodes per gene"]
	num_outputs = settings["number of output nodes per gene"]

	val_len = len(MAIN_VALUE_SEQUENCE)

	i_gene_count = 0
	h_gene_count = 0
	o_gene_count = 0

	for layer in settings["schema"]:
		for item in np.unique(layer):
			if item[0] == 'i':
				i_gene_count += 1
			elif item[0] == 'h':
				h_gene_count += 1
			elif item[0] == 'o':
				o_gene_count += 1
	
	return Genome(
		settings,
		np.array(MAIN_VALUE_SEQUENCE, dtype=F_DTYPE),
		np.random.randint(val_len, size=(i_gene_count, population_size, num_inputs, num_hiddens), dtype=I_DTYPE),
		np.random.randint(val_len, size=(h_gene_count, population_size, num_hiddens, num_hiddens), dtype=I_DTYPE),
		np.random.randint(val_len, size=(o_gene_count, population_size, num_hiddens, num_outputs), dtype=I_DTYPE)
	)

# сборщик нейросетевых матриц из генома
def neuro_builder(genome: Genome) -> List:
	l0 = []
	vals = genome.value_sequence
	for layer in genome.schema: 
		l1 = []
		for row in layer:
			l2 = []
			for item in row:
				if item[0] == 'i':
					l2.append(vals[genome.iGenes[int(item[1])]])
				elif item[0] == 'h':
					l2.append(vals[genome.hGenes[int(item[1])]])
				elif item[0] == 'o':
					l2.append(vals[genome.oGenes[int(item[1])]])
			l1.append(l2)
		l0.append(np.block(l1))

	return l0