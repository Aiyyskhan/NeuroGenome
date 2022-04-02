"""
**************************************

NeuroGenome v.0.1.0

Created in 20.11.2021
by Aiyyskhan Alekseev 

https://github.com/Aiyyskhan
aiyyskhan@gmail.com

License: MIT

**************************************
"""

__author__ = "Aiyyskhan Alekseev"
__version__ = "0.1.0"

from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import json
import numpy as np
import h5py

I_DTYPE = np.uint8
F_DTYPE = np.float16

# example of gene localization scheme
schema_0 = [
	[
		["i0","i1","i0","i1"],
		["h1","h0","h1","h0"]
	],
	[
		["i2","i3","i2","i3"],
		["h3","h2","h3","h2"]
	],
	[
		["o0","o1"],
		["o1","o0"]
	]
]

# example of hyperparameters
SETTINGS = {
	"population size": 50,
	"number of leaders": 5,
	"select by": "max", # "min"
	"number of input nodes per gene": 5,
	"number of hidden nodes per gene": 4,
	"number of output nodes per gene": 3,
	"value sequence": [-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0],
	"schema": schema_0,
}

# структура генома
@dataclass
class Genome:
	settings: Dict[str, Any]
	iGenes: np.ndarray
	hGenes: np.ndarray
	oGenes: np.ndarray

	@property
	def population_size(self) -> int:
		return self.settings["population size"]

	@property
	def num_leaders(self) -> int:
		return self.settings["number of leaders"]

	@property
	def select_dir(self) -> str:
		return self.settings["select by"]

	@property
	def value_sequence(self) -> List:
		return self.settings["value sequence"]

	@property
	def schema(self) -> List:
		return self.settings["schema"]

	@property
	def num_individuals(self) -> int:
		return self.iGenes.shape[1]

	@property
	def num_neurons(self) -> Tuple[int, int, int]:
		return self.iGenes.shape[2], self.hGenes.shape[2], self.oGenes.shape[3]

	@property
	def num_genes(self) -> Tuple[int, int, int]:
		return self.iGenes.shape[0], self.hGenes.shape[0], self.oGenes.shape[0]

# методы-сборщики
# сборщик генома
def genome_builder(settings: Dict[str, Any]) -> Genome:

	population_size = settings["population size"]
	num_inputs = settings["number of input nodes per gene"]
	num_hiddens = settings["number of hidden nodes per gene"]
	num_outputs = settings["number of output nodes per gene"]

	val_len = len(settings["value sequence"])

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
		np.random.randint(val_len, size=(i_gene_count, population_size, num_inputs, num_hiddens), dtype=I_DTYPE),
		np.random.randint(val_len, size=(h_gene_count, population_size, num_hiddens, num_hiddens), dtype=I_DTYPE),
		np.random.randint(val_len, size=(o_gene_count, population_size, num_hiddens, num_outputs), dtype=I_DTYPE)
	)

# сборщик нейросетевых матриц из генома
def neuro_builder(genome: Genome) -> List:
	l0 = []
	val = np.array(genome.value_sequence, dtype=F_DTYPE)
	for layer in genome.schema: 
		l1 = []
		for row in layer:
			l2 = []
			for item in row:
				if item[0] == 'i':
					l2.append(val[genome.iGenes[int(item[1])]])
				elif item[0] == 'h':
					l2.append(val[genome.hGenes[int(item[1])]])
				elif item[0] == 'o':
					l2.append(val[genome.oGenes[int(item[1])]])
			l1.append(l2)
		l0.append(np.block(l1))

	return l0

# методы модификации массивов генома
def adding_random_fragment(arr: np.ndarray, axis: int, max_val: int) -> np.ndarray:
	"""
	Метод добавления случайного фрагмента в массив

	**************

	Parameters
	----------
	arr : np.ndarray
		целевой массив
	axis : int
		ось добавления
	max_val : int
		максимальное значение диапазона случайных значений
	"""
	size = np.array(arr.shape)
	size[axis] = 1
	return np.append(arr, np.random.randint(max_val, size=size), axis)

def fragment_duplication(arr: np.ndarray, axis: int, frag_idx: int) -> np.ndarray:
	"""
	Метод дублирования фрагмента массива

	**************

	Parameters
	----------
	arr : np.ndarray
		целевой массив
	axis : int
		ось добавления
	frag_idx : int
		индекс фрагмента
	"""
	if axis == 0: # дублирование гена
		fragment = arr[frag_idx,:,:,:][None,:,:,:]
	elif axis == 1: # дублирование особи
		fragment = arr[:,frag_idx,:,:][:,None,:,:]
	elif axis == 2: # дублирование строки генов
		fragment = arr[:,:,frag_idx,:][:,:,None,:]
	elif axis == 3: # дублирование столбца генов
		fragment = arr[:,:,:,frag_idx][:,:,:,None]
		
	return np.append(arr, fragment, axis)

def genomes_concatenation(genome_1: Genome, genome_2: Genome) -> None:
	"""
	Метод конкатенации двух геномов

	**************
	
	Parameters
	----------
	genome_1 : Genome
		целевой геном
	genome_2 : Genome
		добавляемый геном
	"""
	genome_1.iGenes = np.append(genome_1.iGenes, genome_2.iGenes, axis=1)
	genome_1.hGenes = np.append(genome_1.hGenes, genome_2.hGenes, axis=1)
	genome_1.oGenes = np.append(genome_1.oGenes, genome_2.oGenes, axis=1)

# эволюционные методы
# метод отбора
def selection(population: Genome, results: List[float]) -> Genome:
	"""
	Метод отбора

	**************
	
	Parameters
	----------
	population : Genome
		геном популяции
	results : List[float]
		список с результатами (вознаграждениями) по каждой особи
	num_leaders : int
		количество необходимых лидеров
	max_first : bool
		направление сортировки: по умолчанию True - особь с максимальным результатом окажется первым
	"""

	# сортировка и отбор лидеров по результатам
	select_dir = population.select_dir
	num_leaders = population.num_leaders
	if select_dir == "max":
		leader_indices = np.argsort(np.array(results))[::-1][:num_leaders]
	else:
		leader_indices = np.argsort(np.array(results))[:num_leaders]

	return Genome(
		population.settings,
		population.iGenes[:, leader_indices].copy(),
		population.hGenes[:, leader_indices].copy(),
		population.oGenes[:, leader_indices].copy()
	)

# метод скрещивания
def crossover(leaders: Genome) -> Genome:
	"""
	Метод кроссинговера
	**************
	
	Parameters
	----------
	leaders : Genome
		геном лидеров
	"""
	num_individuals = leaders.num_individuals
	population_size = leaders.population_size

	return Genome(
		leaders.settings,
		__hybridization(leaders.iGenes.copy(), num_individuals, population_size),
		__hybridization(leaders.hGenes.copy(), num_individuals, population_size),
		__hybridization(leaders.oGenes.copy(), num_individuals, population_size)
	)

# метод мутации
def mutation(population: Genome, mu: float = 0.0, sigma: float = 0.5) -> None:
	"""
	Метод мутации
	**************
	
	Parameters
	----------
	population : Genome
		популяция
	"""
	max_val = len(population.settings["value sequence"]) - 1
	
	__mutate(population.iGenes, max_val, mu, sigma)
	__mutate(population.hGenes, max_val, mu, sigma)
	__mutate(population.oGenes, max_val, mu, sigma)

def __hybridization(genes: np.ndarray, num_individuals: int, population_size: int) -> np.ndarray:
	"""
	Метод гибридизации двух весовых тензоров
	**************
	
	Parameters
	----------
	genes : np.ndarray 
		массив генов
	"""
	# создание дочернего тензора
	child_genes = np.zeros((genes.shape[0], population_size, genes.shape[2], genes.shape[3]), dtype=np.uint8)

	a = np.linspace(0.2, 1.0, num=num_individuals)[::-1]
	b = a/a.sum()
	parents_indices = np.random.choice(num_individuals, population_size, p=b)

	individuals_indices = np.arange(num_individuals)

	for idx_ch, idx_p0 in enumerate(parents_indices):
		idx_p1 = individuals_indices[individuals_indices != idx_p0][np.random.randint(num_individuals-1)]
		mask = np.random.randint(2, size=(genes.shape[2], genes.shape[3])).astype(np.uint8)

		child_genes[:, idx_ch] = genes[:, idx_p0] * mask + genes[:, idx_p1] * (mask ^ 1)

	# возвращаем дочерний тензор
	return child_genes

def __mutate(genes: np.ndarray, max_val: int, mu: float, sigma: float) -> None:
	"""
	Метод мутационной модификации весовых тензоров
	**************
	
	Parameters
	----------
	genes : np.ndarray 
		массив генов
	max_val : int
		максимальное значение
	mu : float
		медиана нормального распределения
	sigma : float
		стандартное отклонение нормального распределения
	"""

	num_individuals = genes.shape[1]

	for gene in genes:
		selected_individuals = np.random.randint(num_individuals, size=np.random.randint(1, num_individuals))
		gene_selected_individuals = gene[selected_individuals].copy()
		mutation_value = np.random.normal(mu, sigma, gene_selected_individuals.shape)
		gene[selected_individuals] = np.clip(np.around(gene_selected_individuals + mutation_value), 0, max_val).astype(I_DTYPE)

# методы сохранения и загрузки геномов
def save_genome(path: str, genome: Genome) -> None:
	""" 
	Метод сохранения генома
	**************
	
	Parameters
	----------
	path : str
		относительный путь
	genome : Genome
		сохраняемый геном
	"""
	if ".hdf5" in path:
		with h5py.File(path, 'w') as f:
			f.attrs["settings"] = json.dumps(genome.settings)
			_ = f.create_dataset("iGenes", data=genome.iGenes)
			_ = f.create_dataset("hGenes", data=genome.hGenes)
			_ = f.create_dataset("oGenes", data=genome.oGenes)
	# elif ".npy" in path:
	# 	with open(path, 'wb') as f:
	# 		np.save(f, genome.val)
	# 		np.save(f, genome.iGenes)
	# 		np.save(f, genome.hGenes)
	# 		np.save(f, genome.oGenes)
	else:
		raise Exception("Неподдерживаемый тип файла")

def load_genome(path: str) -> Genome:
	"""
	Метод загрузки генома
	**************
	
	Parameters
	----------
	path : str
		относительный путь
	"""
	if ".hdf5" in path:
		settings = dict()
		with h5py.File(path, 'r') as f:
			settings = json.loads(f.attrs["settings"])
			iGenes = np.array(f["iGenes"])
			hGenes = np.array(f["hGenes"])
			oGenes = np.array(f["oGenes"])
	# elif ".npy" in path:
	# 	with open(path, 'rb') as f:
	# 		val = np.load(f, allow_pickle=True)
	# 		iGenes = np.load(f, allow_pickle=True)
	# 		hGenes = np.load(f, allow_pickle=True)
	# 		oGenes = np.load(f, allow_pickle=True)
	else:
		raise Exception("Неподдерживаемый тип файла")

	return Genome(
		settings,
		iGenes,
		hGenes,
		oGenes,
	)
