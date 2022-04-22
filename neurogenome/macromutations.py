from typing import List
import numpy as np

from neurogenome.genome import Genome


def adding_new_genes(genome: Genome, new_schema: List[List[str]], mode: str = "r"):
	"""
	Метод добавления генов по новой схеме

	**************

	Parameters
	----------
	genome : Genome
		геном
	new_schema : List[List[str]]
		новая схема с дополнительным набором генов
	mode : str
		режим добавления генов:
			r - случайный выбор способа генерации генов (создание нового гена или дублирование существующего)
			d - дублирование существующего гена
			a - создание нового гена
	"""
	i_gene_old_count, h_gene_old_count, o_gene_old_count = genome.num_genes

	i_gene_new_count = 0
	h_gene_new_count = 0
	o_gene_new_count = 0

	val_len = len(genome.value_sequence)

	for layer in new_schema:
		for item in np.unique(layer):
			if item[0] == 'i':
				i_gene_new_count += 1
			elif item[0] == 'h':
				h_gene_new_count += 1
			elif item[0] == 'o':
				o_gene_new_count += 1

	if i_gene_old_count < i_gene_new_count:
		new_iGenes = __mode_handler(genome.iGenes, mode, val_len, i_gene_old_count)
	elif h_gene_old_count < h_gene_new_count:
		new_hGenes = __mode_handler(genome.hGenes, mode, val_len, h_gene_old_count)
	elif o_gene_old_count < o_gene_new_count:
		new_oGenes = __mode_handler(genome.oGenes, mode, val_len, o_gene_old_count)

	settings = genome.settings.copy()
	settings["schema"] = new_schema

	return Genome(
		settings,
		genome.value_sequence,
		new_iGenes,
		new_hGenes,
		new_oGenes
	)

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

def __mode_handler(genes: np.ndarray, mode: str, max_val: int, gene_old_count: int) -> np.ndarray:
	if mode == "r":
		return __random_selector(genes, max_val, gene_old_count)
	elif mode == "d":
		frag_idx = np.random.randint(gene_old_count)
		return fragment_duplication(genes.copy(), 0, frag_idx)
	elif mode == "a":
		return adding_random_fragment(genes.copy(), 0, max_val)

def __random_selector(genes: np.ndarray, max_val: int, gene_old_count: int) -> np.ndarray:
	if np.random.choice([0,1], p=[0.5, 0.5]) == 0:
		return adding_random_fragment(genes.copy(), 0, max_val)
	else:
		frag_idx = np.random.randint(gene_old_count)
		return fragment_duplication(genes.copy(), 0, frag_idx)