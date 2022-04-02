import numpy as np

from neurogenome.genome import Genome


def adding_genes(genome: Genome, num_genes: int, select_method: str = "r"):
	
	pass

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