from typing import List
import numpy as np

from neurogenome.genome import Genome, I_DTYPE


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
		population.value_sequence,
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
		leaders.value_sequence,
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
	max_val = len(population.value_sequence) - 1
	
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
	child_genes = np.zeros((genes.shape[0], population_size, genes.shape[2], genes.shape[3]), dtype=I_DTYPE)

	a = np.linspace(0.2, 1.0, num=num_individuals)[::-1]
	b = a/a.sum()
	parents_indices = np.random.choice(num_individuals, population_size, p=b)

	individuals_indices = np.arange(num_individuals)

	for idx_ch, idx_p0 in enumerate(parents_indices):
		idx_p1 = individuals_indices[individuals_indices != idx_p0][np.random.randint(num_individuals-1)]
		mask = np.random.randint(2, size=(genes.shape[2], genes.shape[3])).astype(I_DTYPE)

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