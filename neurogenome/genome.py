
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from numpy import ndarray, uint8, float16


I_DTYPE = uint8
F_DTYPE = float16

MAIN_VALUE_SEQUENCE = [-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0]

@dataclass
class Genome:
	settings: Dict[str, Any]
	main_val_seq: ndarray
	iGenes: ndarray
	hGenes: ndarray
	oGenes: ndarray

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
	def value_sequence(self) -> ndarray:
		return self.main_val_seq

	@property
	def schema(self) -> List[List[str]]:
		return self.settings["schema"]

	@property
	def num_individuals(self) -> int:
		return self.iGenes.shape[1]

	@property
	def num_neurons(self) -> Tuple[int, ...]:
		return self.iGenes.shape[2], self.hGenes.shape[2], self.oGenes.shape[3]

	@property
	def num_genes(self) -> Tuple[int, ...]:
		return self.iGenes.shape[0], self.hGenes.shape[0], self.oGenes.shape[0]