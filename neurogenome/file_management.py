import json
import h5py
import numpy as np

from neurogenome.genome import Genome


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
			_ = f.create_dataset("val_seq", data=genome.value_sequence)
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
			val_seq = np.array(f["val_seq"])
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
		val_seq,
		iGenes,
		hGenes,
		oGenes,
	)