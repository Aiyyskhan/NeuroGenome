<img src="https://raw.githubusercontent.com/Aiyyskhan/NeuroGenome/main/docs/NeuroGenome_1_1_white.jpeg" align="middle" width="1000"/>

<p align="center">
<img src="https://img.shields.io/badge/version-v0.1.5-blue.svg?style=flat&colorA=007D8A&colorB=E1523D">
<img src="https://img.shields.io/badge/license-MIT-brightgreen">
</p>

**NeuroGenome** is a bioinspired open-source project that allows you to create artificial neural networks with a genetic code.
The neurogenome allows decomposing the problem of optimization of neural networks by evolutionary algorithms.
Neurogenome genes, like real genes in biological organisms, can mutate, recombine in a crossover process, and be transmitted to new generations.

- [Installation](#installation)
- [Example scheme and settings](#example-scheme-and-settings)
- [Examples](#examples)
- [License](#license)

## Installation
```python
pip install neurogenome
```

## Example scheme and settings

```python
# gene localization scheme
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

# hyperparameters
SETTINGS = {
	"population size": 50,
	"number of leaders": 5,
	"select by": "max",
	"number of input nodes per gene": 5,
	"number of hidden nodes per gene": 4,
	"number of output nodes per gene": 3,
	"schema": schema_0,
}
```

## Examples

[Go to all examples](https://github.com/Aiyyskhan/NeuroGenome/tree/main/examples)

| Examples | Description | Remark |
|:-|:-|:-|
| [Schema and settings](https://github.com/Aiyyskhan/NeuroGenome/blob/main/examples/schema_settings.py)| Example of a gene layout for weight matrix assembly | - |
| [XOR](https://github.com/Aiyyskhan/NeuroGenome/blob/main/examples/XOR)	| Example with a standard XOR task	| - |
| [2D Labyrinth (NumPy)](https://github.com/Aiyyskhan/NeuroGenome/blob/main/examples/2D_Labyrinth_NumPy)	| Example with 2D simulation "Labyrinth" on PyGame. Neural network computation is done with NumPy | Dependencies: `pygame==2.1.2` |
| [2D Labyrinth (PyTorch)](https://github.com/Aiyyskhan/NeuroGenome/blob/main/examples/2D_Labyrinth_PyTorch)	| Example with 2D simulation "Labyrinth" on PyGame. Neural network computation is done with PyTorch | Dependencies: `pygame==2.1.2`, `torch==1.6.0` |

## License

[MIT](https://opensource.org/licenses/MIT)