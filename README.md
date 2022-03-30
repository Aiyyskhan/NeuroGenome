<img src="docs/NeuroGenome_1_1_white.jpeg" align="middle" width="1000"/>

<p align="center">
<img src="https://img.shields.io/badge/version-v0.1.0-blue.svg?style=flat&colorA=007D8A&colorB=E1523D">
<img src="https://img.shields.io/badge/license-MIT-brightgreen">
</p>

**NeuroGenome** is a bioinspired open-source project that allows you to create artificial neural networks with a genetic code.

- [Installation](#installation)
- [Settings example](#settings-example)
- [Examples](#examples)
- [License](#license)

## Installation
```python
pip install neurogenome
```
## Settings example

```python
# gene localization scheme
schema_0 = [
	[
		["i0","i1","h0","h1"],
		["i1","i0","h1","h0"]
	],
	[
		["i2","i3","h2","h3"],
		["i3","i2","h3","h2"]
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
	"value sequence": [-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0],
	"schema": schema_0,
}
```

## Examples

_in the pipeline_

## License

[MIT](https://opensource.org/licenses/MIT)