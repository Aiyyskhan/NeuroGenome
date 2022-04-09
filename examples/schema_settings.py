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