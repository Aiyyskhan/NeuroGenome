import neurogenome as ng

import nn
from player import Player
from ray_casting import RayCast
from settings import *

# examples of schemes
schema_0 = [
	[
		["i0"]
	],
	[
		["h0"]
	],
	[
		["o0"]
	]
]

schema_1 = [
	[
		["i0"]
	],
	[
		["h0","h1","h2","h3","h4"]
	],
	[
		["o0"],
		["o1"],
		["o2"],
		["o3"],
		["o4"],
	]
]

schema_2 = [
	[
		["i0", "i1"]
	],
	[
		["h0","h1","h2","h3","h4"],
        ["h5","h6","h7","h8","h9"]
	],
	[
		["o0"],
		["o1"],
		["o2"],
		["o3"],
		["o4"],
	]
]

# example of a scheme with bilateral symmetry
schema_3 = [
	[
		["i0", "i1"],
		["i1", "i0"]
	],
	[
		["h0","h1","h2","h3","h4","h5"],
        ["h3","h4","h5","h0","h1","h2"]
	],
	[
		["o0", "o6", "o3"],
		["o1", "o7", "o4"],
		["o2", "o8", "o5"],
		["o3", "o9", "o0"],
		["o4", "o10", "o1"],
		["o5", "o11", "o2"],
	]
]

SETTINGS = {
	"population size": NUM_PLAYERS,
	"number of leaders": 7,
	"select by": "max",
	"number of input nodes per gene": 3,
	"number of hidden nodes per gene": 20,
	"number of output nodes per gene": 1,
	"schema": schema_3,
}

# examples of weight sequences
# Explore different options!:)
# VAL_SEQ = [-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.]
# VAL_SEQ = [-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0]
VAL_SEQ = [-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0]
# VAL_SEQ = np.linspace(-3.0, 3.0, 25)
# VAL_SEQ = np.linspace(-1.0, 1.0, 41)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('*** 2D Labyrinth ***')
        self.sc = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont('Arial', 20)

        # game map processing
        self.finish_coords = set()
        self.wall_coord_list = list()
        self.world_map, self.collision_walls = self.get_map(MAP_0, TILE)
        for coord, signature in self.world_map.items():
            if signature == "1":
                self.wall_coord_list.append(coord)
            elif signature == "2":
                self.finish_coords.add(coord)

        # genome creation
        self.genome = ng.builders.genome_builder(SETTINGS, VAL_SEQ)
        # generation of weight matrices from the genome
        weight_list = ng.builders.neuro_builder(self.genome)
        # create neural networks
        self.brains = nn.NeuralNet(weight_list)
        
        # create players
        self.players = []
        players_color = np.random.randint(50, 220, size=(NUM_PLAYERS, 3))
        for i in range(NUM_PLAYERS):
            player = Player(self.sc, self.collision_walls, self.finish_coords)
            player.color = players_color[i]
            player.init_angle = math.pi + (math.pi/2)
            player.rays = RayCast(self.world_map)
            player.setup()
            self.players.append(player)
        
        self.number_of_live_players = NUM_PLAYERS
        self.number_of_winners = 0
        self.game_over = False
        self.generation = 1
        self.epoch = 0

    def get_map(self, map_list, tile_size):
        world_map = {}
        collision_walls = []
        for i, row in enumerate(map_list):
            for j, char in enumerate(row):
                if char != '.':
                    if char == '1':
                        world_map[(j * tile_size, i * tile_size)] = '1'
                        collision_walls.append(pygame.Rect(j * tile_size, i * tile_size, tile_size, tile_size))
                    elif char == '2':
                        world_map[(j * tile_size, i * tile_size)] = '2'
        return world_map, collision_walls

    def game_event(self):
        # collection of observations
        observation = []
        for idx, player in enumerate(self.players):
            if player.dead and not player.verified:
                self.number_of_live_players -= 1
                player.verified = True
            player.ray_casting()
            observation.append(player.rays.depth)

        # processing observations and generating actions by a neural network
        actions = self.brains(np.array(observation) / MAX_DEPTH)

        # player control
        for idx, player in enumerate(self.players):
            player.behavior_control(actions[idx][0], actions[idx][1], actions[idx][2])
            if not player.dead:
                player.movement()
                player.draw()

            if player.reached_finish=="y":
                player.reached_finish = "v"
                self.number_of_winners += 1
        
        # drawing map and information
        self.draw_map()
        self.info()

    def stop_function(self):
        if self.number_of_live_players == 0 or self.epoch >= MAX_EPOCH:
            self.game_over = True
            self.epoch = 0            

    def evolution(self):
        # creating a list of player scores
        result_list = []
        for player in self.players:
            result_list.append(player.reward)

        del self.brains

        # leader selection
        leader_genome = ng.evolution.selection(self.genome, result_list.copy())
        del self.genome

        # change the population size with the subtraction of the number of leaders, since they are preserved
        # SETTINGS["population size"] -= SETTINGS["number of leaders"]
        # leader_genome.settings = SETTINGS
            
        # crossover of leader genomes - combining the genetic information of two parents to produce new offspring
        self.genome = ng.evolution.crossover(leader_genome)

        # genome mutation of new generations
        # if np.random.randint(2) == 0:
        ng.evolution.mutation(self.genome)
        
        # adding leader genomes to the population
        # ng.macromutations.genomes_concatenation(self.genome, leader_genome)

        del leader_genome

        # recovery of population size information
        # SETTINGS["population size"] = NUM_PLAYERS
        # self.genome.settings = SETTINGS

        # generation of weight matrices from a new genome
        weight_list = ng.builders.neuro_builder(self.genome)

        # create new neural networks
        self.brains = nn.NeuralNet(weight_list)

        # players update
        for player in self.players:
            player.setup()
                
        self.number_of_live_players = NUM_PLAYERS
        self.game_over = False

        self.generation += 1

    def draw_map(self):
        for x, y in self.wall_coord_list:
            pygame.draw.rect(self.sc, WALL_COLOR_1, (x, y, TILE, TILE), 2)
        for x, y in self.finish_coords:
            pygame.draw.rect(self.sc, WALL_COLOR_2, (x, y, TILE, TILE), 2)

    def info(self):
        pygame.draw.rect(self.sc, TEXT_BACK_COLOR_1, (INFO_PANEL_PADDING_LEFT, INFO_PANEL_PADDING_TOP, INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT))
        render = self.font.render(f"Generation: {int(self.generation)}", 0, TEXT_COLOR_1)
        self.sc.blit(render, (INFO_PANEL_PADDING_LEFT+40, INFO_PANEL_PADDING_TOP+20))
        render = self.font.render(f"Players: {int(self.number_of_live_players)}", 0, TEXT_COLOR_1)
        self.sc.blit(render, (INFO_PANEL_PADDING_LEFT+40, INFO_PANEL_PADDING_TOP+50))
        render = self.font.render(f"Winners: {int(self.number_of_winners)}", 0, TEXT_COLOR_1)
        self.sc.blit(render, (INFO_PANEL_PADDING_LEFT+40, INFO_PANEL_PADDING_TOP+80))

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            self.sc.fill(BLACK)

            if not self.game_over:
                self.game_event()
                self.stop_function()
            else:
                self.evolution()

            pygame.display.flip()
            self.epoch += 1


if __name__ == "__main__":
    game = Game()
    game.run()