import pygame
import math
import numpy as np 
import random
import sys
import os
os.environ["PATH"] += 'C:/Program Files/Graphviz 2.44.1/bin/'

import neat
import pickle
from visualize import *

pygame.font.init()  # init font

WIDTH = 500
HEIGHT =700
SCREEN_TITLE = 'Smart Dots'
GRAVITY =9.8



FLOOR = 650
gen=0

STAT_FONT = pygame.font.SysFont("comicsans", 50)

pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

pipe_img = (pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bird_images = [(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())

# ---------------------------------------------------------------------------------------------------------------
# Base class for floor
class Base:
    """
    Represnts the moving floor of the game
    """
    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        """
        Initialize the object
        :param y: int
        :return: None
        """
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        """
        move floor so it looks like its scrolling
        :return: None
        """
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        """
        Draw the floor. This is two images that move together.
        :param win: the pygame surface/window
        :return: None
        """
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


# ---------------------------------------------------------------------------------------------------------------
class Bird:
	def __init__(self):


		self.x = 175
		self.y =200
		self.vel = 0
		self.tick_count=0
		self.is_alive=True

		self.IMGS = bird_images
		self.img = self.IMGS[0]

		


	def jump(self):
		self.vel = 10.0
		self.tick_count=0
		self.height = self.y

	def move(self):

		self.tick_count+=1

		if self.tick_count< 5:
			pass
		else:
			if self.vel>-40:
				self.vel-=0.75


		disp = self.vel


	
		if self.y>=HEIGHT-10 and disp<0:
			disp=0

		self.y-= disp

	def update(self):
		self.move()

		if self.y>= 650:
			self.is_alive=False

	def get_mask(self):
		# returns mask of bird
		return pygame.mask.from_surface(self.img)



# ---------------------------------------------------------------------------------------------------------------
class Pipes:
	def __init__(self, startingX):
		self.speed = 5
		self.x= startingX

		self.pipe_top = pygame.transform.flip(pipe_img, False, True)
		self.pipe_bottom = pipe_img

		self.gap = 150
		self.height = random.randrange(100, 400)
		self.top = self.height - self.pipe_top.get_height()
		self.bottom = self.height + self.gap


		self.passed = False
		self.dead = False


	def move(self):

		self.x -=self.speed
		

	def update(self):
		self.move()

		if self.x<120:
			print('----------------------------------------passed')
			self.passed = True

		if self.x<=-50:
			self.dead=True


	def collide(self, bird, screen):
		bird_mask = bird.get_mask()
		top_mask=pygame.mask.from_surface(self.pipe_top)
		bottom_mask = pygame.mask.from_surface(self.pipe_bottom)

		top_offset = (self.x - bird.x, self.top-int(bird.y))
		bottom_offset = (self.x - bird.x, self.bottom-int(bird.y))

		b_point = bird_mask.overlap(bottom_mask, bottom_offset)
		t_point=bird_mask.overlap(top_mask, top_offset)

		if b_point or t_point:
			return True

		return False


# ---------------------------------------------------------------------------------------------------------------
def draw_window(win, birds, pipes, base, score, gen, pipe_ind, draw_lines = True):
	"""
	draws the windows for the main game loop
	:param win: pygame window surface
	:param bird: a Bird object
	:param pipes: List of pipes
	:param score: score of the game (int)
	:param gen: current generation
	:param pipe_ind: index of closest pipe
	:return: None
	"""

	if gen == 0:
		gen = 1
	win.blit(bg_img, (0,0))

	for pipe in pipes:
		win.blit(pipe.pipe_top, (pipe.x, pipe.top))
		win.blit(pipe.pipe_bottom, (pipe.x, pipe.bottom))

	base.draw(win)
	for bird in birds:
        # draw lines from bird to pipe
		if draw_lines:
			try:
				pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
				pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
			except:
				pass
		# draw bird
		win.blit(bird.img, (bird.x, int(bird.y)))

	# score
	score_label = STAT_FONT.render("Score: " + str(score),1,(255,255,255))
	win.blit(score_label, (WIDTH - score_label.get_width() - 15, 10))

	# generations
	score_label = STAT_FONT.render("Gens: " + str(gen-1),1,(255,255,255))
	win.blit(score_label, (10, 10))

	# alive
	score_label = STAT_FONT.render("Alive: " + str(len(birds)),1,(255,255,255))
	win.blit(score_label, (10, 50))
	pygame.draw.aaline(win, (255,0,0), (175,0), (175, 800))
	pygame.display.update()


# ---------------------------------------------------------------------------------------------------------------

def eval_genomes(genomes, config):

	global WIN, gen
	screen = WIN
	gen+=1

	nets=[]
	birds=[]
	ge=[]


	for genome_id, genome in genomes:
		genome.fitness =0
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		nets.append(net)
		birds.append(Bird())
		ge.append(genome)

	# Create objects
	base = Base(650)
	pipes = []
	startingX = WIDTH+50
	for i in range(5):
		pipes.append(Pipes(startingX))
		startingX+= random.randrange(250, 350, 1)



	clock = pygame.time.Clock()
	score = 0
	run = True

	dist = 0
	pipe_ind =0

	while run and len(birds)>0:
		clock.tick(60)

		dist +=1

		for event in pygame.event.get():
			if event.type ==pygame.QUIT:
				run = False
				pygame.quit()
				quit()
				break
		
		delete = False
		

		
		for i in range(len(pipes)):
			pipes[i].update()

			if pipes[i].dead == True:
				delete =True
				continue

			if pipes[i].passed ==True:
				
				pipe_ind = i+1


		
		if delete:
			del pipes[0]
			pipe_ind-=1
			score+=1
			pipes.append(Pipes(pipes[-1].x+random.randrange(275, 325, 1)))

			for g in ge:
				g.fitness+= 5				# Every new pipe +5 points



		for pipe in pipes:
			for bird in birds:
				if pipe.collide(bird, screen):
					bird.is_alive = False
					
					ge[birds.index(bird)].fitness -= 5				# -5 if death
					nets.pop(birds.index(bird))
					ge.pop(birds.index(bird))
					birds.pop(birds.index(bird))



		for x, bird in enumerate(birds):
			#ge[x].fitness = score
			bird.update()

			# Inputs Height, top pipe, bottom pipe
			output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y-pipes[pipe_ind].bottom)))

			if output[0]>0.5:
				bird.jump()

		base.move()



		


		for bird in birds:
			if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
				
				ge[birds.index(bird)].fitness -= 5				# -5 if death
				nets.pop(birds.index(bird))
				ge.pop(birds.index(bird))
				birds.pop(birds.index(bird))

		
		draw_window(WIN, birds, pipes, base, score, gen, pipe_ind)


	node_names = {-1:'Bird Y', -2:'Top Pipe Y', -3:'Bottom Pipe Y', 0:'Output'}
	draw_net(config, genome, filename='analytics', node_names=node_names)

	print('Saved Analytics')
	#sys.exit()
	#plot_stats(neat.StatisticsReporter(), view =True)

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)


    # Create the population, which is the top-level object for a NEAT run.

    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 10000)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))




if __name__ =='__main__':
	local_dir=os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config-feedforward.txt')
	print(config_path)
	run(config_path)











	'''
	game = Game()
	while True:
		game.update()
	

		# ---------------------------------------------------------------------------------------------------------------
class Game:
	
	def __init__(self):
		
		self.screen = WIN
		self.clock = pygame.time.Clock()
		self.dt =0
		
		self.game_speed=60

		self.bird = bird()
		self.base = Base(650)
		self.done=False


		self.pipes = []
		startingX = WIDTH+50
		for i in range(5):
			self.pipes.append(pipes(startingX))
			startingX+= random.randrange(250, 350, 1)
		

	
	def view(self):
		

		self.screen.fill(pygame.Color('lightskyblue'))
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
			elif event.type == pygame.KEYDOWN:
				if event.key ==pygame.K_SPACE:
					self.bird.jump()

		self.screen.blit(bg_img, (0,0))
		self.base.draw(self.screen)


		#pygame.draw.circle(self.screen, (255, 0,0), (self.bird.x, int(self.bird.y)), 10)


		for i in range(len(self.pipes)):
			#pygame.draw.circle(self.screen, (255, 0,0), (self.pipes[i].x, 100), 10)

			self.screen.blit(self.pipes[i].pipe_top, (self.pipes[i].x, self.pipes[i].top))
			self.screen.blit(self.pipes[i].pipe_bottom, (self.pipes[i].x, self.pipes[i].bottom))

		self.base.draw(self.screen)
		self.screen.blit(self.bird.image, (self.bird.x, int(self.bird.y)))
		pygame.display.update()

	
	def manage_pipes(self):
		delete = False
		self.next_pipe = -1


		for i in range(len(self.pipes)):
			self.pipes[i].update()

			if self.pipes[i].dead == True:
				delete =True
			if self.next_pipe ==-1 and self.pipes[i].passed ==False:
				self.next_pipe = i
		
		if delete:
			del self.pipes[0]
			self.pipes.append(pipes(self.pipes[-1].x+random.randrange(250, 350, 1)))




	def manage_bird(self):
		self.bird.update()
		
		if not self.bird.is_alive:
			self.done=True
			print('Game Over')
			pygame.quit()
			sys.exit()





	def update(self):
		self.dt+= self.clock.tick(60)
		if self.dt >0:
			self.manage_bird()
			self.manage_pipes()
			self.base.move()

			if self.pipes[self.next_pipe].collide(self.bird, self.screen):
				self.done=True
				print('Game Over')
				pygame.quit()
				sys.exit()


			self.view()
			self.dt=
	'''