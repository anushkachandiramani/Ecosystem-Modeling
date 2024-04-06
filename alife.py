"""
File: alife.py
Description: A simple artificial life simulation.

"""
import random as rnd
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
import matplotlib as mpl
mpl.use('macosx')

def parse_args():
    """Function to parse command line arguments"""
    parser = argparse.ArgumentParser(description="Artificial life simulation")
    parser.add_argument("--grass_rate", type=float, default=0.1, help="Probability of grass growing")
    parser.add_argument("--fox_k", type=int, default=10, help="Max cycles a fox can go without food")
    parser.add_argument("--size", type=int, default=400, help="x/y dimensions of the field")
    parser.add_argument("--init_rabbits", type=int, default=50, help="# of starting rabbits")
    parser.add_argument("--init_foxes", type=int, default=20, help="# of starting foxes")
    parser.add_argument("--offspring_rabbit", type=int, default=2, help="Max offspring per rabbit")
    parser.add_argument("--speed", type=int, default=1, help="Number of generations per frame")
    return parser.parse_args()

args = parse_args()

# Updating simulation to use the parsed command-line arguments
SIZE = args.size # x/y dimensions of the field
WRAP = True # When moving beyond the border, do we wrap around to the other size
GRASS_RATE = args.grass_rate # Probability of grass growing at any given location, e.g., 2%
INIT_RABBITS = args.init_rabbits # Number of starting rabbits
INIT_FOXES = args.init_foxes # Number of starting foxes
FOX_K = args.fox_k # fox k value
OFFSPRING_RABBIT = args.offspring_rabbit # The number of offspring when a rabbit reproduces
SPEED = args.speed # Number of generations per frame


class Rabbit:

    def __init__(self):
        """ initialize with a random position on field """
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0

    def reproduce(self):
        """ reset eaten amount """
        self.eaten = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        """ increase the eaten amount based on available grass """
        self.eaten += amount

    def move(self):
        """ move the rabbit one step in any direction """
        if WRAP:
            self.x = (self.x + rnd.choice([-1, 0, 1])) % SIZE
            self.y = (self.y + rnd.choice([-1, 0, 1])) % SIZE
        else:
            self.x = min(SIZE-1, max(0, (self.x + rnd.choice([-1, 0, 1]))))
            self.y = min(SIZE-1, max(0, (self.y + rnd.choice([-1, 0, 1]))))

class Fox:
    def __init__(self):
        """ initialize with a random position """
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.k = FOX_K # number of cycles a fox can go without food
        self.hunger = 0 # tracks how many cycles a fox has gone without food

    def move(self):
        """ move the fox 0 to 2 steps in any direction """
        if WRAP:
            self.x = (self.x + rnd.choice([-2, -1, 0, 1, 2])) % SIZE
            self.y = (self.y + rnd.choice([-2, -1, 0, 1, 2])) % SIZE
        else:
            self.x = min(SIZE - 1, max(0, (self.x + rnd.choice([-2, -1, 0, 1, 2]))))
            self.y = min(SIZE - 1, max(0, (self.y + rnd.choice([-2, -1, 0, 1, 2]))))

    def eat(self, caught_rabbit):
        """ resets hunger if a rabbit is caught, otherwise increases hunger """
        if caught_rabbit:
            self.hunger = 0
        else:
            self.hunger += 1

    def starve(self):
       """ check if the fox hunger exceeds max cycles without food """
       return self.hunger > self.k

    def reproduce(self):
        """ fox reproduces when hunger is 0 """
        if self.hunger == 0:
            return copy.deepcopy(self)
        return None

    def is_hungry(self):
        """ determines if the fox is hungry - if hunger is greater than or equal
        to number of cycles before starvation"""
        return self.hunger >= self.k


class Field:
    """ A field is a patch of grass with 0 or more rabbits hopping around
    in search of grass """

    def __init__(self):
        """ initializes the field with a inputted number of rabbits and foxes """
        self.rabbits = [Rabbit() for _ in range(INIT_RABBITS)]
        self.foxes = [Fox() for _ in range(INIT_FOXES)]
        self.field = np.zeros(shape=(SIZE, SIZE), dtype=int)  # Change to 0 for unoccupied
        # Add population tracking lists
        self.rabbit_population = []
        self.fox_population = []
        self.grass_population = []

    def add_rabbit(self, rabbit):
        """ adds a new rabbit to the field """
        self.rabbits.append(rabbit)

    def add_fox(self, fox):
        """ adds a new fox to the field """
        self.foxes.append(fox)

    def move(self):
        """ move all animals on the field and update their positions """

        # Clear previous positions of rabbits and foxes
        self.field[self.field > 0] = 1  # keeps grass but clears animals

        for r in self.rabbits:
            r.move()
            self.field[r.x, r.y] = 2  # mark rabbit position
        for f in self.foxes:
            f.move()
            # checks if fox moves to a rabbit position
            if self.field[f.x, f.y] == 2:
                f.eat(True)
                self.rabbits = [r for r in self.rabbits if r.x != f.x or r.y != f.y]  # removes eaten rabbit
            else:
                f.eat(False)
            if not f.starve():
                self.field[f.x, f.y] = 3  # mark fox position if it's not starving

    def eat(self):
        """ All rabbits try to eat grass at their current location """
        for r in self.rabbits:
            r.eat(self.field[r.x, r.y])
            self.field[r.x, r.y] = 0 # remove the grass after eating

    def survive(self):
        """ Rabbits and foxes that have not eaten die. Otherwise, they live """
        self.rabbits = [r for r in self.rabbits if r.eaten > 0]
        self.foxes = [f for f in self.foxes if not f.starve()]

    def reproduce(self):
        """ handles reproduction for rabbits and foxes """
        born_rabbits = []
        for r in self.rabbits:
            for _ in range(rnd.randint(1, OFFSPRING_RABBIT)):
                born_rabbits.append(r.reproduce())
        self.rabbits += born_rabbits

        born_foxes = [f.reproduce() for f in self.foxes if f.reproduce() is not None]
        self.foxes += born_foxes

    def grow(self):
        """ grass growth on the field """
        growloc = (np.random.rand(SIZE,SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def generation(self):
        """ Run one generation of rabbit actions """
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self.grow()
        self.rabbit_population.append(len(self.rabbits))
        self.fox_population.append(len(self.foxes))
        self.grass_population.append(np.sum(self.field == 1))

def animate(i, field, im):
    """ Animates the field """
    for _ in range(SPEED):
        field.generation()
    im.set_array(field.field)
    plt.title(f"Generation: {i * SPEED} | Rabbits: {len(field.rabbits)} | Foxes: {len(field.foxes)}")
    return im,

def update_plot(frame, field, ax):
    """ Updates the plot for the animation """
    field.generation()

    # Clear previous scatter plots (rabbits and foxes)
    ax.clear()

    # Recreate the plot
    ax.imshow(np.ones((SIZE, SIZE)), cmap=ListedColormap(['white']), interpolation='nearest', aspect='auto')

    # Scatter plot for grass
    grass_x, grass_y = np.where(field.field == 1)
    ax.scatter(grass_x, grass_y, color='green', s=1)  # Adjust size as needed

    # Prepare lists for rabbit and fox positions
    rabbit_positions = [(rabbit.x, rabbit.y) for rabbit in field.rabbits]
    fox_positions = [(fox.x, fox.y) for fox in field.foxes]

    # Extract x and y coordinates for rabbits and foxes
    rabbit_x, rabbit_y = zip(*rabbit_positions) if rabbit_positions else ([], [])
    fox_x, fox_y = zip(*fox_positions) if fox_positions else ([], [])

    # Scatter plot for rabbits and foxes
    if rabbit_positions:
        ax.scatter(rabbit_x, rabbit_y, color='blue', s=5)  # Adjust size as needed
    if fox_positions:
        ax.scatter(fox_x, fox_y, color='red', s=5)  # Adjust size as needed

    plt.title(f'Generation: {frame} with {len(field.rabbits)} rabbits and {len(field.foxes)} foxes.')

def main():
    field = Field()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, SIZE)
    ax.set_ylim(0, SIZE)
    
    # white background
    ax.imshow(np.ones((SIZE, SIZE)), cmap=ListedColormap(['white']), interpolation='nearest', aspect='auto')
    
    # specifies number of generations
    num_generations = 30
    show_interval = 200

    # renders the animation
    anim = animation.FuncAnimation(fig, update_plot, fargs=(field, ax), frames=num_generations, interval=show_interval, repeat=False)

    plt.show()


    generations = range(len(field.rabbit_population)) # number of generations

    # plot populations and colors
    plt.plot(generations, field.rabbit_population, label='Rabbits', color='blue')
    plt.plot(generations, field.fox_population, label='Foxes', color='red')
    plt.plot(generations, field.grass_population, label='Grass', color='green')

    plt.xlabel('Generation')
    plt.ylabel('Population')
    plt.title('Population dynamics over time')
    plt.legend()
    plt.show()

    print(f'Generation complete. Grass: {field.grass_population[-1]}, Rabbits: {field.rabbit_population[-1]}, Foxes: {field.fox_population[-1]}')


if __name__ == '__main__':
    main()
