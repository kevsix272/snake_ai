#for graphics and handling events 
import pygame 
#to place food at random positions
import random
#to create clear, readible constanst and simple classes
from enum import Enum
from collections import namedtuple
import numpy as np

#inisalizes all py.game moduals
pygame.init()

#load font 
font = pygame.font.Font('./fonts/arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

#creats and sets every direction (right,left,up,down) like a compass
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

#creates an object that holds x and y positions
Point = namedtuple('Point', 'x, y')

# rgb colors, all colors used in the game
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (5, 140, 66)
GREEN2 = (6, 145, 69)
BLUE1 = (39,76,119)
BLUE2 = (39,76,119)
BLACK = (0, 0, 0)
GRAY = (169, 169, 169)
DARKGREEN = (2, 49, 23)
LICORICE = (37, 17, 1)

BLOCK_SIZE = 20
SPEED = 20 # snake speed

#encuslates the games state
class SnakeEnv():

    def __init__(self, w=640, h=480):# game window creation and size
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h)) #creates the game window
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock() # controls the frame rate
        self.action_space = [0, 1, 2]  # Straight, Left, Right #gives the dql agent movment options  
        self.reset() #sets the intial game state (food, position , score)

        # the snake starts moving right from the cener form thee center of the window, creates the snakes body, point is used to render the body
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                    Point(self.head.x-BLOCK_SIZE, self.head.y),
                    Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None 
        self._place_food() #randomly places food on the grid
        self.frame_iteration = 0

        # Returnera tillståndet
        return self.get_state(), {} #    retrurns an array of values about the env, the dql agent uses it to decide the next move
   
        #Simulates the movement, the calculation of next position without moving. Sees into the future. 
    def _get_next_position(self, action):
        # Simulate direction changes without modifying the actual direction
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 0:  # Straight
            new_dir = clock_wise[idx]
        elif action == 1:  # Right turn
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # Left turn (action == 2)
            new_dir = clock_wise[(idx - 1) % 4]

        # Calculate hypothetical new head position
        x, y = self.head.x, self.head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE

        return Point(x, y)
    
    #Checks for danger, determines if it is safe to move straight, down, right or left. Calculates the food location. 
    def get_state(self):
        #Danger calc
        danger_straight = self.is_collision(self._get_next_position(0))
        danger_right = self.is_collision(self._get_next_position(1))
        danger_left = self.is_collision(self._get_next_position(2))
        
        body_up = Point(self.head.x, self.head.y - BLOCK_SIZE) in self.snake[1:]
        body_down = Point(self.head.x, self.head.y + BLOCK_SIZE) in self.snake[1:]
        body_left = Point(self.head.x - BLOCK_SIZE, self.head.y) in self.snake[1:]
        body_right = Point(self.head.x + BLOCK_SIZE, self.head.y) in self.snake[1:]
        
        mapvalues = []

         # checks for danger in a 10x10 grid around the head
        for x in range(1,6):
            for y in range(1,6):
                mapvalues.append(Point(self.head.x - BLOCK_SIZE*x, self.head.y - BLOCK_SIZE*y) in self.snake[1:])
        for x in range(1,6):
            for y in range(1,6):
                mapvalues.append(Point(self.head.x - BLOCK_SIZE*x, self.head.y + BLOCK_SIZE*y) in self.snake[1:])
        for x in range(1,6):
            for y in range(1,6):
                mapvalues.append(Point(self.head.x + BLOCK_SIZE*x, self.head.y - BLOCK_SIZE*y) in self.snake[1:])
        for x in range(1,6):
            for y in range(1,6):
                mapvalues.append(Point(self.head.x + BLOCK_SIZE*x, self.head.y + BLOCK_SIZE*y) in self.snake[1:])
        

        #Food location calc
        #food_left = self.food.x < self.head.x
        #food_right = self.food.x > self.head.x
        #food_up = self.food.y < self.head.y
        #food_down = self.food.y > self.head.y
        
        # checks the direction for the closest food
        food_left = self.food.x < self.head.x
        food_right = self.food.x > self.head.x
        food_up = self.food.y < self.head.y
        food_down = self.food.y > self.head.y

        state = [
        # chekcs for mmediate danger to the right, left or ahead
        danger_straight, danger_right, danger_left,

        # body parts
        body_up, body_down, body_left, body_right,

        
        # checks current direction
        self.direction == Direction.LEFT,
        self.direction == Direction.RIGHT,
        self.direction == Direction.UP,
        self.direction == Direction.DOWN,
        # Food location
        food_left, food_right, food_up, food_down
        ]
        
        values = mapvalues + state
        return np.array(values, dtype=np.float32) 
    
    #randomly places food. 
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    #The game engine, works like the openAI gym env. 
    def step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        #Moves the snake, when called updates direction, position, score and adds to the body when food is eaten.  
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        terminated = False
        truncated = False
        if self.is_collision():
            terminated = True
            reward = -20
        elif self.frame_iteration > 100 * len(self.snake):
            truncated = True
            reward = -20
        
        # Check if the snake has eaten food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()  # Place new food
        else:
            self.snake.pop()  # Remove the last segment if no food is eaten
            #reward = -0.1
        
        self._update_ui() #updates the game state/ draws the game on the screen. 
        self.clock.tick(SPEED) #regulates the speed. 
        
        observation = self.get_state()
        info = {}
        return observation, reward, terminated, truncated, info #this return shows the game state, displays reward, terminated or truncated? etc. 

    #checks for collisons, determines if the snakes head hits walls or body. 
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    #Redraws game elemets. clears the screen, draws each segment. Renders the score using the font choosen. 
    def _update_ui(self):

        #Background rendering
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(0, self.h, BLOCK_SIZE):
                color = GREEN1 if (x // BLOCK_SIZE + y // BLOCK_SIZE) % 2 == 0 else GREEN2
                pygame.draw.rect(self.display, color, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))

        #Snake rendering
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        #Score render.
        text = font.render("Score: " + str(self.score), True, LICORICE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    #Updates the snakes direction and head position based on action. Similar to get_next_posistion.
    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 0:  # Straight
            new_dir = clock_wise[idx]
        elif action == 1:  # Right turn
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # Left turn
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)