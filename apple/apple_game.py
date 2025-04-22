import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('./fonts/arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors
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
SPEED = 200

class SnakeEnv():
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.action_space = [0, 1, 2]  # 0: Straight, 1: Right, 2: Left
        self.reset()

    def reset(self):
        # Initialize game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.frame_iteration = 0

        # Instead of one food, we use a list of foods (3 apples)
        self.foods = []
        for _ in range(3):
            self._place_food()

        return self.get_state(), {}

    def _place_food(self):
        #Places a food at a random location not occupied by the snake or another food.
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        new_food = Point(x, y)
        if new_food in self.snake or new_food in self.foods:
            self._place_food()
        else:
            self.foods.append(new_food)

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

    def get_state(self):
        # Danger near
        danger_straight = self.is_collision(self._get_next_position(0))
        danger_right = self.is_collision(self._get_next_position(1))
        danger_left = self.is_collision(self._get_next_position(2))

        #Checks for body parts
        body_up = Point(self.head.x, self.head.y - BLOCK_SIZE) in self.snake[1:]
        body_down = Point(self.head.x, self.head.y + BLOCK_SIZE) in self.snake[1:]
        body_left = Point(self.head.x - BLOCK_SIZE, self.head.y) in self.snake[1:]
        body_right = Point(self.head.x + BLOCK_SIZE, self.head.y) in self.snake[1:]

        #Existiing direction and food checks
        food_left = any(food.x < self.head.x for food in self.foods)
        food_right = any(food.x > self.head.x for food in self.foods)
        food_up = any(food.y < self.head.y for food in self.foods)
        food_down = any(food.y > self.head.y for food in self.foods)


        # For each apple, calculate normalized relative positions.
        apple_states = []
        for food in self.foods:
            # Normalization divides by the screen width/height so that the values are in roughly [-1, 1]
            rel_x = (food.x - self.head.x) / self.w
            rel_y = (food.y - self.head.y) / self.h
            apple_states.extend([rel_x, rel_y])

        #State Array
        state = [

        # Danger directions
        danger_straight, danger_right, danger_left,

        # Body parts
        body_up, body_down, body_left, body_right,

        # Current direction 
        self.direction == Direction.LEFT,
        self.direction == Direction.RIGHT,
        self.direction == Direction.UP,
        self.direction == Direction.DOWN,

        # Food location
        food_left, food_right, food_up, food_down
        ]
        state.extend(apple_states)
        return np.array(state, dtype=np.float32)

    def step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

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

        # Check if the snake has eaten any apple
        if self.head in self.foods:
            self.score += 1
            reward = 10
            self.foods.remove(self.head)  # Remove the eaten apple
            self._place_food()            # Place a new apple to keep 3 apples on the board
        else:
            self.snake.pop()  # Remove the tail if no food eaten
            reward = -0.1

        self._update_ui()
        self.clock.tick(SPEED)
        observation = self.get_state()
        info = {}
        return observation, reward, terminated, truncated, info

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        
        # Check collision with walls
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        # Check collision with itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        #Bakgrundsrendering
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(0, self.h, BLOCK_SIZE):
                color = GREEN1 if (x // BLOCK_SIZE + y // BLOCK_SIZE) % 2 == 0 else GREEN2
                pygame.draw.rect(self.display, color, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # Draw all apples
        for food in self.foods:
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, LICORICE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

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
