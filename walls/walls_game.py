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
SPEED = 20

class SnakeEnv():
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.action_space = [0, 1, 2]  # Straight, Left, Right
        self.reset()

        #load snake head image
        # self.head_img = pygame.image.load('snake_img.png').convert_alpha()
        # self.head_img = pygame.transform.scale(self.head_img, (BLOCK_SIZE, BLOCK_SIZE))

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self.walls = set()
        self._place_food()
        self.frame_iteration = 0
        return self.get_state(), {}

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            new_food = Point(x, y)
            if new_food not in self.snake and new_food not in self.walls:
                self.food = new_food
                break

    def _place_wall(self):
        attempts = 0
        while attempts < 100:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            new_wall = Point(x, y)
            if new_wall not in self.snake and new_wall != self.food and not self._is_adjacent_to_wall(new_wall):
                self.walls.add(new_wall)
                break
            attempts += 1

    def _is_adjacent_to_wall(self, point):
        adjacent_offsets = [
            (-BLOCK_SIZE, 0), (BLOCK_SIZE, 0),
            (0, -BLOCK_SIZE), (0, BLOCK_SIZE),
            (-BLOCK_SIZE, -BLOCK_SIZE), (-BLOCK_SIZE, BLOCK_SIZE),
            (BLOCK_SIZE, -BLOCK_SIZE), (BLOCK_SIZE, BLOCK_SIZE)
        ]
        for dx, dy in adjacent_offsets:
            if Point(point.x + dx, point.y + dy) in self.walls:
                return True
        return False

    def _get_next_position(self, action):
        # Simulera riktning utan att ändra den nuvarande riktningen
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if action == 0:  # Fortsätt rakt fram
            new_dir = clock_wise[idx]
        elif action == 1:  # Höger vändning
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # Vänster vändning
            new_dir = clock_wise[(idx - 1) % 4]
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
        food_left = self.food.x < self.head.x
        food_right = self.food.x > self.head.x
        food_up = self.food.y < self.head.y
        food_down = self.food.y > self.head.y

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
        return np.array(state, dtype=np.float32)

    def step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)
        reward = -0.1
        terminated = False
        truncated = False

        if self.is_collision():
            terminated = True
            reward = -20
        elif self.frame_iteration > 100 * len(self.snake):
            truncated = True
            reward = -20

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()  # Placera ny mat
            self._place_wall()  # Placera en ny vägg
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return self.get_state(), reward, terminated, truncated, {}

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:] or pt in self.walls:
            return True
        return False

    def _update_ui(self):
        #Bakgrundsrendering
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(0, self.h, BLOCK_SIZE):
                color = GREEN1 if (x // BLOCK_SIZE + y // BLOCK_SIZE) % 2 == 0 else GREEN2
                pygame.draw.rect(self.display, color, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))


        # Ritar ut ormen, score, mat och väggar
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        for wall in self.walls:
            pygame.draw.rect(self.display, DARKGREEN, pygame.Rect(wall.x, wall.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, LICORICE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if action == 0:  # Rakt fram
            new_dir = clock_wise[idx]
        elif action == 1:  # Höger vändning
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # Vänster vändning
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
