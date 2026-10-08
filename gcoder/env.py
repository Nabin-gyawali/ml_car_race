
import pygame
import random
import math

# pygame and car , sensor defaults 

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 700

ROAD_LEFT = 100
ROAD_RIGHT = 500
ROAD_TOP = 50
ROAD_BOTTOM = 650

ROAD_WIDTH = ROAD_RIGHT - ROAD_LEFT
ROAD_HEIGHT = ROAD_BOTTOM - ROAD_TOP

CAR_W = 30
CAR_H = 40

OBSTACLE_W = 30
OBSTACLE_H = 30

CAR_SPEED_Y = 15
CAR_SPEED_X = 20

SENSOR_RANGE = 150
SENSOR_WIDTH = 80

FPS = 2



def rect_from_center(cx, cy, width, height):
    return pygame.Rect(
        int(cx - width / 2),
        int(cy - height / 2),
        width,
        height
    )






class Car:

    def __init__(self):
        self.x = (ROAD_LEFT + ROAD_RIGHT) / 2
        self.y = ROAD_TOP + 70

        self.width = CAR_W
        self.height = CAR_H

    @property
    def rect(self):
        return rect_from_center(
            self.x,
            self.y,
            self.width,
            self.height
        )

    def move(self, action):

        # 0 = LEFT
        # 1 = STRAIGHT
        # 2 = RIGHT

        if action == 0:
            self.x -= CAR_SPEED_X

        elif action == 2:
            self.x += CAR_SPEED_X

        # Every action moves the car forward.
        self.y += CAR_SPEED_Y

        # Keep the entire car inside the road.
        half_w = self.width / 2

        self.x =  max(ROAD_LEFT + half_w, min(ROAD_RIGHT - half_w, self.x))
        

class Obstacle:

    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.width = OBSTACLE_W
        self.height = OBSTACLE_H

    @property
    def rect(self):
        return rect_from_center(
            self.x,
            self.y,
            self.width,
            self.height
        )




class CarWorld:

    def __init__(self , render_mode = None , random_seed = None):

        self.car = Car()
        self.obstacles = []
        self.window = None
        self.spawn_obstacles()
        self.render_mode = render_mode
        self.clock = None
        self.finished = False
        self.crashed = False
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)



    def spawn_obstacles(self):


        self.obstacles.clear()

        for _ in range(10):

            while True:

                x = random.randint(
                    ROAD_LEFT + OBSTACLE_W // 2,
                    ROAD_RIGHT - OBSTACLE_W // 2
                )

                y = random.randint(
                    self.car.rect.bottom + OBSTACLE_H // 2,
                    ROAD_BOTTOM - OBSTACLE_H // 2
                )

                new_obstacle = Obstacle(x, y)

                collision = any(
                    new_obstacle.rect.colliderect(ob.rect)
                    for ob in self.obstacles
                )

                if not collision:
                    self.obstacles.append(new_obstacle)
                    break


    # --------------------------------------------------------
    # Sensor geometry
    # --------------------------------------------------------

    def get_sensor_rects(self):

        car = self.car

        # Sensor starts at the bottom edge of the car.
        sensor_top = car.rect.bottom

        sensor_bottom = sensor_top + SENSOR_RANGE

        # Three horizontal sensor zones.
        center_x = car.x

        left_rect = pygame.Rect(
            int(center_x - SENSOR_WIDTH * 1.5),
            int(sensor_top),
            SENSOR_WIDTH,
            SENSOR_RANGE
        )

        straight_rect = pygame.Rect(
            int(center_x - SENSOR_WIDTH / 2),
            int(sensor_top),
            SENSOR_WIDTH,
            SENSOR_RANGE
        )

        right_rect = pygame.Rect(
            int(center_x + SENSOR_WIDTH / 2),
            int(sensor_top),
            SENSOR_WIDTH,
            SENSOR_RANGE
        )

        return left_rect, straight_rect, right_rect

    # --------------------------------------------------------
    # Find nearest obstacle in each sensor
    # --------------------------------------------------------

    def get_sensor_distances(self):

        sensor_rects = self.get_sensor_rects()

        distances = [math.inf, math.inf, math.inf]

        for i, sensor in enumerate(sensor_rects):

            for obstacle in self.obstacles:

                obstacle_rect = obstacle.rect

                # Sensor must see only obstacles in front.
                if obstacle_rect.bottom < self.car.rect.bottom:
                    continue

                # Check whether the obstacle overlaps this sensor.
                if sensor.colliderect(obstacle_rect):

                    # Distance from car's front to obstacle's front.
                    distance = (
                        obstacle_rect.top
                        - self.car.rect.bottom
                    )

                    distance = max(0, distance)

                    distances[i] = min(
                        distances[i],
                        distance
                    )

        return distances

    # --------------------------------------------------------
    # Discretize distances
    # --------------------------------------------------------

    def get_discrete_state(self):

        distances = self.get_sensor_distances()

        state = []

        for distance in distances:

            if distance <= SENSOR_RANGE // 2:
                state.append(0)  # CLOSE
            else:
                state.append(1)  # FAR

        return state



    # --------------------------------------------------------
    # Update world
    # --------------------------------------------------------

    def step(self, action):
        # action implementation
        self.car.move(action)

        # Collision detection
        for obstacle in self.obstacles:

            if self.car.rect.colliderect(
                obstacle.rect
            ):
                self.crashed = True
    


        # Finish
        if self.car.rect.top >= ROAD_BOTTOM:
            self.finished = True

         # reward calculation
        reward = 0

        if self.crashed:
            reward = -10
        elif self.finished:
            reward = 100
        else:
            reward = 1  # Reward for moving forward

        return self.get_discrete_state(),reward,self.finished,self.crashed,{}
    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self):

        self.car = Car()

        self.finished = False
        self.crashed = False

        # self.spawn_obstacles()
        info = {}
        return self.get_discrete_state() , info

    def render(self):
        
        if self.render_mode != "human":
            return 

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT)
            ) 
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        # Handle events
        for event in pygame.event.get():
            # if q is pressed quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    print("Quitting the game.")
                    exit()

        # clear the window
        self.window.fill((0, 0, 0))

        # Draw road
        pygame.draw.rect(
            self.window,
            (50, 50, 50),
            (ROAD_LEFT, ROAD_TOP, ROAD_WIDTH, ROAD_HEIGHT)
        )

        # Draw car
        pygame.draw.rect(
            self.window,
            (0, 255, 0),
            self.car.rect
        )

        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(
                self.window,
                (255, 0, 0),
                obstacle.rect
            )
        
        # Draw sensors rectangles transparent and seperated by lines 
        sensor_rects = self.get_sensor_rects()
        for sensor in sensor_rects:
            pygame.draw.rect(
                self.window,
                (0, 0, 255, 100),
                sensor,
                2
            )

        # text for sensor states
        sensor_states = self.get_discrete_state()
        sensor_texts = [
            "Left: " + ("CLOSE" if sensor_states[0] == 0 else "FAR"),
            "Straight: " + ("CLOSE" if sensor_states[1] == 0 else "FAR"),
            "Right: " + ("CLOSE" if sensor_states[2] == 0 else "FAR"),
        ]
        action_text = "Action: " + ("LEFT" if self.car.x < (ROAD_LEFT + ROAD_RIGHT) / 2 else "RIGHT" if self.car.x > (ROAD_LEFT + ROAD_RIGHT) / 2 else "STRAIGHT")
        sensor_texts.append(action_text)
        for i, text in enumerate(sensor_texts):
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.window.blit(text_surface, (10, 10 + i * 40))

        pygame.display.flip()
        self.clock.tick(FPS)





if __name__ == "__main__":
        # initialize the car world
    car_world = CarWorld(render_mode="human")
    initial_obs , info = car_world.reset()
    print("Initial Observation:", initial_obs)
    # render the initial state
    while True:

        action = random.choice([0, 1, 2])

        observation, reward, finished, crashed, info = car_world.step(action)
        if finished or crashed:
            break

        car_world.render()