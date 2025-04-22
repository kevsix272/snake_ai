import numpy as np 
import random
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
from apple_game import SnakeEnv  # Make sure your environment uses the updated get_state()

# Define DQN model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, h2_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.out = nn.Linear(h2_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# Experience Replay Memory
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# Deep Q-Learning for Snake
class SnakeDQL():
    # Hyperparameters
    num_states = 21  
    num_actions = 3  # Actions: 0=Straight, 1=Right, 2=Left
    learning_rate_a = 0.0005
    discount_factor_g = 0.95
    network_sync_rate = 100
    replay_memory_size = 50_000
    mini_batch_size = 128
    epsilon_min = 0.01
    epsilon_decay = 0.995

    loss_fn = nn.SmoothL1Loss()
    optimizer = None

    ACTIONS = ['S', 'L', 'R']

    def train(self, episodes, render=False):
        env = SnakeEnv()
        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        # Create new policy and target networks with input size 13
        policy_dqn = DQN(in_states=self.num_states, h1_nodes=256, h2_nodes=128, out_actions=self.num_actions)
        target_dqn = DQN(in_states=self.num_states, h1_nodes=256, h2_nodes=128, out_actions=self.num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        scores_per_episode = []
        epsilon_history = []
        losses = []
        steps_per_episode = []
        step_count = 0
      
        for episode in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False
            episode_rewards = 0.0
            episode_steps = 0

            while not terminated and not truncated:
                if random.random() < epsilon:
                    action = random.choice(env.action_space)
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()
                episode_steps += 1
                step_count += 1

                new_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                episode_rewards += reward 
                
                if step_count % 1000 == 0:
                    print(f"Episode {episode} Step {step_count}, Reward: {reward}, Score: {env.score}")

            print(f"Episode {episode}| Total reward: {episode_rewards}")
            scores_per_episode.append(episode_rewards)
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            steps_per_episode.append(episode_steps)
            epsilon_history.append(epsilon)

            if step_count % self.network_sync_rate == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())

            if len(memory) >= self.mini_batch_size:
                loss = self.optimize(memory.sample(self.mini_batch_size), policy_dqn, target_dqn)
                losses.append(loss)
            else:
                losses.append(None)

        # Save the trained policy (delete or rename previous file if needed)
        torch.save(policy_dqn.state_dict(), "snake_dql_apples.pt")
        # (Plotting code omitted for brevity)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            if terminated: 
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        [reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state)).max().item()]
                    )

            current_q = policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            target_q = target_dqn(self.state_to_dqn_input(state))
            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def state_to_dqn_input(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32)

    def test(self, episodes):
        env = SnakeEnv()
        policy_dqn = DQN(in_states=self.num_states, h1_nodes=256, h2_nodes=128, out_actions=self.num_actions)
        policy_dqn.load_state_dict(torch.load("snake_dql_apples.pt"))
        policy_dqn.eval()

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False

            while not terminated and not truncated:
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)
        env.close()

if __name__ == '__main__':
    snake_dql = SnakeDQL()
    snake_dql.train(4000)
    snake_dql.test(10)
