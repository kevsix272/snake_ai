import numpy as np #imports numpy and is used for handling multidimensial arrays. 
import matplotlib.pyplot as plt #For plottning rewards and scores using graphs.
from collections import deque 
import random #Used to randomise numers and values. 
import torch #In this case used to build the neural network, often used for image regenition and other neural networks. 
from torch import nn #build nad train neural network. 
import torch.nn.functional as F #Grants a more deep controll over the neural network. 
from walls_game import SnakeEnv #imports the class SnakeEnv. The agent uses this enviroment in training and testing. 
from IPython import display 

# Define model, gets neural  network module from PyTorch. 
class DQN(nn.Module):
    def __init__(self, in_states=15, h1_nodes=256, h2_nodes=128,out_actions=3): 
        
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes) #fully connected layer  converting 11 inputs to 256 nodes
        self.fc2 = nn.Linear(h1_nodes, h2_nodes) #Another fully connected layer from 256 to 128 nodes. 
        self.out = nn.Linear(h2_nodes, out_actions)#Final layer thet outputs Q-values för 3 actions. 

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x)) # Apply rectified linear unit (ReLU) activation
        return self.out(x)         # Calculate output and return
        

# Define memory for Experience Replay, stores past things done. The agent memory
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition): #Saves new experiences
        self.memory.append(transition)

    def sample(self, sample_size): #Radomly takes a batch of experiences for training. 
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

#This class is responisble for the training and tesing of the DQL-agent. 
class SnakeDQL():
    # Hyperparameters (adjustable)
    num_states = 15                  # number of states in the environment
    num_actions = 3                 # number of actions in the environment
    learning_rate_a = 0.0005         # learning rate (alpha)
    discount_factor_g = 0.95         # discount rate (gamma)    
    network_sync_rate = 100          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 50_000       # size of replay memory
    mini_batch_size = 128            # size of the training data set sampled from the replay memory
    epsilon_min = 0.01               # minimum exploration rate
    epsilon_decay = 0.995            # decay rate for exploration

    # Neural Network
    loss_fn = nn.SmoothL1Loss()           # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer.

    ACTIONS = ['S','L','R']     # for printing 0,1,2 => Straight, Left, Right

    # Train the agent, uses methods defined erlier like reset(), step() and get_state(). 
    def train(self, episodes, render=False):
        # Create Snake instance
        env = SnakeEnv()
        
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size) #transitions are stored in a memory buffer. 
        

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=self.num_states, h1_nodes=256, h2_nodes=128, out_actions=self.num_actions) #Policy network, chooses actions and is updated every training step. 
        target_dqn = DQN(in_states=self.num_states, h1_nodes=256, h2_nodes=128, out_actions=self.num_actions) #Target network, provides stable tareget Q-values and is periodically synchronized with the policy network

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # commentout during training to help with debugging
        # print('Policy (random, before training):')
        # self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        scores_per_episode = []  # Track scores per episode
        rewards_per_episode = []

        # List to keep track of epsilon decay
        epsilon_history = []
        losses = []                         # Track loss per optimization s
        steps_per_episode = []              # Track steps per episode
        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
      
        for episode in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False     
            truncated = False      

            episode_rewards = 0.0
            episode_steps = 0  # Track steps in this episode
       
            while(not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = random.choice(env.action_space) # actions: 0=Straight,1=Left,2=Right
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()
                episode_steps += 1
                step_count += 1

                # Render environment

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count += 1

                episode_rewards += reward 
                if(step_count % 1000 == 0):
                    print(f"episode {episode} step {step_count}, Reward: {reward}, score: {env.score}")
            


            print(f"Episode {episode}| total reward {episode_rewards}")

            rewards_per_episode.append(episode_rewards)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            steps_per_episode.append(episode_steps)
            scores_per_episode.append(env.score)
            epsilon_history.append(epsilon)

            plt.figure(1)
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.clf()
            plt.plot(scores_per_episode)
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.title('Score')

            plt.figure(2)
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.clf()
            plt.plot(rewards_per_episode)
            plt.xlabel('Episode')
            plt.ylabel('reward')
            plt.title('reward')
            plt.show(block=False)
            plt.pause(0.1)

            # Sync policy and target networks
            if step_count % self.network_sync_rate == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                # Optimize and track loss
            if len(memory) >= self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                loss = self.optimize(mini_batch, policy_dqn, target_dqn)
                losses.append(loss)  # Append loss value
            else:
                losses.append(None)  # No optimization occurred, so append None
        # Save policy
        torch.save(policy_dqn.state_dict(), "snake_dql_walls.pt")
        
        plt.figure(1)
        plt.savefig('C:\\Users\\Koo\\Documents\\Kevin\\snake_dql_score.png')
        plt.figure(2)
        plt.savefig('C:\\Users\\Koo\\Documents\\Kevin\\snake_dql_rewards.png')
        plt.show()



    # Optimize policy network, implements the learning step using a mini-batch from replay memory. 
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            if terminated: 
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state)).max()
                    )

            # Get current Q values
            current_q = policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            # Get target Q values
            target_q = target_dqn(self.state_to_dqn_input(state))
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss and optimize
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()  # Return scalar loss value


    def state_to_dqn_input(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32)

    # Run the Snake environment with the learned policy, renders a enviorment using saved data/knowlege, Runs a number of episodes with this knowlege. 
    def test(self, episodes,):
        # Create Snake instance
        env = SnakeEnv()
        num_states = len(env.get_state())
        num_actions= len(env.action_space)
        # num_states = env.observation_space.n
        # num_actions = env.action_space.n
        
        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=256, h2_nodes = 128, out_actions=num_actions) #resets the env, prints Q-values for each action. 
        # policy_dqn = torch.load_state_dict("snake_dql_walls.pt")      # Load model
        
        print(f"num_states: {num_states}")
        policy_dqn.load_state_dict(torch.load("snake_dql_walls.pt", map_location=torch.device('cpu')))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Trained Policy Example Q-values:')
        self.print_dqn(policy_dqn, env)

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent dies 
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it dies(terminated) or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)

        pass

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn, env):
        example_state = env.reset()[0]  #  Pass `env` as an argument
        input_tensor = self.state_to_dqn_input(example_state)
        with torch.no_grad():
            q_values = dqn(input_tensor).tolist()

        print("Example Q-values for initial state:")
        for action, q_value in enumerate(q_values):
            print(f"Action {self.ACTIONS[action]}: {q_value:.2f}")


#Used for the runing training/testing. 
if __name__ == '__main__':
    snake_dql = SnakeDQL()
    snake_dql.train(5000)
    snake_dql.test(10)

    
