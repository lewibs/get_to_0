import torch
from collections import deque
import random
import torch.optim as optim
import math

# Q[(s, a)] += alpha * (r + gamma * max(Q[(s', a')]) - Q[(s, a)])

VALUE_MAP = [-2, 3]
EPSILON = 0.90
EPSILON_DECAY = 0.001
MEMORY = 5000
BATCH = 100
GAMMA = 0.5
LEARNING_RATE = 0.5
TARGET_UPDATE = 10
GRAD_CLIP = 10
TARGET = 0
TRAIN_LOOPS = 500

class SkipNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 1)  # Input layer to hidden layer
        self.fc2 = torch.nn.Linear(1, 1)
        self.fc3 = torch.nn.Linear(1, 2)


    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x)) 
        return x

class Agent():
    def __init__(self):
        # what is the point of having two memories?
        self.current_memory = []
        self.memory = deque(maxlen=MEMORY)
        self.policy_net = SkipNN()
        self.target_net = SkipNN()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.epsilon = EPSILON

    def decay_epsilon(self):
        self.epsilon = max(0, self.epsilon - EPSILON_DECAY)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def step(self, number, step, max_steps): #state action next_state reward done
        state = torch.tensor([number]).float().cpu()

        r = random.random()
        if r < self.epsilon:
            #TODO this may be wrong... look into how this works.
            possible_options = [abs(number + value) for value in VALUE_MAP]
            min_index = possible_options.index(min(possible_options))
            output = torch.zeros(2)
            output[min_index] = 1
        else:
            output = self.policy_net(state)

        delta = self.translate(output)
        next_state = state.clone()
        first_num = state.clone()
        next_state = first_num + delta

        done = 0
        reward = 0

        if next_state.item() == TARGET:
            done = 1
            reward = 10
        elif step > max_steps:
            done = 1
            reward = 0
        else:
            next_diff = abs(TARGET - next_state.item())
            current_diff = abs(TARGET - state.item())

            if current_diff > next_diff:
                reward = 1
            else:
                reward = -1

        done = torch.tensor([[done]]).cpu().int()
        reward = torch.tensor([[reward]]).float().cpu()
        self.current_memory.append([state, torch.argmax(output), next_state, reward, done])

        if done.item() == 1:
            self.decay_epsilon()
            if step % TARGET_UPDATE == 0:
                self.update_target_net()
            self.memory.extend(self.current_memory)
            self.current_memory = []

        self.train()
        return int(next_state.item()), done.item()

    def train(self):
        if len(self.memory) < BATCH:
            return  # Not enough samples to train

        # Sample a batch of experiences from memory
        sample = random.sample(self.memory, BATCH)
        state, action, next_state, reward, done = zip(*sample)

        # Convert lists to tensors
        state_batch = torch.cat(state).reshape(BATCH, 1)
        action_batch = torch.stack(action)
        next_state_batch = torch.stack(next_state)
        reward_batch = torch.cat(reward)

        # Compute the Q values from the current policy
        policy_action_values = self.policy_net(state_batch)

        # Initialize target Q values
        target_next_action_batch = self.target_net(next_state_batch).max(1).values  # Use max action values for next state
        expected_state_action_values = reward_batch + (GAMMA * target_next_action_batch)  # Calculate expected Q values

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(policy_action_values.gather(1, action_batch.unsqueeze(1)), expected_state_action_values.unsqueeze(1))  # Gather Q values for actions taken

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

    def translate(self, model_output):
        delta = 0
        if torch.sum(model_output) != 0:
            max_index = torch.argmax(model_output)
            delta = VALUE_MAP[max_index.item()]

        return torch.tensor([[delta]]).float().cpu()


