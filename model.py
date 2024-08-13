import torch
from collections import deque
import random
import torch.optim as optim
import math

# Q[(s, a)] += alpha * (r + gamma * max(Q[(s', a')]) - Q[(s, a)])

VALUE_MAP = [-1, 1]
EPSILON = 0.70
EPSILON_DECAY = 0.001
MEMORY = 100
BATCH = 10
MAX_STEPS = 50
GAMMA = 0.9
LEARNING_RATE = 0.1
TARGET_UPDATE = 10
GRAD_CLIP = 10

class SkipNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 2)
        self.fc3 = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.softmax(x, dim=1)
        return x

class Agent():
    def __init__(self):
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

    def step(self, number, step): #state action next_state reward done
        state = torch.tensor([[number//abs(number), number]]).float().cpu()

        r = random.random()
        if r < self.epsilon:
            possible_options = [abs(number + value) for value in VALUE_MAP]
            min_index = possible_options.index(min(possible_options))
            output = torch.zeros(1,2)
            output[0][min_index] = 1
        else:
            output = self.policy_net(state)

        delta = self.translate(output)
        delta = torch.tensor([[delta]]).float().cpu()
        next_state = state.clone()
        first_num = state[0][1].item()
        second_num = first_num + delta
        next_state[0][1] = second_num
        if second_num > 0:
            next_state[0][0] = 0
        else:
            next_state[0][0] = 1

        done = 0
        reward = 0

        if next_state[0][1].item() == 0:
            done = 1
            reward = 100
            for i, memory in enumerate(reversed(self.current_memory)):
                memory[3] += 1 / (i+1)
        elif step > MAX_STEPS:
            done = 1
            reward = -100
            for i, memory in enumerate(reversed(self.current_memory)):
                memory[3] -= 1 / (i+1)
        else:
                distance_old = abs(first_num)
                distance_new = abs(second_num)

                if distance_new < distance_old:
                    # Reward the agent for getting closer to 0
                    reward = math.tanh(1 / (distance_new + 1e-6))
                else:
                    # Punish the agent for getting further away from 0
                    reward = -math.tanh(distance_new)

                reward *= 5

        done = torch.tensor([[done]]).cpu().int()
        reward = torch.tensor([[reward]]).float().cpu()
        self.current_memory.append([state, output, next_state, reward, done])

        if done.item() == 1:
            self.decay_epsilon()
            if step % TARGET_UPDATE == 0:
                self.update_target_net()
            self.memory.extend(self.current_memory)
            self.current_memory = []

        self.train()
        return int(next_state[0][1].item()), done.item()

    def train(self):
        if len(self.memory) < BATCH:
            return  # Not enough samples to train
        sample = random.sample(self.memory, BATCH)
        state, action, next_state, reward, done = zip(*sample)


        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state))).cpu().bool()
        non_final_next_states = torch.cat([s for s in next_state if s is not None])
        state_batch = torch.cat(state).squeeze()
        action_batch = torch.cat(action).squeeze()
        reward_batch = torch.cat(reward).squeeze()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.long())

        next_state_values = torch.zeros(BATCH).cpu()
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

    def translate(self, model_output):
        max_index = torch.argmax(model_output)
        return VALUE_MAP[max_index.item()]


