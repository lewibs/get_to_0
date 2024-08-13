import torch
from collections import deque
import random
import torch.optim as optim

# Q[(s, a)] += alpha * (r + gamma * max(Q[(s', a')]) - Q[(s, a)])

VALUE_MAP = [1, -1]
EPSILON = 0.3
MEMORY = 100000
BATCH = 100
GAMMA = 0.9
LEARNING_RATE = 0.01

class SkipNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 50)
        self.fc2 = torch.nn.Linear(50, 50)
        self.fc3 = torch.nn.Linear(50, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.softmax(x, dim=1)
        return x

class Agent():
    def __init__(self):
        self.memory = deque(maxlen=MEMORY)
        self.policy_net = SkipNN()
        self.target_net = SkipNN()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = torch.nn.MSELoss()

    def step(self, number): #state action next_state reward done
        if number > 0:
            state = torch.tensor([[0, number]]).float().cpu()
        else:
            state = torch.tensor([[1, number]]).float().cpu()

        r = random.random()
        if r < EPSILON:
            output = torch.rand(1,2)
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

        done = torch.tensor([[0]], dtype=torch.int)
        if next_state[0][0].item() == 0:
            done = torch.tensor([[1]], dtype=torch.int)

        reward = 0
        if abs(first_num) < abs(second_num):
            reward = -1
        else:
            reward = 1
        reward = torch.tensor([[reward]]).float().cpu()
        self.memory.append([state, output, next_state, reward, done])
        try:
            self.train()
        except:
            print("didnt train ran into an issue")

        return int(next_state[0][1].item())

    def train(self):
        if len(self.memory) < BATCH:
            return  # Not enough samples to train

        sample = random.sample(self.memory, BATCH)
        states, actions, next_states, rewards, dones = zip(*sample)

        # Convert lists to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)

        # Compute Q-values for the current states
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + (1 - dones) * GAMMA * next_q_values

        # Compute the loss
        loss = self.criterion(q_values, targets)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def translate(self, model_output):
        max_index = torch.argmax(model_output)
        return VALUE_MAP[max_index.item()]


