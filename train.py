from model import Agent
import random
import matplotlib.pyplot as plt

TRAIN_LOOPS = 1000

model = Agent()

def plot_normalized_values(normalized, title='Normalized Values Plot', xlabel='Index', ylabel='Normalized Value'):

    normalized_values = normalized

    # Create a figure and axis
    plt.figure()

    # Plot the normalized values
    plt.plot(normalized_values, marker='o', linestyle='-', color='b', label='Normalized Values')

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()

def game(model, start_value):
    steps = 0
    done = 0
    while done == 0:
        start_value, done = model.step(start_value, steps)
        steps += 1
    
    return start_value, steps

def get_start_number(start, end):
    rand = random.randint(start,end)
    if rand == 0:
        return get_start_number(start, end)
    return rand

step_counts = []
numbers = []
noramlized = []

for i in range(TRAIN_LOOPS):
    print("\n")
    print(f"Game {i}:")
    rand = get_start_number(-10, 10)
    print(f"Starting game at {rand}")
    print(f"Epsilon is {model.epsilon}")
    end_value, steps = game(model, rand)
    print(f"Game reached {end_value} in {steps} steps")
    step_counts.append(steps)
    numbers.append(rand)
    noramlized.append(abs(rand/steps))

plot_normalized_values(step_counts)

