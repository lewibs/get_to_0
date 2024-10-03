from model import Agent, TARGET, TRAIN_LOOPS
import random
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

moves = []

def game(model, start_value, max_steps):
    steps = 0
    done = 0
    while done == 0:
        start_value, done = model.step(start_value, steps, max_steps)
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
success_count = 0
fail_count = 0

for i in range(TRAIN_LOOPS):
    print("\n")
    print(f"Game {i}:")
    rand = get_start_number(-50, 50)
    print(f"Starting game at {rand}")
    print(f"Epsilon is {model.epsilon}")
    end_value, steps = game(model, rand, 100)
    if end_value == TARGET:
        success_count = success_count + 1
    else:
        fail_count = fail_count + 1

    print(f"Game reached {end_value} in {steps} steps with an accuracy of {success_count/(success_count+fail_count)}")
    step_counts.append(steps)
    numbers.append(rand)
    noramlized.append(abs(rand/steps))

plot_normalized_values(noramlized)

