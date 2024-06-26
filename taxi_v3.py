import numpy as np
import gymnasium as gym
import random

def generate_explore_exploit_action(Q, obs, epsilon):

    '''
    Computes an action through balancing random exploration
    and exploitation of likely successful transitions. How much random perturbation
    to include is determined by the epsilon hyper-parameter. This parameter is
    currently constant but can also be defined as a function (e.g. exponentially
    decaying towards zero) which might yield improvments for more complex cases.
    '''

    if (random.uniform(0,1) > epsilon):
        # Exploit
        action = np.argmax(Q[obs])
    else:
        # Explore
        action = random.randint(0, env.action_space.n - 1)
    return action

def show_state(step, env, obs, reward):

    '''
    Visulizes the current state of the Taxi-v3 problem.
    '''

    ansi_state = env.render()

    array_state = list(env.unwrapped.decode(obs))

    print(f"Step {step}: {array_state}, Reward: {reward}")

    print(ansi_state)
    return

def train_Q_model(alpha, gamma, epsilon, n_epochs, n_steps):

    '''
    Trains the model using Q-learning according the Bellman equation:
        Q(s,a) = Q(s,a) + alpha [R(s,a) + gamma * max(Q(s',a')) - Q(s,a)]

    where s is the state, a the action, R the reward, alpha learning rate,
    and gamma the discount factor.
    '''

    print("Training Q model...")

    # Initialize Q-table (n_states x n_actions)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for epoch in range(n_epochs):
        obs, info = env.reset()

        for i in range(n_steps):

           # Compute the action
           action = generate_explore_exploit_action(Q, obs, epsilon)

           # Generate the next state
           new_obs, reward, terminated, truncated, info = env.step(action)

           # Update the Q-table according to the Bellman equation
           Q[obs][action] = Q[obs][action] + alpha * (reward + gamma * np.max(Q[new_obs]) - Q[obs][action])

           # Check if the task was succesfully completed before the time limit
           if terminated:
               break

           obs = new_obs

    print("Training successfully completed!\n")

    return Q

def evaluate_model(Q, n_steps, n_tests):

    print("Evaluating model...")

    reward_sum = 0

    for t in range(n_tests):
        print("TEST EXAMPLE ", t + 1)
        obs, info = env.reset()
        show_state(0, env, obs, 0)

        for i in range(n_steps):
            # Compute optimal action according to the trained model
            action = np.argmax(Q[obs][:])
            new_obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward

            # Visualize the taxi trajectory
            show_state(i + 1, env, obs, reward)

            if terminated:
                break

            obs = new_obs

    print("Average sum of rewards: ", reward_sum/n_tests)

    return

# User-defined hyper-parameters
learning_rate = 0.1             # Learning rate (alpha) in the Bellman equation
discount_factor = 0.9           # Discount factor (gamma) in the Bellman equation
explore_exploit_rate = 0.9      # Exploration rate used for determining action (can also be a lambda function)
n_training_epochs = 2000        # Number of training examples
n_training_steps = 1000         # Number of maxmimum training steps for each example
n_test_examples = 4             # Number of test examples to generate

# Initialize the Taxi-v3 environment
env = gym.make("Taxi-v3", render_mode="ansi")
obs, info = env.reset()

# Train the model using Q-learning
model = train_Q_model(learning_rate, discount_factor, explore_exploit_rate, n_training_epochs, n_training_steps)

# Evaluate the trained model and visualize trajectories
evaluate_model(model, n_training_steps, n_test_examples)

# Clean up the environment
env.close()
