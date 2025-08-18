#
# model_based_rl.py
#
# A simple implementation of a Model-Based Reinforcement Learning agent
# using the Dyna-Q algorithm in a grid world environment.
#
# The agent learns a model of the environment and uses it to plan
# and update its Q-values, making it more sample-efficient than
# a pure model-free approach.
#

import random
import matplotlib.pyplot as plt

# --- 1. Environment: A Simple Grid World ---
class GridWorld:
    """
    A simple 2D grid world environment.
    The agent can move up, down, left, or right.
    Reaching the goal state (8, 12) gives a positive reward.
    """
    def __init__(self):
        # Define the grid dimensions and goal state
        self.rows = 16
        self.cols = 16
        self.goal_state = (8, 12)
        self.start_state = (0, 0)
        self.current_state = self.start_state
        self.is_terminal = False

    def reset(self):
        """Resets the agent to the start state."""
        self.current_state = self.start_state
        self.is_terminal = False
        return self.current_state

    def step(self, action):
        """
        Takes a step in the environment based on the given action.
        Returns the new state, reward, and a flag indicating if the episode is done.
        """
        if self.is_terminal:
            return self.current_state, 0, True

        row, col = self.current_state
        reward = -1  # Default small negative reward for each step

        # Determine the next state based on the action
        if action == "up":
            next_row, next_col = max(0, row - 1), col
        elif action == "down":
            next_row, next_col = min(self.rows - 1, row + 1), col
        elif action == "left":
            next_row, next_col = row, max(0, col - 1)
        elif action == "right":
            next_row, next_col = row, min(self.cols - 1, col + 1)
        else:
            raise ValueError("Invalid action")

        self.current_state = (next_row, next_col)

        # Check if the new state is the goal
        if self.current_state == self.goal_state:
            reward = 100  # Large positive reward for reaching the goal
            self.is_terminal = True
            #print("Goal Reached!")

        return self.current_state, reward, self.is_terminal

# --- 2. Agent: A Model-Based Dyna-Q Agent ---
class ModelBasedAgent:
    """
    Implements a Dyna-Q agent that learns a model of the environment.
    It combines real experience with simulated experience for learning.
    """
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1, num_planning_steps=50):
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.num_planning_steps = num_planning_steps

        # Q-table to store state-action values
        self.q_table = {}

        # The learned model of the environment.
        # It maps (state, action) -> (next_state, reward)
        self.model = {}
        
        # Data for plotting
        self.path_history = []
        self.q_value_history = []
        self.target_state_action = ((0, 0), "right") # Choose a specific SA pair to track

    def get_q_value(self, state, action):
        """Retrieves a Q-value, initializing it to 0 if it doesn't exist."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return random.choice(self.actions)
        else:
            # Exploit: choose the best action based on Q-values
            q_values = {action: self.get_q_value(state, action) for action in self.actions}
            max_q = max(q_values.values())
            # Handle cases with multiple actions having the max Q-value
            best_actions = [action for action, q in q_values.items() if q == max_q]
            return random.choice(best_actions)

    def update_q_value(self, state, action, next_state, reward):
        """
        Updates the Q-value for a given state-action pair using the
        Temporal Difference (TD) learning rule.
        """
        current_q = self.get_q_value(state, action)
        
        # Find the max Q-value for the next state
        if next_state not in [s for s, _ in self.q_table.keys()]:
            # Next state is new, so no Q-values to consider
            max_next_q = 0
        else:
            max_next_q = max([self.get_q_value(next_state, a) for a in self.actions])

        # TD Update Rule
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # Track the Q-value for plotting
        if (state, action) == self.target_state_action:
            self.q_value_history.append(self.get_q_value(state, action))

    def learn_from_experience(self, state, action, next_state, reward):
        """
        The main learning loop. This is the model-based part.
        1. Learn from real experience to update the model.
        2. Use the model for planning to update Q-values.
        """
        # Step 1: Update Q-value based on real-world experience (TD-learning)
        self.update_q_value(state, action, next_state, reward)

        # Step 2: Update the internal model with the new experience
        self.model[(state, action)] = (next_state, reward)

        # Step 3: Planning - Use the model for simulated experience
        # This is the "model-based" part that separates it from model-free
        for _ in range(self.num_planning_steps):
            # Choose a random state-action pair from the model's memory
            if not self.model:
                break
            
            # Select a random state-action pair that has been experienced
            s_rand, a_rand = random.choice(list(self.model.keys()))
            
            # Use the model to predict the outcome of this action
            s_next_rand, r_rand = self.model[(s_rand, a_rand)]
            
            # Update the Q-value for the simulated experience
            self.update_q_value(s_rand, a_rand, s_next_rand, r_rand)

# --- 3. Main Training Loop ---
def train(agent, env, num_episodes):
    """
    The main training function.
    """
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        # Reset data for plotting at the start of each episode
        agent.path_history = [state]
        if episode == 0:  # Only track Q-values for the first episode to avoid a messy plot
            agent.q_value_history = []
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Record the new state for plotting the path
            agent.path_history.append(next_state)
            
            # Learn from this one real experience and then perform planning
            agent.learn_from_experience(state, action, next_state, reward)
            
            total_reward += reward
            state = next_state
        
#print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        
    # --- 4. Plotting the results ---
    # Plot the agent's path in the final episode
    path_rows = [s[0] for s in agent.path_history]
    path_cols = [s[1] for s in agent.path_history]
    
    plt.figure(figsize=(6, 6))
    plt.plot(path_cols, path_rows, marker='o', linestyle='-', color='blue')
    plt.plot(path_cols[0], path_rows[0], 'go', markersize=10, label='Start')
    plt.plot(path_cols[-1], path_rows[-1], 'ro', markersize=10, label='Goal')
    plt.text(path_cols[0], path_rows[0], '  Start', verticalalignment='bottom')
    plt.text(path_cols[-1], path_rows[-1], '  Goal', verticalalignment='bottom')
    
    plt.title('Agent Path in the Final Episode')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.grid(True)
    plt.gca().set_xticks(range(env.cols))
    plt.gca().set_yticks(range(env.rows))
    plt.gca().invert_yaxis()  # Invert y-axis to match typical grid orientation
    plt.legend()
    plt.savefig('agent_path.png')
    plt.show()

    # Plot the Q-value for a specific state-action pair over the first episode
    plt.figure()
    plt.plot(agent.q_value_history, marker='o', linestyle='-')
    plt.title(f'Q-Value for State {agent.target_state_action[0]} & Action {agent.target_state_action[1]}')
    plt.xlabel('Planning Step')
    plt.ylabel('Q-Value')
    plt.grid(True)
    plt.savefig('q_value_change.png')
    plt.show()

# --- 5. Run the simulation ---
if __name__ == "__main__":
    # Define the environment and agent
    env = GridWorld()
    actions = ["up", "down", "left", "right"]
    
    # Initialize the agent with Dyna-Q parameters
    # The num_planning_steps is key: more steps mean more learning from the model
    agent = ModelBasedAgent(actions, num_planning_steps=20)
    
    # Train the agent
    print("Starting Model-Based RL Training...")
    train(agent, env, num_episodes=100)
    
    print("\nTraining complete. Plots have been saved as 'agent_path.png' and 'q_value_change.png'.")