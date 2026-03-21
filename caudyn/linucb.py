'''
Algorithm: LinUCB (Linear Upper Confidence Bound)
Inputs:
  α : Exploration parameter (a strictly positive scalar)
  d : Dimension of the context feature vector

Initialization:
  FOR each arm a ∈ A:
      A_a = Identity Matrix of size d × d, Covariance Matrix for arm $a$
      b_a = Zero Vector of size d × 1, The Reward Vector for arm $a$
  END FOR

Execution Loop:
  FOR each time step t = 1, 2, ..., T:
      
      1. Observe the user context vector x_t (size d × 1)
      
      2. FOR each available arm a ∈ A:
          // Calculate the learned weights (Ridge Regression)
          // The Causal Weights for arm $a$
          θ_a = Inverse(A_a) * b_a
          
          // Calculate the expected reward (Exploitation)
          expected_reward = Transpose(θ_a) * x_t
          
          // Calculate the confidence bound (Exploration)
          confidence_bound = α * sqrt( Transpose(x_t) * Inverse(A_a) * x_t )
          
          // Calculate the Upper Confidence Bound (UCB)
          p_a = expected_reward + confidence_bound
      END FOR
      
      3. Choose arm a_t = argmax(p_a)  // Break ties randomly
      
      4. Execute action a_t and observe the actual reward r_t
      
      5. Update the matrices for the chosen arm a_t:
          A_{a_t} = A_{a_t} + (x_t * Transpose(x_t))
          b_{a_t} = b_{a_t} + (r_t * x_t)
          
  END FOR
'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import UberMarketplaceEnvironment # Importing our Phase 1 world

class LinUCBAgent:
    def __init__(self, n_actions, n_features, alpha=1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha # The Exploration parameter
        
        # Initialize the A matrix (covariance) and b vector (rewards) for each arm
        # A is initialized as an Identity matrix (Ridge regression prior)
        self.A = [np.identity(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]
        
    def _get_context_vector(self, state_dict):
        """
        Converts the state dictionary into a normalized numpy vector.
        Normalization is crucial for LinUCB so the exploration parameter (alpha)
        scales evenly across all features.
        """
        return np.array([
            1.0, # Intercept bias term
            state_dict['recency'] / 30.0,
            state_dict['frequency'] / 20.0,
            state_dict['weather_active'],
            state_dict['surge_multiplier'] / 3.0
        ])

    def choose_action(self, state_dict):
        x = self._get_context_vector(state_dict)
        p = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            # Calculate A inverse
            A_inv = np.linalg.inv(self.A[a])
            
            # Calculate theta (the learned causal weights for this specific arm)
            theta_a = A_inv.dot(self.b[a])
            
            # Calculate the expected reward (Exploitation)
            expected_reward = theta_a.dot(x)
            
            # Calculate the confidence bound (Exploration)
            exploration_bonus = self.alpha * np.sqrt(x.dot(A_inv).dot(x))
            
            # Upper Confidence Bound
            p[a] = expected_reward + exploration_bonus
            
        # Break ties randomly, otherwise pick the max UCB
        # np.isclose is used to handle floating point precision ties
        max_ucb = np.max(p)
        best_actions = np.where(np.isclose(p, max_ucb))[0]
        return np.random.choice(best_actions)

    def update(self, action, state_dict, reward):
        """Updates the covariance matrix and reward vector after observing reality."""
        x = self._get_context_vector(state_dict)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def get_learned_weights(self, action):
        """Helper to peek into the Bandit's brain."""
        A_inv = np.linalg.inv(self.A[action])
        return A_inv.dot(self.b[action])

# =====================================================================
# THE ONLINE SIMULATION RUNNER
# =====================================================================
if __name__ == "__main__":
    env = UberMarketplaceEnvironment(seed=100)
    
    # 3 Actions (0%, 10%, 20%), 5 Features (Intercept, Recency, Freq, Weather, Surge)
    agent = LinUCBAgent(n_actions=3, n_features=5, alpha=0.5)
    
    n_steps = 15000
    cumulative_rewards = 0
    reward_history = []
    cumulative_regret = 0
    regret_history = []
    mse_history = []
    
    print(f"1. Launching LinUCB Online Simulation for {n_steps} riders...")
    
    state = env.reset()
    for step in range(n_steps):
        # 1. Agent chooses action based on Context
        action = agent.choose_action(state)
        
        # =========================================================
        # THE ORACLE (Ground Truth Evaluation)
        # We calculate the true probabilities for ALL arms to grade the Bandit
        # =========================================================
        true_probs = [env._calculate_true_conversion(state, a) for a in range(agent.n_actions)]
        
        # Oracle's best action and reward
        optimal_prob = np.max(true_probs)
        
        # Bandit's chosen reward
        chosen_prob = true_probs[action]
        
        # Metric 1: Regret (How much conversion did we lose by not being perfect?)
        step_regret = optimal_prob - chosen_prob
        cumulative_regret += step_regret
        regret_history.append(cumulative_regret)
        
        # Metric 2: Prediction Error (How far off was the Bandit's internal math?)
        x = agent._get_context_vector(state)
        bandit_prediction = agent.get_learned_weights(action).dot(x)
        squared_error = (bandit_prediction - chosen_prob)**2
        mse_history.append(squared_error)
        
        # =========================================================
        
        # 2. Environment reacts
        next_state, reward, _ = env.step(action)
        
        # 3. Agent learns from the outcome
        agent.update(action, state, reward)
        
        # Track metrics
        cumulative_rewards += reward
        reward_history.append(cumulative_rewards / (step + 1))
        
        state = next_state
        
        if (step + 1) % 5000 == 0:
            print(f"   -> Step {step + 1}: Average Conversion Rate = {reward_history[-1]:.1%}")
            current_rmse = np.sqrt(np.mean(mse_history[-5000:]))
            print(f"   -> Step {step + 1}:")
            print(f"      Cumulative Regret: {cumulative_regret:.2f}")
            print(f"      Recent Prediction RMSE: {current_rmse:.4f}")

    print("\n2.1. Peeking into the Bandit's Brain (Learned Weights)...")
    feature_names = ["Intercept", "Recency", "Frequency", "Weather", "Surge"]
    
    for a in range(3):
        discount = env.discount_levels[a] * 100
        weights = agent.get_learned_weights(a)
        print(f"\n--- Arm {a} ({discount}% Discount) Learned Weights ---")
        for name, weight in zip(feature_names, weights):
            print(f"  {name:10}: {weight:+.3f}")
            
    print("\n2.2. Evaluation Complete.")
    print("If the RMSE drops significantly between step 5000 and 15000,")
    print("it mathematically proves the Bandit is converging on the true hidden physics.")

    print("\n3. Analysis:")
    print("Look at the 'Frequency' weight across the three arms.")
    print("The Bandit should have learned that giving a 20% discount (Arm 2)")
    print("to a high-frequency user is mathematically worse or negligible compared")
    print("to giving it to a low-frequency user.")
    
    # plot all the metrics in one figure
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(reward_history, label='Average Conversion Rate')
    plt.xlabel('Steps')
    plt.ylabel('Average Conversion')
    plt.title('LinUCB Average Conversion Over Time')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(regret_history, label='Cumulative Regret', color='orange')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('LinUCB Cumulative Regret Over Time')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(mse_history, label='Prediction MSE', color='green')
    plt.xlabel('Steps')
    plt.ylabel('MSE')
    plt.title('LinUCB Prediction MSE Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()