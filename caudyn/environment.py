import numpy as np
import pandas as pd

class UberMarketplaceEnvironment:
    def __init__(self, seed=42):
        self.np_random = np.random.RandomState(seed)
        
        # Action Space: 0 = 0% discount, 1 = 10% discount, 2 = 20% discount
        self.discount_levels = [0.0, 0.1, 0.2]
        self.num_actions = len(self.discount_levels)
        
        # This will hold the current user context
        self.current_state = None 

    def _generate_user_context(self):
        """Generates a random rider context and market conditions."""
        recency = self.np_random.randint(1, 31)      # Days since last ride (1-30)
        frequency = self.np_random.randint(1, 21)    # Rides per month (1-20)
        weather_active = self.np_random.binomial(1, 0.2) # 20% chance of rain
        
        # Surge is causally driven by weather
        if weather_active == 1:
            surge_multiplier = self.np_random.uniform(1.5, 2.5)
        else:
            surge_multiplier = self.np_random.uniform(1.0, 1.4)
            
        return {
            'recency': recency,
            'frequency': frequency,
            'weather_active': weather_active,
            'surge_multiplier': round(surge_multiplier, 2)
        }

    def _calculate_true_conversion(self, context, action_idx):
        """The hidden laws of physics. Calculates true P(Convert)."""
        discount_value = self.discount_levels[action_idx]
        
        # 1. Base propensity to book (Without any discount)
        # TWEAKED: High frequency users now have a MUCH higher baseline probability.
        # This exaggerates the selection bias of the legacy system.
        base_prob = 0.2 \
                    + (0.04 * context['frequency']) \
                    - (0.01 * context['recency']) \
                    + (0.10 * context['weather_active']) \
                    - (0.10 * context['surge_multiplier'])
        
        # 2. The True Causal Treatment Effect (CATE)
        # TWEAKED: Discount is completely useless for anyone riding > 5-10 times a month.
        # The penalty for frequency (0.10) heavily outweighs the base lift (0.5).
        treatment_multiplier = 0.5 - (0.10 * context['frequency']) + (0.04 * context['recency'])
        
        # Ensure the multiplier doesn't go negative
        treatment_multiplier = max(0, treatment_multiplier)
        
        treatment_effect = discount_value * treatment_multiplier
        
        # 3. Final Probability
        final_prob = np.clip(base_prob + treatment_effect, 0.0, 1.0)
        return final_prob

    def reset(self):
        """Spawns a new user and returns their context."""
        self.current_state = self._generate_user_context()
        return self.current_state

    def step(self, action_idx):
        """Applies a discount, observes the outcome, and returns the reward."""
        if self.current_state is None:
            raise ValueError("Call reset() before step()")
            
        # Calculate true hidden probability
        true_prob = self._calculate_true_conversion(self.current_state, action_idx)
        
        # Flip a biased coin to see if they actually booked the ride
        reward = self.np_random.binomial(1, true_prob)
        
        # In a Bandit, every step is a new episode. We return the reward and a new state.
        next_state = self.reset()
        
        return next_state, reward, true_prob

    def generate_biased_historical_data(self, num_samples=50000):
        """
        Phase 1 Crucial Step: Simulates a bad legacy algorithm that creates Selection Bias.
        It wrongly gives the biggest discounts to the most loyal users.
        """
        data = []
        for _ in range(num_samples):
            context = self.reset()
            
            # THE BIAS: The legacy system looks at frequency to assign discounts
            if context['frequency'] >= 15:
                # Highly loyal users get the 20% discount (Bad business logic!)
                action = 2 
            elif context['frequency'] >= 7:
                action = 1
            else:
                action = 0
                
            # Add some randomness to the legacy policy so the propensity score isn't perfect 1.0 or 0.0
            if self.np_random.rand() < 0.15:
                action = self.np_random.randint(0, 3)
                
            _, reward, true_prob = self.step(action)
            
            row = context.copy()
            row['treatment'] = action
            row['discount_value'] = self.discount_levels[action]
            row['converted'] = reward
            row['true_prob'] = true_prob # We save this just for our own offline grading later
            data.append(row)
            
        return pd.DataFrame(data)


class UberMarketplaceEnvironmentWithShock(UberMarketplaceEnvironment):
    """
    Extended environment that simulates a macroeconomic shock at a specific step.
    
    Before the shock: Baseline economics (good economy, weak discount effect)
    After the shock: Riders become frugal, base conversion crashes, discount sensitivity skyrockets
    
    This simulates non-stationary environments and tests algorithm adaptability.
    """
    
    def __init__(self, seed=42, shock_step=17500):
        super().__init__(seed=seed)
        
        # Track the current simulation step for shock timing
        self.current_step = 0
        self.shock_step = shock_step  # The exact moment the economy crashes

    def step(self, action_idx):
        """Applies a discount, observes the outcome, and returns the reward."""
        if self.current_state is None:
            raise ValueError("Call reset() before step()")
            
        # Calculate true hidden probability (now aware of the current step)
        true_prob = self._calculate_true_conversion(self.current_state, action_idx)
        
        # Flip a biased coin to see if they actually booked the ride
        reward = self.np_random.binomial(1, true_prob)
        
        # Increment our global clock
        self.current_step += 1
        
        # In a Bandit, every step is a new episode. We return the reward and a new state.
        next_state = self.reset()
        
        return next_state, reward, true_prob

    def _calculate_true_conversion(self, context, action_idx):
        """The hidden laws of physics. Now with a built-in economic shock."""
        discount_value = self.discount_levels[action_idx]
        
        # ==========================================
        # PHASE 1: THE GOOD ECONOMY (Steps 0 to shock_step)
        # ==========================================
        if self.current_step < self.shock_step:
            # Pre-shock: Original baseline economics
            base_prob = 0.2 \
                        + (0.04 * context['frequency']) \
                        - (0.01 * context['recency']) \
                        + (0.10 * context['weather_active']) \
                        - (0.10 * context['surge_multiplier'])
            
            # Discounts are weakly effective, mostly for infrequent riders
            treatment_multiplier = 0.5 - (0.10 * context['frequency']) + (0.04 * context['recency'])
            
        # ==========================================
        # PHASE 2: THE ECONOMIC SHOCK (Steps shock_step+)
        # ==========================================
        else:
            # 1. Softer post-shock baseline: average conversion around 10-15%.
            base_prob = 0.15 \
                        + (0.01 * context['frequency']) \
                        - (0.005 * context['recency']) \
                        + (0.05 * context['weather_active']) \
                        - (0.05 * context['surge_multiplier'])

            # 2. Discounts are uniformly harmful after the shock.
            treatment_multiplier = -0.8

        # Keep pre-shock behavior unchanged (discounts weakly positive at best).
        if self.current_step < self.shock_step:
            treatment_multiplier = max(0, treatment_multiplier)
            treatment_effect = discount_value * treatment_multiplier
        else:
            treatment_effect = discount_value * treatment_multiplier
        
        # Final Probability
        final_prob = np.clip(base_prob + treatment_effect, 0.0, 1.0)
        return final_prob

    
if __name__ == "__main__":
    print("1. Instantiating the Marketplace Environment...")
    # We use a fixed seed for reproducible simulation results
    env = UberMarketplaceEnvironment(seed=42)
    
    print("2. Generating 100,000 rows of biased historical data...")
    df = env.generate_biased_historical_data(num_samples=100000)
    
    print("3. Saving to 'historical_marketplace_logs.csv'...")
    df.to_csv("historical_marketplace_logs.csv", index=False)
    print("   Done! Dataset ready for Phase 2.\n")
    
    # =====================================================================
    # THE PROOF OF CONFOUNDING BIAS
    # =====================================================================
    print("--- NAIVE BASELINE ANALYSIS (The Confounding Illusion) ---")
    
    # 1. NAIVE ATE (Average Treatment Effect)
    # How much does a 20% discount lift conversion compared to 0% discount 
    # if we just look at the raw historical averages?
    mean_conv_0 = df[df['treatment'] == 0]['converted'].mean()
    mean_conv_20 = df[df['treatment'] == 2]['converted'].mean()
    
    naive_lift = mean_conv_20 - mean_conv_0
    
    print(f"Naive Conversion @ 0% Discount:  {mean_conv_0:.1%}")
    print(f"Naive Conversion @ 20% Discount: {mean_conv_20:.1%}")
    print(f"Naive Causal Lift (SQL Average): +{naive_lift:.1%}")
    print("   -> CONCLUSION: 'Wow, the 20% discount is incredibly effective!'\n")
    
    # 2. TRUE ATE (Ground Truth)
    # Let's calculate what the EXACT causal lift of a 20% discount is for the 
    # entire population by looking at our hidden physics engine.
    true_effects = []
    for _, row in df.iterrows():
        # Reconstruct the context dictionary
        context = {
            'recency': row['recency'],
            'frequency': row['frequency'],
            'weather_active': row['weather_active'],
            'surge_multiplier': row['surge_multiplier']
        }
        # True prob if we forced 0% discount
        prob_0 = env._calculate_true_conversion(context, 0)
        # True prob if we forced 20% discount
        prob_20 = env._calculate_true_conversion(context, 2)
        
        # The true individual causal effect (CATE)
        true_effects.append(prob_20 - prob_0)
        
    actual_lift = np.mean(true_effects)
    
    print("--- TRUE CAUSAL ANALYSIS (The Hidden Physics) ---")
    print(f"True Average Causal Lift:        +{actual_lift:.1%}")
    print("   -> CONCLUSION: 'Wait, the discount actually does very little overall!'\n")
    
    # 3. THE DIAGNOSIS
    error_margin = naive_lift - actual_lift
    print("--- THE VERDICT ---")
    print(f"The naive analysis overestimates the effectiveness of the promotion by {error_margin:.1%}!")
    print("Why? Because the legacy system heavily targeted 'high frequency' (loyal) riders.")
    print("They were going to book anyway, but the naive math gave the credit to the discount.")