import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

def true_physics_cate(row):
    """
    We recreate the hidden physics formula from Phase 1 here strictly 
    for grading the R-Learner. The algorithm NEVER sees this function.
    """
    multiplier = 0.5 - (0.10 * row['frequency']) + (0.04 * row['recency'])
    return max(0, multiplier)

if __name__ == "__main__":
    print("1. Loading the biased Data Lake...")
    df = pd.read_csv("historical_marketplace_logs.csv")
    
    # Define our Context (X), Treatment (T), and Outcome (Y)
    features = ['recency', 'frequency', 'weather_active', 'surge_multiplier']
    X = df[features]
    T = df['discount_value']
    Y = df['converted']
    
    print("\n2. Executing Double Machine Learning (R-Learner)...")
    
    # =====================================================================
    # STAGE 1: RESIDUALIZATION (Isolating the Noise)
    # We use cross_val_predict to prevent the models from overfitting to themselves.
    # =====================================================================
    print("   -> Modeling baseline conversion (Y ~ X)...")
    # We use a Regressor even though Y is binary, as we want the probability
    model_y = XGBRegressor(max_depth=5, random_state=42)
    Y_pred = cross_val_predict(model_y, X, Y, cv=5)
    Y_res = Y - Y_pred  # What conversion behavior couldn't be explained by context?
    
    print("   -> Modeling treatment assignment bias (T ~ X)...")
    # This models the legacy system's bad policy
    model_t = XGBRegressor(max_depth=5, random_state=42)
    T_pred = cross_val_predict(model_t, X, T, cv=5)
    T_res = T - T_pred  # What was the true 'randomness' in the discount given?
    
    # =====================================================================
    # STAGE 2: SIGNAL EXTRACTION (The Nie-Wager Formulation)
    # We want to find CATE(X) such that Y_res = CATE(X) * T_res.
    # We transform this into a weighted regression problem.
    # =====================================================================
    print("   -> Training final CATE estimator on residuals...")
    # Add tiny epsilon to prevent division by zero
    epsilon = 1e-6
    cate_target = Y_res / (T_res + epsilon)
    weights = T_res**2  # Weight by T_res^2 to penalize instances where T_res was tiny
    
    # Train the final model to predict the CATE based on user context
    cate_model = XGBRegressor(max_depth=5, random_state=42)
    cate_model.fit(X, cate_target, sample_weight=weights)
    
    # Predict the individual causal effect for every user in the dataset
    df['predicted_cate'] = cate_model.predict(X)
    
    # =====================================================================
    # EVALUATION: DID IT BEAT THE ILLUSION?
    # =====================================================================
    print("\n3. Grading the R-Learner against the Hidden Physics...")
    df['true_cate'] = df.apply(true_physics_cate, axis=1)
    
    # The ATEs
    true_ate = df['true_cate'].mean()
    predicted_ate = df['predicted_cate'].mean()
    
    print(f"   True Global Average Elasticity:      {true_ate:.3f}")
    print(f"   R-Learner Global Average Elasticity: {predicted_ate:.3f}")
    
    # The MSE
    mse = mean_squared_error(df['true_cate'], df['predicted_cate'])
    print(f"   Mean Squared Error (MSE):            {mse:.4f}")
    
    print("\n--- INDIVIDUAL PROFILE CHECKS ---")
    
    # 1. The Loyal User (High Freq, Low Recency)
    loyal_mask = (df['frequency'] >= 18) & (df['recency'] <= 5)
    loyal_true = df[loyal_mask]['true_cate'].mean()
    loyal_pred = df[loyal_mask]['predicted_cate'].mean()
    
    # 2. The Churning User (Low Freq, High Recency)
    churn_mask = (df['frequency'] <= 3) & (df['recency'] >= 25)
    churn_true = df[churn_mask]['true_cate'].mean()
    churn_pred = df[churn_mask]['predicted_cate'].mean()
    
    print("Profile 1: Highly Loyal User (The Trap)")
    print(f"  True Causal Lift:    {loyal_true:.1%} lift per unit of discount")
    print(f"  R-Learner Predicted: {loyal_pred:.1%} lift per unit of discount")
    print("  -> Did it learn not to give them discounts? " + ("YES!" if loyal_pred < 0.1 else "NO."))

    print("\nProfile 2: Churning User (The Target)")
    print(f"  True Causal Lift:    {churn_true:.1%} lift per unit of discount")
    print(f"  R-Learner Predicted: {churn_pred:.1%} lift per unit of discount")
    print("  -> Did it identify the massive elasticity? " + ("YES!" if churn_pred > 0.5 else "NO."))