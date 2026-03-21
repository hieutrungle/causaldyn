import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error


class RLearner:
    """R-learner for continuous treatment using XGBoost nuisance and CATE models.

    The model fits:
      1) mu(x) = E[Y|X]
      2) e(x) = E[T|X]
      3) tau(x) from weighted regression on residualized targets

    Then predicts absolute outcomes as mu(x) + t * tau(x).
    """

    def __init__(self, random_state=42, cv=5, epsilon=1e-6):
        self.random_state = random_state
        self.cv = cv
        self.epsilon = epsilon

        self.model_y = XGBRegressor(max_depth=5, random_state=random_state)
        self.model_t = XGBRegressor(max_depth=5, random_state=random_state)
        self.model_tau = XGBRegressor(max_depth=5, random_state=random_state)

        self._is_fitted = False

    def fit(self, X, T, Y):
        """Fits nuisance models and CATE model.

        Args:
            X: Context dataframe or numpy array (n, d)
            T: Continuous treatment vector (n,)
            Y: Outcome vector (n,)
        """
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        T_arr = np.asarray(T, dtype=float)
        Y_arr = np.asarray(Y, dtype=float)

        # Cross-fitted nuisance predictions for residualization.
        y_hat = cross_val_predict(self.model_y, X_df, Y_arr, cv=self.cv)
        t_hat = cross_val_predict(self.model_t, X_df, T_arr, cv=self.cv)

        y_res = Y_arr - y_hat
        t_res = T_arr - t_hat

        cate_target = y_res / (t_res + self.epsilon)
        sample_weight = (t_res ** 2) + self.epsilon

        # Fit final models on full data for serving.
        self.model_y.fit(X_df, Y_arr)
        self.model_t.fit(X_df, T_arr)
        self.model_tau.fit(X_df, cate_target, sample_weight=sample_weight)

        self._is_fitted = True
        return self

    def predict_cate(self, X):
        """Predicts CATE tau(x)."""
        if not self._is_fitted:
            raise ValueError("RLearner must be fitted before predict_cate().")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self.model_tau.predict(X_df)

    def predict_mu(self, X):
        """Predicts baseline mu(x)=E[Y|X]."""
        if not self._is_fitted:
            raise ValueError("RLearner must be fitted before predict_mu().")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self.model_y.predict(X_df)

    def predict_outcome(self, X, treatment_value):
        """Predicts E[Y|X, T=treatment_value]."""
        mu = self.predict_mu(X)
        tau = self.predict_cate(X)
        return mu + (float(treatment_value) * tau)

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
    print("   -> Fitting RLearner with XGBoost nuisance and CATE models...")
    r_learner = RLearner(random_state=42, cv=5, epsilon=1e-6).fit(X, T, Y)
    df['predicted_cate'] = r_learner.predict_cate(X)
    
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