"""
Next Best Action with LinUCB Contextual Bandit on synthetic fintech data.
"""

import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(42)

ACTIONS = [
    "savings_boost",
    "premium_account",
    "credit_card_upsell",
    "personal_loan",
    "investment_product",
    "retention_nudge",
    "cashback_offer",
    "do_nothing"
]

def generate_synthetic_customers(n=2000):
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": np.random.randint(22, 65, n),
        "tenure_months": np.random.exponential(18, n).astype(int) + 1,
        "balance": np.random.lognormal(8.5, 1.2, n).round(2),
        "monthly_tx": np.random.poisson(12, n),
        "products_held": np.random.randint(1, 5, n),
        "engagement_score": np.random.beta(2, 5, n),
        "churn_risk": np.random.beta(1.5, 4, n),
        "income_proxy": np.random.lognormal(10.2, 0.6, n).round(0),
    })
    return df

def context_vector(row):
    # Simple feature engineering for the bandit
    return np.array([
        row["age"] / 65,
        min(row["tenure_months"] / 60, 1),
        np.log1p(row["balance"]) / 12,
        min(row["monthly_tx"] / 30, 1),
        row["products_held"] / 5,
        row["engagement_score"],
        row["churn_risk"],
        np.log1p(row["income_proxy"]) / 12,
        1.0  # bias
    ])

class LinUCB:
    def __init__(self, n_actions, n_features, alpha=1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.eye(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]

    def select_action(self, x):
        p = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            conf = self.alpha * np.sqrt(x @ A_inv @ x)
            p.append(theta @ x + conf)
        return int(np.argmax(p))

    def update(self, action, x, reward):
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

def simulate_reward(customer, action_idx):
    """Synthetic reward function that depends on customer context and action."""
    a = ACTIONS[action_idx]
    base = 0.0
    if a == "savings_boost" and customer["balance"] > 5000 and customer["engagement_score"] > 0.4:
        base = 0.35
    elif a == "premium_account" and customer["income_proxy"] > 40000 and customer["products_held"] >= 2:
        base = 0.28
    elif a == "credit_card_upsell" and customer["monthly_tx"] > 8 and customer["age"] < 45:
        base = 0.25
    elif a == "personal_loan" and customer["churn_risk"] < 0.3 and customer["tenure_months"] > 12:
        base = 0.22
    elif a == "investment_product" and customer["balance"] > 15000:
        base = 0.30
    elif a == "retention_nudge" and customer["churn_risk"] > 0.5:
        base = 0.40
    elif a == "cashback_offer" and customer["monthly_tx"] > 15:
        base = 0.18
    elif a == "do_nothing":
        base = 0.05
    noise = np.random.normal(0, 0.08)
    return float(np.clip(base + noise, 0, 1))

def main():
    print("Generating synthetic fintech customers...")
    customers = generate_synthetic_customers(2000)
    print(f"Customers: {len(customers)}")

    n_features = 9
    bandit = LinUCB(len(ACTIONS), n_features, alpha=0.8)

    cumulative_reward = 0
    rewards = []
    action_counts = np.zeros(len(ACTIONS))

    # Online simulation: process customers sequentially
    for _, row in customers.iterrows():
        x = context_vector(row)
        a = bandit.select_action(x)
        r = simulate_reward(row, a)
        bandit.update(a, x, r)
        cumulative_reward += r
        rewards.append(cumulative_reward)
        action_counts[a] += 1

    print("\n=== Final cumulative reward ===")
    print(round(cumulative_reward, 2))
    print("\n=== Action selection counts ===")
    for i, name in enumerate(ACTIONS):
        print(f"{name:22s}: {int(action_counts[i]):4d}")

    # Baseline: always best fixed action (estimated)
    print("\n=== Comparison (last 500 customers average reward) ===")
    print("LinUCB average reward:", round(np.mean(rewards[-500:]) / 500 if len(rewards) > 500 else 0, 4))

    print("\n--- Recommended actions for a fintech NBA system ---")
    print("1. Serve the bandit policy as a daily/weekly batch scoring job into the CRM.")
    print("2. Use early engagement and balance signals as strong predictors of offer acceptance.")
    print("3. Keep a small exploration budget (controlled by alpha) to discover new high-value segments.")
    print("4. Validate incremental lift with holdout groups before full rollout.")

if __name__ == "__main__":
    main()
