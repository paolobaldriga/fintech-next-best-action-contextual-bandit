# Fintech Next Best Action — Contextual Bandit

**Personalized Next Best Action using a LinUCB Contextual Bandit on synthetic multi-product fintech data.**

A practical demonstration of how a bank or neobank can select the most relevant customer action (product offer, engagement nudge, retention intervention) while balancing exploitation of known high-value actions and exploration of new ones.

> Portfolio piece by Paolo Baldriga — Marketing Measurement & AI Executive (MMM, Incrementality, CRM/Loyalty, Customer Analytics).

## Why Contextual Bandits for NBA

Classic rules or static propensity models pick the historically best action. Contextual Bandits (here LinUCB) treat each customer context as a feature vector and learn which action maximises expected reward, while systematically exploring alternatives. This is ideal for Next Best Action where:

- Rewards are delayed or sparse (conversion, incremental revenue, retention)
- Customer contexts change (balance, activity, product holdings)
- The cost of always showing the same offer is high (fatigue, missed opportunities)

## What the project delivers

| Component | Method | Output |
|-----------|--------|--------|
| Synthetic fintech data | Customer profiles + possible actions | Realistic contexts (balance, tx, products, engagement) |
| Contextual Bandit | LinUCB (linear upper confidence bound) | Recommended action + expected reward + uncertainty |
| Simulation | Online-style learning loop | Cumulative reward, regret proxy, action distribution |
| Evaluation | Comparison vs random / best-fixed | Lift in simulated reward |

## Quick start

```bash
pip install -r requirements.txt
python src/demo_contextual_bandit.py
```

Or open `notebooks/nba_contextual_bandit.ipynb` and run all cells.

## Key outputs

- Recommended action per customer with confidence
- Learning curve (cumulative reward)
- Action selection frequency over time (exploration → exploitation)
- Comparison against a static best-action baseline

## Architecture notes

- Pure Python + NumPy / scikit-learn style LinUCB (no heavy RL stack)
- Modular so the same logic can sit on real warehouse features (dbt + BigQuery / Snowflake)
- Designed for CRM / marketing automation: the policy can be served as a batch scoring job or near-real-time API
- Incrementality mindset: rewards should ultimately be validated with holdouts or geo tests

## Author

Paolo Baldriga  
Marketing Measurement & AI Executive  
Hamburg / Rome  
[GitHub](https://github.com/paolobaldriga)

---

*Independent portfolio project. Synthetic data only.*
