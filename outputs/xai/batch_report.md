# RouteLens batch evaluation

**Prompts evaluated:** 5 (sampled from `data/raw/tinyshakespeare.txt`)

- `Before we proceed any further, hear me speak.`
- `You are all resolved rather to die than to famish?`
- `First, you know Caius Marcius is chief enemy to the people.`
- `We know't, we know't.`
- `Let us kill him, and we'll have corn at our own price.`

| explainer | mean del_auc | mean ins_auc | mean comp | mean suff | mean edit_frac | flip rate | mean plausibility | divergence rate |
|---|---|---|---|---|---|---|---|---|
| attention | 0.187 | 0.723 | 0.865 | 0.225 | 0.985 | 0.00 | 3.20 | 0.40 |
| shap | 0.225 | 0.0732 | -0.266 | -0.0049 | 0.117 | 1.00 | 2.80 | 0.20 |
| lime | 0.138 | 0.139 | 0.888 | 0.879 | 0.129 | 1.00 | 4.40 | 0.80 |

## Router context (not a statistical correlation -- too few prompts to fit one honestly)

Expert-collapse score per layer (0 = balanced, 1 = collapsed onto one expert): [0.042, 0.009, 0.008, 0.015]
