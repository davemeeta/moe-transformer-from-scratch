# RouteLens batch evaluation

**Checkpoint:** `checkpoints/moe_compare_v2/step_000300` &middot; **Prompts:** 50 sampled from `data/raw/tinyshakespeare.txt` &middot; **Seed:** 1337 &middot; values are mean [95% bootstrap CI over prompts]

## Red-team and judge

| explainer | flip rate | mean edit fraction | plausibility (1-5) | divergence rate |
|---|---|---|---|---|
| attention | 0.70 [0.56, 0.82] | 0.46 [0.35, 0.56] | 3.96 [3.88, 4.00] | 0.48 [0.34, 0.62] |
| shap | 1.00 [1.00, 1.00] | 0.17 [0.14, 0.21] | 3.96 [3.88, 4.00] | 0.86 [0.76, 0.94] |
| lime | 0.98 [0.94, 1.00] | 0.20 [0.16, 0.25] | 4.02 [4.00, 4.06] | 0.88 [0.78, 0.96] |

## Faithfulness

| explainer | del_auc | ins_auc | comprehensiveness | sufficiency | router τ |
|---|---|---|---|---|---|
| attention | 0.173 [0.152, 0.194] | 0.477 [0.408, 0.541] | 0.471 [0.361, 0.575] | 0.326 [0.231, 0.421] | -0.119 [-0.195, -0.048] |
| shap | 0.181 [0.152, 0.211] | 0.089 [0.070, 0.109] | -0.138 [-0.192, -0.086] | 0.031 [-0.005, 0.065] | -- |
| lime | 0.153 [0.122, 0.189] | 0.082 [0.069, 0.098] | 0.553 [0.441, 0.668] | 0.665 [0.575, 0.745] | -- |

Router expert-collapse score per layer (0 = balanced, 1 = collapsed): [0.028, 0.002, 0.014, 0.026]
