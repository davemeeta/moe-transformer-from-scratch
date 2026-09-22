# RouteLens batch evaluation

**Checkpoint:** `checkpoints/moe_compare_v2/step_000600` &middot; **Prompts:** 50 sampled from `data/raw/tinyshakespeare.txt` &middot; **Seed:** 1337 &middot; values are mean [95% bootstrap CI over prompts]

## Red-team and judge

| explainer | flip rate | mean edit fraction | plausibility (1-5) | divergence rate |
|---|---|---|---|---|
| attention | 0.86 [0.76, 0.94] | 0.33 [0.25, 0.42] | 4.00 [4.00, 4.00] | 0.64 [0.50, 0.76] |
| shap | 1.00 [1.00, 1.00] | 0.16 [0.14, 0.18] | 3.96 [3.88, 4.00] | 0.88 [0.78, 0.96] |
| lime | 0.98 [0.94, 1.00] | 0.18 [0.15, 0.22] | 3.98 [3.88, 4.06] | 0.90 [0.82, 0.98] |

## Faithfulness

| explainer | del_auc | ins_auc | comprehensiveness | sufficiency | router τ |
|---|---|---|---|---|---|
| attention | 0.214 [0.190, 0.238] | 0.426 [0.359, 0.492] | 0.399 [0.290, 0.512] | 0.504 [0.410, 0.595] | -0.025 [-0.101, 0.048] |
| shap | 0.168 [0.132, 0.205] | 0.102 [0.085, 0.123] | -0.105 [-0.177, -0.033] | 0.041 [-0.010, 0.090] | -- |
| lime | 0.185 [0.139, 0.237] | 0.109 [0.086, 0.143] | 0.611 [0.519, 0.703] | 0.704 [0.606, 0.790] | -- |

Router expert-collapse score per layer (0 = balanced, 1 = collapsed): [0.041, 0.009, 0.006, 0.013]
