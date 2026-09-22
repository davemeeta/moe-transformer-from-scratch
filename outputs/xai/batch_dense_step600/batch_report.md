# RouteLens batch evaluation

**Checkpoint:** `checkpoints/dense_compare_v2/step_000600` &middot; **Prompts:** 50 sampled from `data/raw/tinyshakespeare.txt` &middot; **Seed:** 1337 &middot; values are mean [95% bootstrap CI over prompts]

## Red-team and judge

| explainer | flip rate | mean edit fraction | plausibility (1-5) | divergence rate |
|---|---|---|---|---|
| attention | 0.82 [0.70, 0.92] | 0.34 [0.25, 0.44] | 3.88 [3.72, 4.00] | 0.64 [0.50, 0.78] |
| shap | 1.00 [1.00, 1.00] | 0.15 [0.13, 0.18] | 3.88 [3.72, 4.00] | 0.88 [0.78, 0.96] |
| lime | 0.98 [0.94, 1.00] | 0.19 [0.15, 0.24] | 4.00 [4.00, 4.00] | 0.92 [0.84, 0.98] |

## Faithfulness

| explainer | del_auc | ins_auc | comprehensiveness | sufficiency | router τ |
|---|---|---|---|---|---|
| attention | 0.196 [0.171, 0.219] | 0.399 [0.330, 0.466] | 0.407 [0.293, 0.519] | 0.432 [0.345, 0.521] | -- |
| shap | 0.063 [0.049, 0.079] | 0.084 [0.061, 0.108] | 0.051 [0.012, 0.091] | 0.090 [0.052, 0.130] | -- |
| lime | 0.136 [0.098, 0.181] | 0.104 [0.070, 0.145] | 0.649 [0.550, 0.743] | 0.681 [0.594, 0.767] | -- |
