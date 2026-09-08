# RouteLens explanation report

**Prompt:** `'First Citizen:\nBefore we proceed any further, hear me speak.'`
**Predicting:** `P(next_token='\n')`

## Router trace (last layer, top-1 expert per token)

| token | expert |
|---|---|
| `'First'` | 3 |
| `' Citizen'` | 1 |
| `':'` | 2 |
| `'\n'` | 0 |
| `'Before'` | 2 |
| `' we'` | 2 |
| `' proceed'` | 1 |
| `' any'` | 2 |
| `' further'` | 1 |
| `','` | 2 |
| `' hear'` | 1 |
| `' me'` | 2 |
| `' speak'` | 1 |
| `'.'` | 2 |

## Top attributions per explainer

- **attention:** `'First'` (+0.213), `':'` (+0.134), `'.'` (+0.131), `' Citizen'` (+0.102), `'\n'` (+0.089)
- **shap:** `'speak'` (+0.182), `'me '` (-0.089), `'Before '` (+0.068), `'any '` (+0.031), `'hear '` (-0.025)
- **lime:** `'speak'` (+0.849), `'Before'` (+0.106), `'we'` (+0.017), `'further'` (-0.017), `'Citizen'` (+0.009)

## Faithfulness + red-team + judge

| explainer | del_auc | ins_auc | comp | suff | router_τ | edits | flipped | plausibility | faithfulness | divergent |
|---|---|---|---|---|---|---|---|---|---|---|
| attention | 0.246 | 0.688 | 0.83 | 0.192 | 0.209 | 11 | False | 5 | faithful | False |
| shap | 0.177 | 0.0642 | -0.485 | 0.0214 | -- | 1 | True | 4 | unfaithful | True |
| lime | 0.0837 | 0.0612 | 0.931 | 0.925 | -- | 1 | True | 4 | unfaithful | True |

## Findings

- **shap** sounds plausible (score 4/5) but is **unfaithful** -- the explanation doesn't survive the red-team attack. Judge's reasoning: The explanation makes sense because the signed importance values indicate that 'speak' is the most probable continuation, while the negative values for 'me ' and 'hear ' suggest that the model is correcting for the fact that they are incomplete or incorrectly formatted tokens, making the prediction of '
' plausible as a way to correct the error.
- **lime** sounds plausible (score 4/5) but is **unfaithful** -- the explanation doesn't survive the red-team attack. Judge's reasoning: The high importance of 'speak' suggests that the model is emphasizing the speaker's words, and the surrounding context supports this interpretation, making the explanation plausible to a human reader.

_Deeper red-team retry ran for: attention._
