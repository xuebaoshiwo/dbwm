# TradingBotHard raw tool chains

Generated 45 / 45 symbolic plans. Batch seed: 20260929.

Each case samples distinct targets uniformly from the catalog; targets stay fixed during retries.
`holdings` monitors the complete map. Each group has its stated minimum length and write goal.
Maximum length: 60. Dependency write maximum: 1.
Linked distractors enabled: True. Chains fill available padding slots when feasible.
Shared effects can exceed minimum writes. Order lifecycles may fall short of the goal; actual counts are below.
These plans have no concrete arguments, initial states or backend execution.

Regenerate from the repository root into an empty output directory:

```powershell
python -m bfcl_eval.consistency.data_generator_v2.generate_trajectory_grid --count-per-group 5 --seed 20260929 --max-length 60 --attempts 200 --linked-distractors --output-dir <empty-output-directory>
```

| Case | Targets | Actual length | Actual writes | Write shortfalls | Distractor chain lengths |
| --- | --- | ---: | --- | --- | --- |
| [l7_w2_001](min_length_7/min_writes_2/l7_w2_001.json) | watch_list | 7 | watch_list=2 | none | [2] |
| [l7_w2_002](min_length_7/min_writes_2/l7_w2_002.json) | watch_list | 7 | watch_list=2 | none | [2] |
| [l7_w2_003](min_length_7/min_writes_2/l7_w2_003.json) | balance | 7 | balance=2 | none | [] |
| [l7_w2_004](min_length_7/min_writes_2/l7_w2_004.json) | holdings | 7 | holdings=2 | none | [] |
| [l7_w2_005](min_length_7/min_writes_2/l7_w2_005.json) | watch_list | 7 | watch_list=2 | none | [2] |
| [l7_w3_001](min_length_7/min_writes_3/l7_w3_001.json) | transaction_history | 7 | transaction_history=3 | none | [2] |
| [l7_w3_002](min_length_7/min_writes_3/l7_w3_002.json) | transaction_history | 7 | transaction_history=3 | none | [2] |
| [l7_w3_003](min_length_7/min_writes_3/l7_w3_003.json) | holdings | 8 | holdings=3 | none | [] |
| [l7_w3_004](min_length_7/min_writes_3/l7_w3_004.json) | orders | 7 | orders=2 | orders=1 | [] |
| [l7_w3_005](min_length_7/min_writes_3/l7_w3_005.json) | watch_list | 7 | watch_list=3 | none | [2] |
| [l7_w4_001](min_length_7/min_writes_4/l7_w4_001.json) | balance | 7 | balance=4 | none | [] |
| [l7_w4_002](min_length_7/min_writes_4/l7_w4_002.json) | balance | 7 | balance=4 | none | [] |
| [l7_w4_003](min_length_7/min_writes_4/l7_w4_003.json) | orders | 7 | orders=2 | orders=2 | [] |
| [l7_w4_004](min_length_7/min_writes_4/l7_w4_004.json) | holdings | 10 | holdings=4 | none | [] |
| [l7_w4_005](min_length_7/min_writes_4/l7_w4_005.json) | transaction_history | 7 | transaction_history=4 | none | [] |
| [l14_w2_001](min_length_14/min_writes_2/l14_w2_001.json) | transaction_history, orders | 14 | transaction_history=2, orders=2 | none | [] |
| [l14_w2_002](min_length_14/min_writes_2/l14_w2_002.json) | balance, watch_list | 14 | balance=2, watch_list=2 | none | [2] |
| [l14_w2_003](min_length_14/min_writes_2/l14_w2_003.json) | transaction_history, holdings | 14 | transaction_history=2, holdings=2 | none | [] |
| [l14_w2_004](min_length_14/min_writes_2/l14_w2_004.json) | transaction_history, orders | 14 | transaction_history=2, orders=2 | none | [7] |
| [l14_w2_005](min_length_14/min_writes_2/l14_w2_005.json) | orders, holdings | 14 | orders=2, holdings=2 | none | [] |
| [l14_w3_001](min_length_14/min_writes_3/l14_w3_001.json) | orders, balance | 14 | orders=2, balance=3 | orders=1 | [] |
| [l14_w3_002](min_length_14/min_writes_3/l14_w3_002.json) | transaction_history, holdings | 14 | transaction_history=3, holdings=3 | none | [] |
| [l14_w3_003](min_length_14/min_writes_3/l14_w3_003.json) | balance, holdings | 14 | balance=4, holdings=3 | none | [] |
| [l14_w3_004](min_length_14/min_writes_3/l14_w3_004.json) | watch_list, holdings | 14 | watch_list=3, holdings=3 | none | [] |
| [l14_w3_005](min_length_14/min_writes_3/l14_w3_005.json) | holdings, transaction_history | 14 | holdings=3, transaction_history=3 | none | [] |
| [l14_w4_001](min_length_14/min_writes_4/l14_w4_001.json) | balance, watch_list | 14 | balance=4, watch_list=4 | none | [] |
| [l14_w4_002](min_length_14/min_writes_4/l14_w4_002.json) | balance, orders | 14 | balance=4, orders=2 | orders=2 | [] |
| [l14_w4_003](min_length_14/min_writes_4/l14_w4_003.json) | orders, watch_list | 14 | orders=2, watch_list=4 | orders=2 | [] |
| [l14_w4_004](min_length_14/min_writes_4/l14_w4_004.json) | watch_list, transaction_history | 14 | watch_list=4, transaction_history=4 | none | [2] |
| [l14_w4_005](min_length_14/min_writes_4/l14_w4_005.json) | transaction_history, orders | 14 | transaction_history=4, orders=2 | orders=2 | [5] |
| [l21_w2_001](min_length_21/min_writes_2/l21_w2_001.json) | holdings, transaction_history, orders | 21 | holdings=2, transaction_history=2, orders=2 | none | [] |
| [l21_w2_002](min_length_21/min_writes_2/l21_w2_002.json) | watch_list, holdings, balance | 21 | watch_list=2, holdings=2, balance=2 | none | [] |
| [l21_w2_003](min_length_21/min_writes_2/l21_w2_003.json) | transaction_history, orders, watch_list | 21 | transaction_history=2, orders=2, watch_list=2 | none | [] |
| [l21_w2_004](min_length_21/min_writes_2/l21_w2_004.json) | balance, transaction_history, orders | 21 | balance=2, transaction_history=2, orders=2 | none | [] |
| [l21_w2_005](min_length_21/min_writes_2/l21_w2_005.json) | transaction_history, balance, watch_list | 21 | transaction_history=2, balance=2, watch_list=2 | none | [2] |
| [l21_w3_001](min_length_21/min_writes_3/l21_w3_001.json) | watch_list, balance, orders | 21 | watch_list=3, balance=3, orders=2 | orders=1 | [] |
| [l21_w3_002](min_length_21/min_writes_3/l21_w3_002.json) | transaction_history, balance, watch_list | 21 | transaction_history=3, balance=4, watch_list=3 | none | [] |
| [l21_w3_003](min_length_21/min_writes_3/l21_w3_003.json) | holdings, transaction_history, orders | 21 | holdings=3, transaction_history=3, orders=2 | orders=1 | [] |
| [l21_w3_004](min_length_21/min_writes_3/l21_w3_004.json) | orders, holdings, transaction_history | 21 | orders=2, holdings=3, transaction_history=3 | orders=1 | [] |
| [l21_w3_005](min_length_21/min_writes_3/l21_w3_005.json) | balance, watch_list, orders | 21 | balance=3, watch_list=3, orders=2 | orders=1 | [] |
| [l21_w4_001](min_length_21/min_writes_4/l21_w4_001.json) | balance, orders, holdings | 21 | balance=6, orders=2, holdings=5 | orders=2 | [] |
| [l21_w4_002](min_length_21/min_writes_4/l21_w4_002.json) | balance, transaction_history, orders | 21 | balance=4, transaction_history=4, orders=2 | orders=2 | [] |
| [l21_w4_003](min_length_21/min_writes_4/l21_w4_003.json) | transaction_history, watch_list, holdings | 22 | transaction_history=4, watch_list=4, holdings=4 | none | [] |
| [l21_w4_004](min_length_21/min_writes_4/l21_w4_004.json) | watch_list, orders, balance | 21 | watch_list=4, orders=2, balance=4 | orders=2 | [] |
| [l21_w4_005](min_length_21/min_writes_4/l21_w4_005.json) | watch_list, holdings, balance | 21 | watch_list=4, holdings=4, balance=4 | none | [] |
