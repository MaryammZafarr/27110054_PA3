| policy | accuracy_pct | mean_token_cost | mean_rollouts | minus_fixed32_pp |
|---|---|---|---|---|
| fixed N=1 self-consistency | 64.350 | 669.773 | 1 |  |
| fixed N=2 self-consistency | 64.350 | 1339.865 | 2 |  |
| fixed N=4 self-consistency | 69.200 | 2679.694 | 4 |  |
| fixed N=8 self-consistency | 71.787 | 5344.108 | 8 |  |
| fixed N=16 self-consistency | 73.112 | 10677.743 | 16 |  |
| fixed N=32 self-consistency | 73.787 | 21347.018 | 32 |  |
| adaptive agreement (tau=0.625, gamma=None) | 73.638 | 10139.981 | 11.824 | -0.1 [-0.5, +0.2] |
| adaptive quality (tau=0.5, gamma=0.0) | 73.175 | 7663.055 | 9.008 | -0.6 [-1.2, -0.1] |
