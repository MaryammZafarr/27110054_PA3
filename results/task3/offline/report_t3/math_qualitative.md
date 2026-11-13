# MATH (Qwen2.5-3B): qualitative cases
Selection rule (fixed in advance, no gold used to choose): held-out questions in seed-6304 permutation order; the first 3 where, at N=32 and reveal order 0, self-consistency and Self-certainty + RWC pick different answers; and the first 3 incorrect early stops (quality-aware policy) where fixed-32 self-consistency was correct.

## Selection disagreements (SC vs Self-certainty + RWC)
### `test:2460`
**Q:** At a competition with $N$ players, the number of players given elite status is equal to $2^{1+\lfloor \log_{2} (N-1) \rfloor}-N$. Suppose that $19$ players are given elite status. What is the sum of the two smallest possible values of $N$?

- gold: `154`
- SC: `58` (✗)
- Self-certainty + RWC: `154` (✓)
- answer-group votes among 32: {'154': 11, '58': 11, 'NO_ANSWER': 3, '45': 1, '98': 1}

### `test:2892`
**Q:** There is a unique polynomial $P(x)$ of degree $8$ with rational coefficients and leading coefficient $1,$ which has the number \[\sqrt{2} + \sqrt{3} + \sqrt{5}\]as a root. Compute $P(1).$

- gold: `-71`
- SC: `1` (✗)
- Self-certainty + RWC: `0` (✗)
- answer-group votes among 32: {'0': 4, '1': 4, '23': 2, '-1': 1, 'NO_ANSWER': 1}

### `test:2948`
**Q:** Find all solutions to the equation $$(z^2 - 3z + 1)^2 - 3(z^2 - 3z + 1) + 1 = z.$$

- gold: `1+\sqrt 2,\ 1-\sqrt 2,\ 2+\sqrt 3,\ 2-\sqrt 3`
- SC: `1` (✗)
- Self-certainty + RWC: `2 + \sqrt{3}, 2 - \sqrt{3}, \frac{3 + \sqrt{13 + 4\sqrt{3}}}{2}, \frac{3 - \sqrt{13 + 4\sqrt{3}}}{2}, \frac{3 + \sqrt{13 - 4\sqrt{3}}}{2}, \frac{3 - \sqrt{13 - 4\sqrt{3}}}{2}` (✗)
- answer-group votes among 32: {'1': 5, '1 \\text{ and } 2': 4, '2 + \\sqrt{3}, 2 - \\sqrt{3}': 3, '2 + \\sqrt{3}, 2 - \\sqrt{3}, \\frac{3 + \\sqrt{13 + 4\\sqrt{3}}}{2}, \\frac{3 - \\sqrt{13 + 4\\sqrt{3}}}{2}, \\frac{3 + \\sqrt{13 - 4\\sqrt{3}}}{2}, \\frac{3 - \\sqrt{13 - 4\\sqrt{3}}}{2}': 3, 'NO_ANSWER': 2}

## Premature (incorrect) early stops recovered by fixed-32
### `test:3504` (order 7, stopped at n=4)
**Q:** What is the least positive multiple of 72 that has exactly 16 positive factors?

- gold: `216`
- revealed answers: {'120': 2, '72': 1, '216': 1}
- correct fraction in full pool of 32: 0.41

### `test:2460` (order 7, stopped at n=4)
**Q:** At a competition with $N$ players, the number of players given elite status is equal to $2^{1+\lfloor \log_{2} (N-1) \rfloor}-N$. Suppose that $19$ players are given elite status. What is the sum of the two smallest possible values of $N$?

- gold: `154`
- revealed answers: {'58': 3, '154': 1}
- correct fraction in full pool of 32: 0.34

### `test:534` (order 2, stopped at n=4)
**Q:** Recently, Frank took a one-hundred question aptitude test where each correct answer scored $5$ points, each incorrect answer scored $-2$ points, and each question not answered scored zero points. Frank answered $80$ questions and scored $232$ points. How many questions did he answer correctly?

- gold: `56`
- revealed answers: {'48': 2, '44': 1, '56': 1}
- correct fraction in full pool of 32: 0.72
