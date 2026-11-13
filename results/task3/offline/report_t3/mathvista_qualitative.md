# MathVista (Qwen2.5-VL-3B): qualitative cases
Selection rule (fixed in advance, no gold used to choose): held-out questions in seed-6304 permutation order; the first 3 where, at N=32 and reveal order 0, self-consistency and VisConf + RWC pick different answers; and the first 3 incorrect early stops (quality-aware policy) where fixed-32 self-consistency was correct.

## Selection disagreements (SC vs VisConf + RWC)
### `552`
**Q:** How many queries have a p-value lower than 0.50?

- gold: `4`
- SC: `4` (✓)
- VisConf + RWC: `2` (✗)
- answer-group votes among 32: {'4': 14, '3': 9, '2': 9}

### `643`
**Q:** As shown in the figure, CD is the diameter of ⊙O, chord DE ∥ OA, if the degree of ∠D is 50.0, then the degree of ∠C is ()
A. 25°
B. 30°
C. 40°
D. 50°

- gold: `A`
- SC: `C` (✗)
- VisConf + RWC: `A` (✓)
- answer-group votes among 32: {'C': 11, 'A': 10, 'NO_ANSWER': 8, 'D': 3}

### `313`
**Q:** What is the age gap between these two people in image?

- gold: `10`
- SC: `UNPARSEABLE:10-20 years` (✗)
- VisConf + RWC: `20` (✗)
- answer-group votes among 32: {'UNPARSEABLE:10-15 years': 5, '10': 5, 'UNPARSEABLE:10-20 years': 5, '20': 4, 'UNPARSEABLE:20-30 years': 3}

## Premature (incorrect) early stops recovered by fixed-32
### `358` (order 4, stopped at n=4)
**Q:** Which of the cubes is the same as the unfolded cube?
A. A
B. B
C. C
D. D
E. E

- gold: `A`
- revealed answers: {'D': 2, 'A': 1, 'E': 1}
- correct fraction in full pool of 32: 0.44

### `821` (order 4, stopped at n=4)
**Q:** Are most of the people young men?
A. Yes
B. No

- gold: `A`
- revealed answers: {'B': 3, 'A': 1}
- correct fraction in full pool of 32: 0.66

### `434` (order 0, stopped at n=4)
**Q:** In the figure above, side AC of triangle ABC is on line l. What is x in terms of k?
A. 60-k
B. k
C. 60+k
D. 120-k
E. 120-2*k

- gold: `A`
- revealed answers: {'C': 2, 'NO_ANSWER': 1, 'A': 1}
- correct fraction in full pool of 32: 0.53
