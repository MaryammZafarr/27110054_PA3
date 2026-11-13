# Aligned complementary-pair examples
Selection rule: for each final-test suite and question type, the first 2 pairs in the saved (seed 6304) split order. Selected before inspecting predictions.

## grounding_test · counting · `counting-val2014-23-203635-357219`
**Q:** How many bears can you see?

- **image A** `val2014:203635` · reference: 2
  - correct: `2`
  - swap: `3`
  - mismatched: `0`
  - none: `1`
- **image B** `val2014:357219` · reference: 4
  - correct: `3`
  - swap: `2`
  - mismatched: `1`
  - none: `1`

## grounding_test · counting · `counting-val2014-6-534673-222548`
**Q:** How many buses can you see?

- **image A** `val2014:534673` · reference: 1
  - correct: `1`
  - swap: `4`
  - mismatched: `0`
  - none: `1`
- **image B** `val2014:222548` · reference: 2
  - correct: `4`
  - swap: `1`
  - mismatched: `0`
  - none: `1`

## grounding_test · presence · `presence-val2014-82-80369-87356`
**Q:** Does this image contain a refrigerator?

- **image A** `val2014:80369` · reference: yes
  - correct: `no`
  - swap: `no`
  - mismatched: `no`
  - none: `yes`
- **image B** `val2014:87356` · reference: no
  - correct: `no`
  - swap: `no`
  - mismatched: `no`
  - none: `yes`

## grounding_test · presence · `presence-val2014-37-135604-415434`
**Q:** Does this image contain a sports ball?

- **image A** `val2014:135604` · reference: yes
  - correct: `yes`
  - swap: `no`
  - mismatched: `no`
  - none: `yes`
- **image B** `val2014:415434` · reference: no
  - correct: `no`
  - swap: `yes`
  - mismatched: `no`
  - none: `yes`

## vqa_test · number · `vqa-val2014-301266002-474609018`
**Q:** How many people are wearing glasses?

- **image A** `val2014:301266` · reference: 2  (10 refs: 2, 2, 2, 2, 2, 1, 2, 2, 2, 2)
  - correct: `0`
  - swap: `1`
  - mismatched: `0`
  - none: `1`
- **image B** `val2014:474609` · reference: 1  (10 refs: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
  - correct: `1`
  - swap: `0`
  - mismatched: `0`
  - none: `1`

## vqa_test · number · `vqa-val2014-277932001-296404004`
**Q:** How many animals are in the image?

- **image A** `val2014:277932` · reference: 3  (10 refs: 5, 3, 3, 3, 3, 1, 3, 3, 3, 3)
  - correct: `2`
  - swap: `2`
  - mismatched: `0`
  - none: `1`
- **image B** `val2014:296404` · reference: 2  (10 refs: 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
  - correct: `2`
  - swap: `2`
  - mismatched: `4`
  - none: `1`

## vqa_test · other · `vqa-val2014-509185000-305526002`
**Q:** What color are the hands on the clock?

- **image A** `val2014:509185` · reference: gold  (10 refs: gold, gold, gold, gold, gold, gold, gold, gold, gold, gold)
  - correct: `red`
  - swap: `red`
  - mismatched: `red`
  - none: `red`
- **image B** `val2014:305526` · reference: black  (10 refs: black, black, black, black, black, black, black, black, black, black)
  - correct: `red`
  - swap: `red`
  - mismatched: `red`
  - none: `red`

## vqa_test · other · `vqa-val2014-240655001-209757036`
**Q:** What is on the person's head?

- **image A** `val2014:240655` · reference: helmet  (10 refs: helmet, helmet, helmet, helmet, helmet, hat, helmet, helmet, cap, helmet)
  - correct: `green`
  - swap: `skateboard`
  - mismatched: `head`
  - none: `head`
- **image B** `val2014:209757` · reference: hat  (10 refs: hat, cap, hat, hat, hat, hat, hat, hat, cap, hat)
  - correct: `skateboard`
  - swap: `green`
  - mismatched: `cellular phone`
  - none: `head`

## vqa_test · yes/no · `vqa-val2014-80949002-523292006`
**Q:** Is the computer turned on?

- **image A** `val2014:80949` · reference: yes  (10 refs: yes, yes, yes, yes, yes, yes, yes, yes, yes, yes)
  - correct: `yes`
  - swap: `yes`
  - mismatched: `no`
  - none: `yes`
- **image B** `val2014:523292` · reference: no  (10 refs: no, no, no, no, no, no, no, no, no, no)
  - correct: `yes`
  - swap: `yes`
  - mismatched: `no`
  - none: `yes`

## vqa_test · yes/no · `vqa-val2014-143644001-540830000`
**Q:** Is the bear in a pocket?

- **image A** `val2014:143644` · reference: yes  (10 refs: yes, yes, yes, yes, yes, yes, yes, yes, yes, yes)
  - correct: `yes`
  - swap: `no`
  - mismatched: `no`
  - none: `no`
- **image B** `val2014:540830` · reference: no  (10 refs: no, no, no, no, no, no, no, no, no, yes)
  - correct: `no`
  - swap: `yes`
  - mismatched: `no`
  - none: `no`
