| phase | trainable_params | time_min | peak_gpu_GB | note |
|---|---|---|---|---|
| feature caching (SigLIP, separate) | 0 | 3.269 | 1.329 | 9333 images, 2.62 GB FP16 |
| train caption_alignment | 1837056 | 47.741 | 1.542 | connector 1,837,056 + LoRA 0 |
| train visual_instruction | 6424576 | 38.017 | 1.564 | connector 1,837,056 + LoRA 4,587,520 |
| train natural_vqa | 6424576 | 18.611 | 1.560 | connector 1,837,056 + LoRA 4,587,520 |
| development evaluations (all) | 0 | 1.630 | nan | 12 evaluation points |
| final evaluation generation | 0 | 0.861 | 1.664 | grounding_test/correct: 37.8 ms/answer; grounding_test/swap: 35.6 ms/answer; grounding_test/mismatched: 36.9 ms/answer; grounding_test/none: 17.2 ms/answer; vqa_test/correct: 41.8 ms/answer; vqa_test/swap: 40.8 ms/answer; vqa_test/mismatched: 39.6 ms/answer; vqa_test/none: 19.5 ms/answer |
