# verify.py: one failed check (oplora16 orthogonality)

Check: "oplora16: Uk^T B~ ~ 0 and A~ Vk ~ 0 for every module (FP32)", threshold: max_out_rel < 1e-4 and max_in_rel < 1e-4
(verify.py line 103-104; the selftest used 1e-5 on random factors of a tiny model).

Measured (196 adapted modules, oplora_constraint_check.json):
- max_out_rel = 1.20e-4 (median 3.4e-5; 2 of 196 modules above 1e-4)
- max_in_rel  = 1.86e-4 (median 1.7e-4; 140 of 196 modules above 1e-4)
- the same numbers for the full update dW: 1.2e-4 / 1.9e-4

Finding: the saved protected bases (runs/oplora16/bases_k128.pt) are only approximately orthonormal.
In float64, max|Uk^T Uk - I| has median 4.9e-4 and max 9.2e-4; for Vk median 5.2e-4, max 1.1e-3.
Projection removes components along the columns of Uk/Vk, so it leaves a residual of about |I - Uk^T Uk| times the
projected quantity, which matches the measured 1e-4 to 2e-4. This is not drift from mixed precision in training: the
projection itself runs in FP32 (autocast disabled in LoRALinear.effective_factors), and the residual is set by how
orthonormal the bases are. (Likely cause, NOT verified: torch.linalg.svd on the GPU, which can return less
orthonormal factors than a CPU/float64 SVD.)

Size: leaked relative norm ~2e-4 means about 4e-8 of the update's energy lies in the protected subspace. The constraint
holds to ~1e-4 relative; it is a miss of the self-chosen 1e-4 tolerance, not a violation of the method.
