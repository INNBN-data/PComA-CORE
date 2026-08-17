# Concordance Notes and Locked-Object Requirements

## PComA-R sparse score

The manuscript describes a 0–17-point PComA-R instrument with an exploratory high-risk threshold of at least 11 points. The supplied reference script assigned weights of 2, 2, 2, 2, 2, and 3, which sum to 13 rather than 17. Because the manuscript is authoritative, this repository does not silently relabel the 13-point implementation or invent replacement weights.

`score_distillation/pcoma_r_score.py` therefore validates inputs and raises a clear error until the locked integer-weight vector is supplied. The provisional vector is retained only in `score_distillation/pcoma_r_score_provisional.py` for traceability and must not be cited as the final manuscript score.

## Exact model reproduction

The public reference architecture cannot guarantee identical model estimates without:

- the locked de-identified analytical dataset;
- original patient-grouped fold assignments;
- fitted preprocessing objects;
- tuned hyperparameters and selected iteration counts;
- OOF and temporal predictions;
- calibration objects;
- bootstrap distributions and confidence-interval method;
- source tables underlying figures.

This limitation is documented rather than concealed. It does not imply that the underlying clinical cohort is synthetic.
