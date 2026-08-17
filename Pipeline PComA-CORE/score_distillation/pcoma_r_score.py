"""Locked PComA-R score entrypoint.

The manuscript declares a 0-17 point score. The integer weights supplied in the
reference script sum to 13, so a final 0-17 implementation cannot be recovered
without the locked score object. This module intentionally refuses to fabricate
weights. See pcoma_r_score_provisional.py and docs/CONCORDANCE_NOTES.md.
"""


def compute_score(df):
    raise RuntimeError(
        "Final PComA-R integer weights are unresolved: manuscript range 0-17, "
        "supplied provisional weights sum to 13. Provide the locked 0-17 weight vector."
    )


def main():
    compute_score(None)


if __name__ == "__main__":
    main()
