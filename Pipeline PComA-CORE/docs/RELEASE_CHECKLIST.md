# Public Release Checklist

- [ ] Confirm that no direct identifiers or free-text clinical fields are present.
- [ ] Confirm that no patient-level rows are included in the public archive.
- [ ] Confirm that operator identifiers and reversible linkage keys are excluded.
- [ ] Confirm exact author order, affiliations, corresponding authors, and ethics wording.
- [ ] Confirm that the final manuscript, figures, tables, and `reported_metrics.yaml` agree.
- [ ] Resolve the PComA-R 0–17 integer-weight vector before calling the score implementation final.
- [ ] Replace no manuscript result with a newly generated value unless the complete locked analysis has been reproduced and approved.
- [ ] Add the final repository URL and DOI to `CITATION.cff` only after assignment.
- [ ] Run `python repository_audit.py` and verify a zero-error result.
- [ ] Verify `MANIFEST.sha256` and the ZIP-level SHA-256 checksum.
