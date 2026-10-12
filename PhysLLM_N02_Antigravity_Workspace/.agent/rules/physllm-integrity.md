# Mandatory integrity rules

1. No source data overwrite; maintain SHA-256 before/after.
2. Every repair is a new candidate version with a row-wise change record.
3. An automated pass is not a content review.
4. Never convert a warning to a pass without a documented reason.
5. Never run confirmatory statistics until the manifest has an external approval record and is explicitly frozen.
6. If file versions/hashes disagree, stop and report the discrepancy rather than guessing which version is authoritative.
