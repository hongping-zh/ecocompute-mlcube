# Security Policy

## Supported release

Only the latest tagged release receives fixes. At the time of writing this is
`schema-1.3-r5`.

## Reporting a vulnerability

Please report privately via
[GitHub security advisories](https://github.com/hongping-zh/ecocompute-mlcube/security/advisories)
("Report a vulnerability" -> draft a new security advisory).

Please do **not** open a public issue for anything security-sensitive.

## Scope

This container measures GPU energy during LLM inference; it does not handle
credentials or personal data. Supply-chain integrity is central to the project
(checksummed release assets, pinned installs, SBOM, non-root container), so
reports about tampering, dependency substitution or checksum mismatches are
especially welcome.
