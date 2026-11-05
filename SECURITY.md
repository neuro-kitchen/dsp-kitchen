# Security policy

## Reporting a vulnerability

Please report it privately, not in a public issue: open a
[security advisory](https://github.com/neuro-kitchen/dsp-kitchen/security/advisories/new) on this
repository. Include what is affected (crate, Python package, command line, `dspk-cli-sorter`), the
version, and how to reproduce it.

You will get an answer within a week. Fixes are released as a new version and described in the
advisory and the [CHANGELOG](CHANGELOG.md).

## Supported versions

dspk is in early development (0.0.x): only the latest release receives fixes.

## Scope

Of particular interest: the network sessions of `dspk-stream` (QUIC + TLS), downloads through
`dspk-hub` (models, sorter arrays; and the CLI's installer of the `dspk-cli-sorter` program), each
verified by SHA-256, and file parsers in `dspk-io` / `dspk-synapse` (`io`) reading untrusted
recordings.
