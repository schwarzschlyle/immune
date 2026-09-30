# Governance

## Independence

Immune is an independent open-source project. It is not affiliated with, endorsed by or funded by TypeSafe AI or any
model provider. Use of the Jev model is subject to TypeSafe AI's terms.

## Maintainers

Maintainers review and merge changes, cut releases and triage security reports. New maintainers are nominated by an
existing maintainer and confirmed by consensus of the current maintainers.

## Decisions

Day-to-day decisions happen in pull requests. Changes to the public API, the spec format, default enforcement or data
flows need an architecture decision record in `docs/adr/` and approval from two maintainers.

## Releases

Releases are cut from tags by the release workflow using PyPI Trusted Publishing. Every release must pass the full test
suite, the scenario library and the supply-chain checks described in `SECURITY.md`.
