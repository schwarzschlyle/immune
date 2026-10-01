# Laboratory

Where Immune's library vaccines are developed and measured. Each vaccine in `src/immune/vaccines/library/` has a
folder here with its evidence, the Jev answers recorded for it, and its measured card.

| Path | Contents |
| --- | --- |
| `vaccines/<id>/` | Evidence, manifest, recorded Jev answers (`cassette.jsonl`), the card and notes for one vaccine |
| `self/` | Everyday traffic that no vaccine should fire on, per stage, with its manifest |

`tools/vaccine_lab.py` does the work (`new`, `record`, `fit`, `measure`, `index`, `check`). The full guide is
[docs/contributing/vaccine-laboratory.md](../docs/contributing/vaccine-laboratory.md).

Everything here is synthetic or permissively licensed, and each source is listed in a manifest. It is never
customer traffic. The laboratory is kept out of the wheel and the sdist.
