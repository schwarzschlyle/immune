# CLI

Every command prints help with `immune <command> --help`. Commands that read state take `--state-dir` (default:
`IMMUNE_HOME` or `~/.immune`), and commands that read configuration take `--config` or use `IMMUNE_CONFIG` or
`./immune.yaml`. Exit code `0` means success, `1` a failed check, and `2` a usage or configuration error.

## Try and inspect

| Command | Purpose |
| --- | --- |
| `immune test "<message>" [--reply] [--operator] [--data] [--signal key=value] [--live]` | Screen one message and a draft reply, and print the verdict |
| `immune replay [ids or files...] [--kind] [--live]` | Run scenarios (default: your `./scenarios`, then Immune's built-in library); skips scenarios whose provider SDK is not installed and names it; exits 1 when an expectation fails |
| `immune threats [--stage]` | List the built-in threat catalog |
| `immune explain <threat>` | Show the invariant, floor, actions, head and Jev questions behind a threat |
| `immune version` | Installed version and spec version |

## Vaccines

| Command | Purpose |
| --- | --- |
| `immune vaccines new <id> [--kind keywords\|regex\|questions\|tool\|python] [--stage] [--title] [--dir] [--force]` | Scaffold a commented vaccine file that passes its own example tests |
| `immune vaccines list [--site] [--custom] [--library] [--off] [--config]` | Every built-in threat and vaccine, whether it is on, and the setting that decided it; `--library` shows the vaccine library |
| `immune vaccines show <id> [--site] [--config]` | A vaccine's card: what it catches, its measured numbers, its state and how to switch it on |
| `immune vaccines fork <id> --as <your.id> [--dir] [--force]` | Copy a library vaccine into your own vaccines to tailor it |
| `immune vaccines test [paths...] [--live] [--site] [--config]` | Run your vaccines' positive and negative examples; exits 1 on a failure |
| `immune vaccines trial <id or file> [--corpus FILE] [--no-self] [--limit] [--max-rate] [--live]` | Estimate how often a vaccine fires on everyday traffic; exits 1 above `--max-rate` |
| `immune vaccinate <id> --stage S (--keyword … \| --regex … \| --question … \| --tool … [--argument …] \| --python …) --positive … [--negative …] [--message] [--site] [--dir] [--corpus] [--live]` | Build a vaccine from examples, test and trial it, and write it observed-first |

## Configuration and health

| Command | Purpose |
| --- | --- |
| `immune config validate [path]` | Check a configuration file |
| `immune config schema` | Print the JSON Schema for `immune.yaml` |
| `immune init [--output] [--force]` | Write an `immune.yaml` that pins what Immune inferred about each site |
| `immune doctor [--live]` | Check configuration, dependencies, interception and Jev connectivity |

## Operating a deployment

| Command | Purpose |
| --- | --- |
| `immune status` | Sites, profiles and promotions in the state directory |
| `immune posture` | Configuration risks for every observed site |
| `immune promote` | Which observed defenses have earned automatic enforcement, and why |
| `immune label [--limit]` | Review observed hits from the verdict log and record whether they were right |
| `immune calibrate [--min-samples]` | Fit per-threat calibration from labels |
| `immune export labels\|verdicts [--format jsonl\|csv] [--output]` | Export labels or recorded verdicts |

## Evidence: training the antibodies

| Command | Purpose |
| --- | --- |
| `immune evidence collect <examples> --out FILE [--sensor jev\|mock\|replay:…\|record:…]` | Run labeled examples through Immune and record head features |
| `immune evidence fit <features> --out heads.json [--card FILE] [--jev-model]` | Train heads and write an artifact and model card |
| `immune evidence drift <baseline> <candidate> [--tolerance]` | Fail when a candidate artifact regresses against a baseline |
| `immune evidence study <features>` | Measure how calibrated each Jev question is on its own |
| `immune evidence questions <examples> --variants FILE [--sensor]` | Rank alternative phrasings of Jev questions |
