# Releasing Immune

Immune is published to PyPI as [`immune-ai`](https://pypi.org/project/immune-ai/). Releases are driven by git tags:
the version is read from the tag at build time (hatch-vcs), and GitHub Actions builds, checks, attests and publishes
the package. Nobody uploads from a laptop and no API tokens are stored anywhere.

## Branches and environments

| Branch | Purpose | Every push builds | Published to |
| --- | --- | --- | --- |
| `dev` | Integration: feature pull requests merge here | a development build, `0.2.0.devN` | TestPyPI (`testpypi` environment), then a smoke test installs it back |
| `staging` | Release preparation: `dev` is promoted here when it is ready to ship | a development build, and `vX.Y.ZrcN` tags | TestPyPI, then the same smoke test; release candidates also get a GitHub pre-release |
| `main` | Released code only: `staging` is promoted here, and every release is a tag on `main` | nothing until tagged `vX.Y.Z` | PyPI (`pypi` environment, approval required) and a GitHub release |

Changes always flow `dev` → `staging` → `main` through pull requests, and the `promotion` check in CI rejects
anything else. The exceptions are `release/*` branches, which carry the changelog roll-over into `staging`, and
`hotfix/*` branches, which may target `staging` or `main`. After each release, the workflow opens pull requests
that bring `staging` and `dev` back in sync with `main`.

Merge promotion pull requests (`dev` → `staging`, `staging` → `main` and the back-merges) with **Create a merge
commit**, never squash or rebase. The three branches then share one history: release tags on `main` are reachable
from `dev` and `staging`, and the back-merges stay trivial. Feature pull requests into `dev` may be squashed.

## Versioning

Immune follows [Semantic Versioning](https://semver.org/) with [PEP 440](https://peps.python.org/pep-0440/) spellings.

| Change | Bump | Example |
| --- | --- | --- |
| Breaking change to the public API, configuration or verdict format | major (minor while `0.x`) | `0.3.0`, later `2.0.0` |
| New feature, new threat, spec version change | minor | `0.2.0` |
| Bug fix, documentation, performance | patch | `0.1.1` |
| Release candidate before a final release | `rcN` suffix | `0.2.0rc1` |

- The public API is what `tools/api_check.py` snapshots in `api/public-api.json`, plus the configuration schema and
  the `Verdict` fields. A change to either needs the right bump.
- While the version is `0.x`, minor releases may break compatibility; every such change is listed under **Changed**
  or **Removed** in the changelog. `0.x` releases are marked as pre-releases on GitHub; `1.0.0` is the first stable
  release.
- Everything between releases is a development release of the next version: `0.2.0.devN` after `0.1.0`, or
  `0.2.0rc2.devN` after `0.2.0rc1`. These are valid on TestPyPI, sort above every earlier version, and are never
  mistaken for a release. In CI, N is the workflow run number, which is unique across `dev` and `staging`
  (`tools/release.py dev-version`); a local `python -m build` uses the commit distance from the last tag instead.
- Tags are `v` + the version: `v0.1.0`, `v0.2.0rc1`.

`tools/release.py` computes and checks versions so nobody has to do it by hand:

```bash
python tools/release.py next --bump minor          # 0.1.0 -> 0.2.0
python tools/release.py next --bump minor --rc     # -> 0.2.0rc1, or the next rc number
python tools/release.py prepare 0.2.0              # CHANGELOG: [Unreleased] -> [0.2.0] - today
python tools/release.py check v0.2.0               # the tag is valid and the changelog has its section
python tools/release.py notes 0.2.0                # the release notes the workflow publishes
python tools/release.py dev-version --number 57    # 0.2.0.dev57: what CI builds on dev and staging
```

## Everyday development

1. Branch from `dev`, open a pull request into `dev`, and add an entry under `## [Unreleased]` in `CHANGELOG.md`.
2. CI must pass. Merging publishes a development build to TestPyPI, so the change can be installed with
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ immune-ai==<version>`.

## Cutting a release

```bash
# 1. Promote dev to staging: open a pull request dev -> staging and merge it (merge commit).

# 2. Decide the version and roll the changelog over on a release branch, then merge it into staging.
git switch staging && git pull
VERSION=$(python tools/release.py next --bump minor)       # or --bump patch / --bump major
git switch -c "release/v$VERSION"
python tools/release.py prepare "$VERSION"
git commit -am "chore(release): prepare v$VERSION"
git push -u origin "release/v$VERSION"                    # open a pull request into staging and merge it

# 3. Optional but recommended: tag a release candidate on staging. It goes to TestPyPI and a GitHub pre-release.
git switch staging && git pull
RC=$(python tools/release.py next --bump minor --rc)       # same --bump as step 2
git tag -a "v$RC" -m "Immune $RC" && git push origin "v$RC"

# 4. Promote staging to main (pull request staging -> main, merge commit), then tag the release on main.
git switch main && git pull
python tools/release.py check "v$VERSION"
git tag -a "v$VERSION" -m "Immune $VERSION" && git push origin "v$VERSION"
```

The tag starts the `release` workflow: it checks the tag against the changelog and `main`, builds the sdist and
wheel, verifies that they are reproducible and that the version matches the tag, attaches an SBOM and build
provenance, waits for a maintainer to approve the `pypi` environment, publishes to PyPI with trusted publishing, and
creates the GitHub release from the changelog section. Then it opens the back-merge pull requests.

## Hotfixes

Branch `hotfix/<name>` from `main`, fix, add the changelog entry, and open a pull request into `main`. After merging,
roll the changelog over for the patch version (`python tools/release.py next --bump patch`) and tag it on `main` as
above. The back-merge pull requests carry the fix to `staging` and `dev`.

## If a release goes wrong

- PyPI never accepts the same version twice. Fix forward with a new patch release.
- A broken release can be **yanked** on PyPI (project settings), which stops new installs from choosing it without
  breaking pinned installs. Note the yank in the changelog.
- A failed workflow can be re-run from the Actions tab; publishing steps are idempotent for TestPyPI and refuse
  duplicates on PyPI.

## One-time setup

These steps are done once, by a maintainer with admin rights. None of them store a secret.

1. **PyPI and TestPyPI trusted publishing.** On [pypi.org](https://pypi.org/manage/account/publishing/) and
   [test.pypi.org](https://test.pypi.org/manage/account/publishing/), add a *pending publisher* (the project does not
   exist until its first upload):

   | Field | PyPI | TestPyPI |
   | --- | --- | --- |
   | Project name | `immune-ai` | `immune-ai` |
   | Owner / repository | `schwarzschlyle` / `immune` | `schwarzschlyle` / `immune` |
   | Workflow | `release.yml` | `release.yml` |
   | Environment | `pypi` | `testpypi` |

2. **GitHub environments** (Settings → Environments):
   - `testpypi`: no reviewers; deployment branches and tags `dev`, `staging` and `v*`.
   - `pypi`: required reviewer (the release manager); deployment tags `v*` only.
3. **Branch protection** (Settings → Rules → Rulesets, or Settings → Branches) for `main`, `staging` and `dev`:
   require pull requests, require the `ci` checks (including `promotion`), and block force pushes and deletion. For
   `main`, also require one approving review. Add a tag ruleset that lets only maintainers create `v*` tags.
4. **Actions permissions** (Settings → Actions → General): allow GitHub Actions to create pull requests, for the
   back-merge step.
