# Still to do

Housekeeping the codebase is carrying. Nothing here blocks a model run: every
non-DSGE model estimates and charts end to end. Ordered by value, not by size.

Counts are as measured, and go stale. Re-measure before starting:

```bash
uv run ruff check src --statistics                      # errors by rule
grep -rc "# noqa" src --include="*.py" | grep -v ":0"   # suppressions by file
```

---

## 1. Remove the `# noqa` suppressions

**190 directives, 178 of them outside DSGE.** The rule is in CLAUDE.md: agree a
rule-level ignore in `pyproject.toml` or fix the code, never suppress a line.
These predate that rule.

The reason this is first: a suppression carries a justification nobody
rechecks. Three in this repo asserted a circular import that did not exist, and
one of them hid a real defect for as long as it had been there.

Five rules are 86% of the total. Each is one decision, not N:

| rule | n | what it is | likely answer |
| --- | --- | --- | --- |
| `SLF001` | 77 | private access, mostly `ystar_ustar/analyse.py` setting `ystar_analyse._LFOOTER` and friends | the footer-override pattern is the real problem; fix the pattern, not the lint |
| `ANN401` | 31 | `Any` on pytensor expressions that have no useful type | rule-level ignore, or a `PyTensorLike` alias |
| `S301` | 18 | `pickle.load` of our own saved traces, 17 files | rule-level ignore: it is never untrusted input here |
| `PD013` | 13 | `.stack()` flagged as pandas when it is xarray | rule-level ignore: the rule cannot tell them apart |
| `PLR2004` | 6 | magic values, missed in the pass that named the rest | name them |

`SLF001` is the one with substance. 29 of the 77 are `ystar_ustar/analyse.py`
reaching into `ystar.analyse` and `ustar.analyse` to swap module-level footer
globals before drawing their charts, then putting them back. That works, but it
is why those modules cannot be read in isolation. Passing the footers as
arguments would remove the suppressions and the coupling together.

**Do not** run `ruff check --select <RULE> --fix`. Narrowing `--select`
deselects every other rule, so `RUF100` judges nearly every remaining directive
unused and strips them all: 283 across 111 files, in one command.

## 2. Decide what DSGE is

**146 of the 190 ruff errors are in `src/models/dsge/`,** a package that is
experimental, partly broken, and not scheduled. So `ruff check src` is 77% noise
about code nobody is working on, which buries the 44 findings in code that runs.

Two ways to stop that:

- `per-file-ignores` for `src/models/dsge/*` covering `ANN`, `PLR09`, `C901`,
  `D1`, `BLE`, `TRY`. Keep `F821`, `F401` and `F841` enforced: those catch real
  breakage, and the `F841` unused variables there (`cy`, `iy`, `r_t`, `M`) may
  be dropped terms in an equation rather than dead code.
- Or clean it, which is only worth it if the package is going somewhere.

**If it is developed further, consolidate the Blanchard-Kahn solver first.**
`solver.py` is already a complete general solver and nothing imports it; four
models each reimplement it. See `src/models/dsge/MODELS_EXPLAINED.md` under
Recommendations for the detail, including why adopting it is not a drop-in.

Two findings already sitting there, both DSGE-only:

- **`F841`, 8 sites** (`cy`, `iy` in `fa_nk_model`, `r_t` in
  `hlw_nairu_phillips_model`, `M` and `n_shocks` in `solver`/`kalman`). An
  assigned-but-unused local in model code is often a term that was meant to
  enter an equation. Worth an eye before dismissing as dead code.
- **`NPY002`, 4 sites.** `np.random.seed` / `randn` use one global RNG shared
  with every library in the process, so a "seeded" run is only reproducible if
  nothing else drew first. `default_rng(seed)` is isolated. Fixing changes which
  numbers come out, so do it deliberately.

Also unrecorded: what "a bit broken" means. Running the DSGE models once and
writing the failures into the notes would cost ten minutes and save
rediscovering them cold.

**One cheap experiment, for whenever this is picked up.** `fa_nk_bayes.py` is
the only file in the package with a sampling path, and replacing the MLE hard
bounds with explicit priors is what took phi_pi off its cap for FA-NK and
FA-NK-wage. `NK` and `NK-TwoStar` have never had that treatment, and their
recorded symptoms ("many params at bounds", "phi_pi at the determinacy floor")
are the same thing it fixed. `run_bayes` takes a `ModelSpec` and both models
have one, so pointing it at them is a small change. The answer informs either
way: if phi_pi still pegs under a proper prior, that is a finding about the
data rather than a numerical artefact.

## 3. Loose ends in maintained code

- **`PLR0917`, 38 sites, 23 maintained.** More than five positional arguments;
  the worst take 11. Reviewed and deliberately left: the one-character `*` fix
  needs every caller checked, and the tempting alternative (bundle into a config
  object) trades this for too-many-locals. Revisit only if an argument-order bug
  actually bites.
- **mypy.** Around 500 findings, nearly all noise from pandas-stubs and arviz's
  dynamic attributes. Worth running after a refactor anyway: its `call-arg`
  check found a silently dropped chart label that ruff could not see.

## 4. Refactor suggestions considered and not taken

Recorded so they are not re-proposed from scratch:

- **Generic summary engine** for the five `*_summary` packages. Rejected: the
  weight is in each `sources.py`, which encodes which model and which flags to
  run. A spec covering all five needs refresh-off-by-default, re-run-if-stale,
  per-spec prefixes, nominal conversion and two-equation LOO. That is a
  thirty-switch generic, and the shells it would replace are 24 to 55 lines.
- **Deduplicating `observations.py`.** Left alone: observation construction is
  part of a model's specification. Deliberate reuse where two models genuinely
  consume the same object is fine and already happens.
