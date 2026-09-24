# Still to do

Housekeeping the codebase is carrying. Nothing here blocks a model run: every
non-DSGE model estimates and charts end to end. Ordered by value, not by size.

Counts are as measured, and go stale. Re-measure before starting:

```bash
uv run ruff check src --statistics                      # errors by rule
grep -rc "# noqa" src --include="*.py" | grep -v ":0"   # suppressions by file
```

---

## 1. Fix what the `# noqa` suppressions hid

**No directives remain, so ruff reports everything: 224 errors, 64 of them
outside DSGE.** The rule is in CLAUDE.md: agree a rule-level ignore in
`pyproject.toml` or fix the code, never suppress a line.

The reason this is first: a suppression carries a justification nobody
rechecks. Three in this repo asserted a circular import that did not exist, and
one of them hid a real defect for as long as it had been there.

What is left outside DSGE:

| rule | n | what it is | likely answer |
| --- | --- | --- | --- |
| `PLR0917` | 20 | more than five positional arguments | see section 3 |
| complexity (`C901`, `PLR0912`, `PLR0915`) | 15 | long functions: `nairu/estimate.py`, `gdp_nowcast_bvar/model.py`, `cobb_douglas/model.py` and others | refactor only where a function is being worked on anyway |
| `ARG001` | 7 | unused function arguments, mostly `nairu/equations` | check each: an unused argument in an equation can be a dropped term |
| `SLF001` | 6 | private access | two patterns, below |
| singletons | 16 | unused imports, `l` as a name, docstring format, one blind except | mechanical |

The six `SLF001`s are two patterns, each one decision:

- **`model._descriptions` / `model._config` set on a PyMC model** (`nairu/estimate.py`,
  `rstar_hlw/estimate.py`). The same job `common/model_constants.py` does for
  imposed constants: carry run information on an object PyMC owns, through
  functions in one file, under a name unlikely to collide.
- **`results._extra(...)` called from outside** (`rstar_bonds/analyse.py`,
  `rstar_rba/analyse.py`). A method other packages need is public by use; make
  it so.

**Do not** run `ruff check --select <RULE> --fix`. Narrowing `--select`
deselects every other rule, so `RUF100` judges nearly every remaining directive
unused and strips them all: 283 across 111 files, in one command.

## 2. Decide what DSGE is

**160 of the 224 ruff errors are in `src/models/dsge/`,** a package that is
experimental, partly broken, and not scheduled. So `ruff check src` is 71% noise
about code nobody is working on, which buries the 64 findings in code that runs.

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

- **`F841`, 9 sites** (`cy`, `iy` in `fa_nk_model`, `r_t` in
  `hlw_nairu_phillips_model`, `M` and `n_shocks` in `solver`/`kalman`, `model`
  in `fa_nk_bayes`). An
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

- **`PLR0917`, 35 sites, 20 maintained.** More than five positional arguments;
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
