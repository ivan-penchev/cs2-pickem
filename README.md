# cs2-pickem

Predictive analytics and an optimiser for **Counter-Strike 2 Major Pick'Em** (Valve's in-game Pick'Em).
It ships configured for the **PGL Major Singapore 2026**. Every stage and every future Major is just another YAML file.

* Team strength: Valve VRS points + HLTV ranking + recent-form Elo (current roster only), with manual overrides
* Per-map strength (Bayesian-shrunk) and a game-theoretic **map veto** for Bo1 / Bo3 / Bo5
* A fast **Monte Carlo Swiss** simulator that follows Valve's Major pairing rules (Buchholz, no rematches, Bo3 deciders)
* A single-elimination **playoff bracket** simulator (Bo3, Bo5 final)
* **Pick'Em optimisers**:
  * Swiss: maximise P(≥5 of 10 correct), using the joint simulated outcomes rather than marginals
  * Bracket: maximise expected correct picks
* **Live mode**: enter finished matches and every remaining round is re-simulated conditional on them.
  Stage N+1 is then built from the Stage N qualifiers plus the direct invites.
* Offline by default from committed data snapshots. `fetch --refresh` re-scrapes the sources.
  User-editable CSVs override anything.

## Quick start

```bash
make install                      # python3 -m venv .venv && pip install -e '.[dev]'
source .venv/bin/activate

cs2pickem rate -s stage1          # blended ratings
cs2pickem simulate -s stage1      # P(3-0) / P(qualify) / P(0-3) + record distribution
cs2pickem pickem -s stage1        # optimal 2x 3-0, 2x 0-3, 6x advance
cs2pickem matchup NiP M80 --bo 3  # head-to-head, map probabilities, predicted veto
```

Python ≥ 3.12. With uv instead of pip: `uv venv && uv pip install -e '.[dev]'`.

## CLI

| command | what it does |
|---|---|
| `fetch [--refresh] [--source vrs\|hltv]` | Show the committed snapshots, or re-scrape VRS (GitHub) and HLTV (Playwright) into new snapshots |
| `rate [-s STAGE]` | Rating table showing every component |
| `matchup A B [--bo N]` | Series win probability, per-map probabilities and the predicted veto line |
| `simulate -s STAGE [-n SIMS] [--no-results] [--flat] [--json out.json]` | Simulate a Swiss or playoff stage |
| `pickem -s STAGE [--lock '3-0=Team'] [-n SIMS]` | Optimal Pick'Em for a Swiss stage or the playoff bracket |
| `playoffs` | Bracket simulation. Teams come from config, from Stage 3 results, or from projection |
| `result add A B WINNER [-s STAGE] [--round R]` | Record a finished match (live mode) |
| `result list / result remove A B` | Show standings and results, or undo a result |
| `animate -s STAGE [-n SIMS] [--runs 300] [--out site/STAGE.html] [--open]` | Write an animated, self-contained HTML visualisation of the simulation |
| `next-stage -s stage2 [--write]` | Seed the next stage (invites + qualifiers). `--write` stores the seeding in the YAML |

`--event` selects the config directory under `configs/`. The default is the only (or first) event.
`--flat` turns off the map/veto model, which leaves plain Elo.

### Live mode during an event

```bash
cs2pickem result add "Ninjas in Pyjamas" M80 "Ninjas in Pyjamas"   # round inferred from standings
cs2pickem result add Liquid GOSH GOSH
cs2pickem result list -s stage1
cs2pickem pickem -s stage1               # conditional on the results so far
# once Stage 1 is over:
cs2pickem next-stage -s stage2 --write
cs2pickem pickem -s stage2
```

Results are stored in `data/results/<stage>.yaml`, which you can edit by hand. Team names accept any alias from `event.yaml`.
Later stages that are not decided yet are **projected**: Stage 1 is simulated to obtain the most likely qualifiers.
Projected output is labelled as such.

## Animated visualisation

```bash
cs2pickem animate --stage stage1 --open        # → site/stage1.html (or: make animate STAGE=stage1)
cs2pickem animate --stage playoffs             # → site/playoffs.html
```

![Animated Swiss bracket – screenshot placeholder](docs/animate.png)

<!-- TODO: replace docs/animate.png with a screenshot/GIF of site/stage1.html -->

The output is **one static HTML file**. It has no build step and no CDN; it uses vanilla JS with SVG/CSS and works offline, so you can share it as an attachment or on GitHub Pages.
The Python side embeds a compact JSON payload of about 0.3 MB for 100k sims:

* **Replay runs.** About 300 fully logged simulated runs, with every match as round, teams, Bo1/Bo3, winner and whether it was a live result. Each match also carries its win probability and predicted veto line.
* **Convergence tallies.** Cumulative counts over *all* N sims at about 240 log-spaced checkpoints: per-team final record, the correct-pick histogram and upsets. The browser animates 1 → N without the raw data.

On screen you get:

* **Swiss bracket.** It uses the classic Major layout: 0-0 → 1-0/0-1 → … → 3-x advanced / x-3 eliminated. Team badges move between record buckets round by round. Gold marks a Bo3 decider. Hover a match to see its win probability and likely maps.
  For playoffs, the bracket fills in up to the champion.
* **Converging probability bars.** Bars for P(3-0), P(advance) and P(0-3), or P(semis), P(final) and P(champion), each with a 95% CI band and a tick for the final value. A sim counter and a sims/sec display sit alongside.
* **Pick'Em panel.** The recommended picks, a live P(≥5 correct) gauge and a histogram of the number of correct picks. Picks hit or missed in the current replay are highlighted.
* **Chaos meter.** The share of matches won by the pre-match underdog.
* **Heatmap.** The final record distribution, team × record.

Controls are play/pause (space), step one run (→), fast-forward to N, reset and a speed slider.
URL flags: `?ff` jumps straight to the converged state and `?paused` disables autoplay.
Recorded live results (`data/results/<stage>.yaml`) are respected: those matches are fixed in every run and marked ✓.
`site/stage1.html` and `site/playoffs.html` are committed as a demo. `.github/workflows/pages.yml` can publish `site/` via GitHub Pages; enable it under Settings → Pages → Source: GitHub Actions.

## Swiss rules implemented (Valve Major Supplemental Rulebook)

* 16 teams. 3 wins advance, 3 losses eliminate, at most 5 rounds.
* **Round 1**: the official draw if one is configured, otherwise seed *i* vs *i+8*.
* **Rounds 2–5**: teams are grouped by W-L record. Within a group, teams are ordered by Buchholz
  (the sum of opponents' current W−L), then by initial seed.
  * Rounds 2–3: the top team plays the lowest team it has not met yet. Backtracking guarantees a full pairing.
  * Rounds 4–5: groups of six use Valve's priority table (1v6 2v5 3v4, 1v6 2v4 3v5, …).
    The first row with no rematch is chosen.
* **Bo3** for every match where either team is at 2 wins or 2 losses (advancement or elimination). All other matches are **Bo1**.
* In each pairing the better-placed team is the higher seed. It is team A in the veto, and in Bo1 it gets the last ban.
* Final standings: wins, then losses, then Buchholz, then seed. The top half qualifies, seeded in that order.

Everything is configurable per stage (`wins_to_advance`, `losses_to_eliminate`, `default_bo`, `decider_bo`).
For example, `configs/singapore2026/stage3.yaml` sets `default_bo: 3`.

## Model

### 1. Team rating (`rating/`)
Each component is z-scored across the event's teams:
* **VRS** points
* **HLTV** points (log scale)
* **Form Elo**: an Elo run over the last `elo_window_days` of results, starting from VRS as the prior.
  Matches are exponentially time-weighted (`elo_half_life_days`).
  Matches played with a different roster (fewer than `min_roster_overlap` current players) count at `old_roster_weight`.

The rating is `1500 + rating_scale · Σ w_k z_k / Σ w_k`, plus any manual `delta` from `data/ratings_override.csv`.
If a component is missing, its weight is dropped and the remaining weights are renormalised.
All weights are set under `rating:` in `event.yaml`.

### 2. Maps and veto (`maps/`)
* **Per-map offsets.** A team's record on each Active Duty map is shrunk towards its own overall map win rate,
  using `prior_maps` pseudo-games, and converted to a logit offset clamped to `±max_offset`.
  The offsets only redistribute the team's strength across maps; they never change its overall level.
* **Map win probability**: `logit P = ln10·(R_i − R_j)/400 + off_i[m] − off_j[m]`.
* **Veto** is solved exactly by minimax over the 7-map pool:
  * Bo1: A ban, A ban, B ban×3, A ban, then the decider.
  * Bo3: A ban, B ban, A pick, B pick, B ban, A ban, then the decider.
  * Bo5: ban, ban, then picks.

  Set `veto_temperature > 0` for a softer, less-than-perfect veto policy.
  The series probability is the probability of winning a majority of the selected maps.

### 3. Swiss simulator (`swiss/`)
* 100k simulations take about 5 s.
* Pairings for rounds 1–3 depend only on the result history, so they are cached.
* Known results (live mode) are forced, including partly played rounds.
* Output per team:
  * P(3-0), P(qualify) and P(0-3)
  * the full record distribution
  * per-simulation outcome masks, which the optimiser uses

### 4. Playoffs (`playoffs.py`)
* 8 teams, single elimination. QF and SF are Bo3; the Grand Final is Bo5 (configurable).
* The bracket is `[[1,8],[4,5],[2,7],[3,6]]`.

### 5. Pick'Em optimisers (`pickem/`)
* **Swiss**: choose 2 teams for 3-0, 2 for 0-3 and 6 for advance (3-1/3-2), with no team used twice.
  * The objective is P(correct ≥ 5), evaluated on the simulated outcomes themselves, so correlations count.
    For example, two teams that would meet in a 2-0 Bo3 cannot both finish 3-0.
  * Search is a greedy start followed by swap/move local search with random restarts.
  * The output also reports E[correct] and the full distribution of correct picks.
  * `--lock` forces specific picks.
* **Bracket**: dynamic programming over the bracket tree. It maximises expected correct picks under the
  consistency constraint that a pick can only advance a team you picked in the previous round.

## Stage 1 prediction (pre-event, snapshot 2026-11-22, 100k sims)

| pick | team | P(correct) |
|---|---|---|
| 3-0 | PARIVISION | 31.5% |
| 3-0 | Liquid | 27.0% |
| advance | FaZe, HEROIC, NiP, Eternal Fire, 100 Thieves, M80 | 52–56% each |
| 0-3 | Chinggis Warriors | 51.7% |
| 0-3 | Lynn Vision | 42.8% |

The full set has **P(≥5 correct) = 56.5%** and E[correct] = 4.77.
To reproduce it, run `cs2pickem pickem -s stage1`.

## Data (`data/`, see [DATA_SOURCES.md](DATA_SOURCES.md))

| file | purpose |
|---|---|
| `snapshots/vrs/<date>/` | Valve VRS standings and match details (from `ValveSoftware/counter-strike_regional_standings`) |
| `snapshots/hltv/<date>/` | HLTV ranking, recent results and per-map stats |
| `latest.json` | Which snapshot the tool uses |
| `teams.csv` | Manual VRS/HLTV points that override the snapshot |
| `matches.csv` | Extra series results for the form Elo |
| `maps.csv` | Per-map record overrides |
| `ratings_override.csv` | Gut-feel Elo adjustments (`+50` ≈ +7 pp vs an equal team) |
| `results/<stage>.yaml` | Live results |

`cs2pickem fetch --refresh` needs the scrape extra: `pip install -e '.[scrape]' && playwright install chromium`.
It caches pages in `.cache/` and rate-limits politely.

## A new stage or a new Major

1. Copy `configs/singapore2026/` to `configs/<event>/`.
2. Edit `event.yaml`: dates, map pool, team registry with aliases and rosters.
3. Edit the per-stage YAMLs: teams in seed order and the Round-1 draw, or a `source:` with invites.
4. Run `cs2pickem fetch --refresh -e <event>`, then `cs2pickem pickem -e <event> -s stage1`.

## Development

```bash
make test      # pytest
make lint      # ruff
make predict   # simulate + pickem for STAGE=stage1
make animate   # site/$STAGE.html
```

CI (`.github/workflows/ci.yml`) runs ruff, pytest and an offline CLI smoke test.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Disclaimer

This is a fan-made statistical tool with no affiliation to Valve, PGL, HLTV or Liquipedia.
The predictions are probabilities, not guarantees.
