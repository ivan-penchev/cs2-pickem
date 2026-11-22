# cs2-pickem

Monte Carlo predictions and Pick'Em picks for Counter-Strike 2 Majors. It is set up for the PGL Major Singapore 2026 (Nov 24 to Dec 13). Another stage or another Major is another YAML file under `configs/`.

The live dashboard is at <https://ivan-penchev.github.io/cs2-pickem/>.

For a Swiss stage it picks 2 teams to go 3-0, 2 to go 0-3 and 6 to advance, choosing the set with the highest chance of 5 or more correct, which is what Valve's coin needs. For the playoffs it picks the bracket with the most expected correct picks.

## Quick start

```bash
make install                      # creates .venv and installs the package with dev extras
source .venv/bin/activate

cs2pickem rate -s stage1          # team ratings and their components
cs2pickem simulate -s stage1      # P(3-0), P(advance), P(0-3) and record distribution
cs2pickem pickem -s stage1        # recommended Pick'Em
cs2pickem matchup NiP M80 --bo 3  # series odds, per-map odds and the predicted veto
```

Requires Python 3.12 or newer. Everything runs offline from the data committed in `data/`.

## Commands

| command | what it does |
|---|---|
| `fetch [--refresh] [--source vrs\|hltv]` | Lists the committed snapshots. `--refresh` scrapes VRS and HLTV into new ones |
| `rate [-s STAGE]` | Rating table with every component |
| `matchup A B [--bo N]` | Series win probability, per-map probabilities and the predicted veto |
| `simulate -s STAGE [-n SIMS] [--no-results] [--flat] [--json FILE]` | Simulates a Swiss or playoff stage |
| `pickem -s STAGE [--lock '3-0=Team'] [-n SIMS]` | Recommended Pick'Em for a Swiss stage or the playoff bracket |
| `playoffs` | Bracket simulation, with teams from config, Stage 3 results or a projection |
| `result add A B WINNER [-s STAGE] [--round R]` | Records a finished match |
| `result list` / `result remove A B` | Shows standings, or removes a result |
| `next-stage -s STAGE [--write]` | Seeds the next stage from invites and qualifiers. `--write` saves it to the YAML |
| `animate -s STAGE [-n SIMS] [--runs 300] [--out FILE] [--open]` | Writes the animated HTML page for a stage |
| `live update [--force] [--today YYYY-MM-DD]` | Scrapes finished matches into `data/results/` |
| `site build [-n SIMS] [--no-record] [--note TEXT]` | Re-runs every stage, appends to the prediction history and rebuilds `site/`. `--note` forces a history entry, for example after a model change |
| `backtest prepare -e MAJOR` / `backtest run [-e MAJOR]` | Point-in-time backtest of past Majors |

`-e/--event` picks a directory under `configs/`. It defaults to the first one. `--flat` turns off the map and veto model and uses the team ratings alone.

## Entering results by hand

The GitHub Action below does this automatically. To do it locally:

```bash
cs2pickem result add "Ninjas in Pyjamas" M80 "Ninjas in Pyjamas"   # round is inferred
cs2pickem result list -s stage1
cs2pickem pickem -s stage1               # re-simulated with these results fixed
cs2pickem next-stage -s stage2 --write   # after Stage 1 finishes
```

Results go to `data/results/<stage>.yaml`, which you can also edit directly. Team names accept any alias listed in `event.yaml`. When a later stage has no teams yet, the tool simulates the earlier stage to guess them and labels the output as projected.

## Automatic updates during the event

`.github/workflows/live.yml` runs at minute 17 of every hour. Outside the event dates (plus one day either side) it exits straight away. Running it by hand with `force: true` skips that check. Inside the window it:

1. Scrapes finished matches. It tries HLTV with headless Playwright, then the Liquipedia MediaWiki API (HLTV blocks most CI runners), then DFRAG.
2. Merges them into `data/results/<stage>.yaml`. A failed or partial scrape never deletes or changes a recorded result.
3. Writes the next stage's seeding once a Swiss stage has 8 qualifiers.
4. Runs `cs2pickem site build`, commits `data/` and `site/` to `main` with `[skip ci]`, and deploys GitHub Pages.

A scrape failure does not fail the job. The error goes to the job summary and nothing is changed.

Each `site build` that sees new results, or gets a `--note`, saves a snapshot to `data/predictions/<stage>/<UTC timestamp>.json`. The first one for a stage is the baseline and is never overwritten. The dashboard compares the baseline with the latest snapshot, plots each team's odds over time with the match that moved them, and grades the original picks as the stage plays out.

`pages.yml` also deploys when a push to `main` changes `site/`.

## Animated page

```bash
cs2pickem animate -s stage1 --open   # writes site/stage1.html
```

The page is one HTML file with no external scripts, so it works offline. It replays 300 logged runs on a Swiss bracket while P(3-0), P(advance) and P(0-3) settle over all N simulations, with 95% intervals. It also shows the recommended picks with a P(≥5) gauge, the share of upsets and a team-by-record heatmap. Space pauses, the right arrow steps one run. Add `?ff` to the URL to jump to the end or `?paused` to stop autoplay.

## Backtests

`cs2pickem backtest run` replays every CS2 Major with a Swiss format using only data from before each stage, meaning the VRS file and HLTV ranking from that date and match history cut at the stage start. That is 13 Swiss stages: PGL Copenhagen 2024 and Perfect World Shanghai 2024 have two each, and BLAST Austin 2025, StarLadder Budapest 2025 and IEM Cologne 2026 have three each. Copenhagen predates the Valve Regional Standings, so its VRS input is a synthetic table built from the HLTV ranking of 2024-03-11, and it has no point-in-time match history. The report says so on the page.

Match-level Brier over all 464 matches (lower is better):

| event | matches | BT model | BT+VRS ensemble | old blend | VRS-only | coin flip |
|---|---|---|---|---|---|---|
| Copenhagen 2024 | 73 | 0.233 | 0.244 | 0.263 | 0.263 | 0.250 |
| Shanghai 2024 | 73 | 0.210 | 0.208 | 0.212 | 0.213 | 0.250 |
| Austin 2025 | 106 | 0.248 | 0.250 | 0.255 | 0.256 | 0.250 |
| Budapest 2025 | 106 | 0.224 | 0.234 | 0.248 | 0.250 | 0.250 |
| Cologne 2026 | 106 | 0.216 | 0.222 | 0.229 | 0.234 | 0.250 |

What matters for the coin is correct picks, so the backtest also compares pick strategies. All of them use the same simulator (20k sims, fixed seed) and optimiser. Only the win probabilities, the objective and the tie-break change. The intervals are bootstrap 95% CIs over the 13 stages, and the differences are paired per stage.

| strategy | mean correct of 10 | vs VRS-only | coins | Brier |
|---|---|---|---|---|
| BT, max P(≥5) (default) | 4.31 [3.69, 4.85] | +0.15 [-0.54, +0.92] | 7/13 | 0.227 |
| BT, seed-anchored tie-break | 4.31 [3.69, 4.92] | +0.15 [-0.46, +0.77] | 8/13 | 0.227 |
| BT+VRS 50/50 on probabilities | 4.23 [3.46, 4.92] | +0.08 [-0.46, +0.69] | 7/13 | 0.231 |
| VRS seeds only | 4.15 [3.31, 4.92] | 0 | 7/13 | 0.244 |
| BT+VRS 50/50 on logits | 4.15 [3.38, 4.85] | +0.00 [-0.46, +0.46] | 7/13 | 0.232 |
| old blend | 4.15 [3.54, 4.77] | +0.00 [-0.31, +0.31] | 6/13 | 0.242 |
| BT, max E[correct] | 4.23 [3.62, 4.77] | +0.08 [-0.62, +0.85] | 6/13 | 0.227 |

The default stays BT optimised for P(≥5). It has the highest mean correct picks, and it was highest or tied for highest on all three simulation seeds tried at 100k sims. The honest reading is that the strategies cannot be told apart. Every paired difference has a CI that straddles zero, and changing only the simulation seed moves a strategy by about 0.2 picks per stage, which is as large as the gaps between them. Since nothing beats BT and it has the best match probabilities, there is no reason to switch. Optimising expected correct picks instead of P(≥5) never helped. Expect a bit over 4 correct picks and a coin in roughly half of the stages. Full results, including per-category hit rates and every pick, are in `site/backtest.html`.

## Swiss rules

These follow Valve's Major Supplemental Rulebook.

* 16 teams. Three wins advance, three losses eliminate.
* Round 1 uses the official draw if the config has one, otherwise seed *i* plays seed *i+8*.
* From round 2, teams with the same record are ordered by Buchholz (sum of opponents' wins minus losses), then by seed.
  * Rounds 2 and 3 pair the top team with the lowest team it has not played yet.
  * Rounds 4 and 5 use Valve's six-team priority table and take the first row without a rematch.
* Matches where either team can advance or be eliminated are Bo3. All others are Bo1.
* The higher-placed team is the higher seed. In a Bo3 it is team A in the veto. In a Bo1 it chooses whether to be team A or team B.
* Final order is wins, losses, Buchholz, seed.

Each stage YAML can change `wins_to_advance`, `losses_to_eliminate`, `default_bo` and `decider_bo`. Stage 3 sets `default_bo: 3`.

## Model

**Rating.** The default (`method: bt`) is a Bradley-Terry fit over every series in the last 180 days, including opponents outside the event, with each match's weight halving every 120 days. Each team's prior is centred on its VRS points, at half weight, with an L2 strength of 3. Matches played with fewer than 4 current players, or before a recorded roster change, count at 0.35. The fitted spread around 1500 is then multiplied by a temperature of 0.85, because the backtests showed the raw ratings were overconfident. `data/ratings_override.csv` adds manual deltas on top.

The older blend is still available with `method: blend`. It z-scores VRS points, HLTV points (log scale) and a form Elo across the event and computes `1500 + 200 · weighted mean` before the same temperature, with weights 0.4 VRS, 0.25 HLTV and 0.35 Elo. That Elo covers the last 90 days, starts from VRS and halves a match's weight every 30 days. All of these numbers live under `rating:` in `event.yaml`.

**Maps.** A team's win rate on each map is pulled toward its overall map win rate by 8 pseudo-games, then turned into a logit offset capped at ±0.8. The offsets move strength between maps without changing the total. Map win probability is `logit P = ln10·(R_i − R_j)/400 + off_i[m] − off_j[m]`.

**Veto.** Minimax over the 7-map pool. Bo1 is A ban, A ban, B ban ×3, A ban, decider. Bo3 is A ban, B ban, A pick, B pick, B ban, A ban, decider. Setting `veto_temperature` above 0 makes teams veto less perfectly.

**Simulation.** 100k Swiss runs take about 5 seconds. Recorded results are fixed in every run. Playoffs are an 8-team bracket (1v8, 4v5, 2v7, 3v6) with Bo3 quarters and semis and a Bo5 final.

**Pick'Em.** The Swiss optimiser scores each candidate set against the simulated runs directly, so it knows two teams that meet early can't both go 3-0. It starts greedy, then swaps picks with random restarts. `--lock` fixes picks you've already made. The bracket optimiser uses dynamic programming and only advances teams you picked in the previous round.

## Data

See [DATA_SOURCES.md](DATA_SOURCES.md) for sources and licences.

| path | contents |
|---|---|
| `data/snapshots/vrs/<date>/` | Valve VRS standings and match details |
| `data/snapshots/hltv/<date>/` | HLTV ranking, recent results and map stats |
| `data/latest.json` | Which snapshots the tool reads |
| `data/teams.csv` | Manual VRS or HLTV points, overriding the snapshot |
| `data/matches.csv` | Extra series results for the rating fit |
| `data/maps.csv` | Per-map record overrides |
| `data/ratings_override.csv` | Manual Elo deltas. +50 is about +7 points of win chance against an equal team |
| `data/results/<stage>.yaml` | Match results for the current event |
| `data/predictions/<stage>/` | Prediction history. The first file is the baseline |
| `data/backtests/<major>/` | Backtest inputs and reports |

Scraping needs Playwright, which `make install-scrape` installs.

## Adding a stage or a Major

1. Copy `configs/singapore2026/` to `configs/<event>/`.
2. In `event.yaml`, set the dates, map pool and teams with aliases and rosters.
3. In each stage YAML, list teams in seed order with the Round 1 draw, or a `source:` with invites.
4. Run `cs2pickem fetch --refresh -e <event>`, then `cs2pickem pickem -e <event> -s stage1`.

## Development

```bash
make test    # pytest
make lint    # ruff
```

CI runs ruff, pytest and an offline CLI smoke test on every push and pull request. See [CONTRIBUTING.md](CONTRIBUTING.md).

Fan project. Not affiliated with Valve, PGL, HLTV or Liquipedia.
