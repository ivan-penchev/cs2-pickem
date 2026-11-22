# Data sources

| source | used for | access | licence / terms |
|---|---|---|---|
| [Valve Regional Standings](https://github.com/ValveSoftware/counter-strike_regional_standings) | VRS points, seeding, match history | `raw.githubusercontent.com` / GitHub API (httpx) | Published by Valve for public use; data © Valve |
| [HLTV.org](https://www.hltv.org) | World ranking points, recent results, per-map stats | Headless Playwright, one page at a time, ≥4 s apart, cached | © HLTV. Personal and non-commercial research use only; respect their ToS |
| [Liquipedia](https://liquipedia.net/counterstrike) | Rosters and stage formats (checked by hand, encoded in `configs/`) | Manual / API with ≥2.5 s rate limit | CC BY-SA 3.0 |
| PGL / Valve rulebooks | Swiss and playoff format, Pick'Em scoring | Manual | n/a |

The snapshots in `data/snapshots/` are small derived tables (team-level aggregates), included so the tool works offline and results can be reproduced.
`data/latest.json` records which snapshot date the tool uses.
If you are a rights holder and want something removed, open an issue.

Fallback and override files (`data/*.csv`) are user-maintained and take precedence over the snapshots.
