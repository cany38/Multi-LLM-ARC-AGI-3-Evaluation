# Candidate task chains for transfer experiments

This note lists candidate `T1` to `T2` (and three-step) task chains selected from the 25 surveyed offline games in `data/survey/games.csv` and their initial frames in `data/survey/frames/`.

## Selection criteria

All 25 surveyed games use a 64x64 grid, support starting at an arbitrary level (`can_start_at_level=True`), reset deterministically (`repeatable=True`), and run offline (`works_offline=True`). To keep budget consumption predictable under `budget = round(baseline_actions * 3.0)`, candidate chains start on levels with low baseline action counts and either share an input modality (`ACTION1` to `ACTION4` movement vs. `ACTION6` click) or share visible interface structures such as HUD progress bars, reference panels, or container targets.

## Within-game level chains (baseline sanity transfer)

Running level 0 as `T1` and level 1 as `T2` keeps game physics identical and checks whether extracted notes or ontology statements improve transfer before testing cross-game pairs.

| Game ID | Actions | Baselines (levels 0 to 2) | Notes |
| --- | --- | --- | --- |
| `tu93-0768757b` | `ACTION1` to `ACTION4` | 19, 16, 34 | Small budgets; bottom border and path layout |
| `ls20-9607627b` | `ACTION1` to `ACTION4` | 22, 123, 73 | Progress bar, life counter, pickup object, corridor walls |
| `ft09-0d8bbf25` | `ACTION6` | 43, 12, 23 | Pure click; 2x2 grid of 3x3 red/blue blocks |
| `cd82-fb555c5d` | `ACTION1` to `ACTION6` | 55, 8, 41 | Top-left color reference panel plus board playfield |
| `lp85-305b61c3` | `ACTION6` | 17, 38, 31 | Pure click; color ring selector and step dash meter |

## Cross-game candidate chains

### Chain 1: Grid navigation and goal reaching

`tu93-0768757b` (level 0, baseline 19) -> `ls20-9607627b` (level 0, baseline 22) -> `wa30-ee6fef47` (level 0, baseline 71)

All three frames show discrete corridor or bounded grid worlds navigated primarily with `ACTION1` through `ACTION4` (`wa30-ee6fef47` also exposes `ACTION5`, though its `changing_actions` column lists only `ACTION1` through `ACTION4`). Both `tu93` and `wa30` share a magenta bottom boundary frame, while `ls20` introduces an explicit bottom HUD (yellow step bar, red life indicators, inventory slot) and pickups before opening a barrier. `games.csv` tags `tu93-0768757b` as `keyboard_click`, even though its action list only includes `ACTION1` to `ACTION4`.

### Chain 2: Click-only spatial alignment

`vc33-5430563c` (level 0, baseline 7) -> `s5i5-18d95033` (level 0, baseline 20) -> `su15-1944f8ab` (level 0, baseline 22)

Every game in this chain uses `ACTION6` as its main changing action (`su15` also exposes `ACTION7`). Baselines are small (7 to 22 actions), keeping multi-seed comparisons cheap. Visually, `vc33` shows a minimal horizontal connector between gray/white blocks and a blue target, `s5i5` shows colored bars paired with matching U-shaped sockets and red markers, and `su15` shows a diagonal dotted alignment line from a corner square to a blue ring.

### Chain 3: Pattern grids and color-coded reference panels

`lp85-305b61c3` (level 0, baseline 17) -> `ft09-0d8bbf25` (level 0, baseline 43) -> `sc25-635fd71a` (level 0, baseline 36)

These games organize the playfield into regular tile blocks with a small color palette. `lp85` and `ft09` accept only `ACTION6`; `sc25` adds directional keys `ACTION1` through `ACTION4` alongside `ACTION6` over a 3x3 layout and a four-way control shape. Transfer here tests whether knowledge about tile selection, local symmetry, or pattern matching carries into mixed keyboard-and-click play.

### Chain 4: Reference HUD plus board manipulation

`sb26-7fbdac44` (level 0, baseline 18) -> `cd82-fb555c5d` (level 0, baseline 55) -> `tr87-cd924810` (level 0, baseline 54)

All three initial frames place a structured color/symbol panel along the top or bottom border separate from the active center region. `sb26` shows four outlined color slots at the top and four matching color tiles at the bottom; `cd82` shows a two-swatch reference box top-left; `tr87` shows a cyan/magenta glyph legend above the playfield and a five-glyph strip below it.

### Chain 5: Room barriers and two-container mechanics

`ka59-38d34dbb` (level 0, baseline 28) -> `sk48-d8078629` (level 0, baseline 61) -> `dc22-fdcac232` (level 0, baseline 59)

All three use `keyboard_click` (`ACTION1` to `ACTION4` plus `ACTION6`; `sk48` adds `ACTION7`) and divide the board into distinct functional zones: bridged rooms and a purple divider in `ka59`, paired yellow U-buckets in `sk48`, and a left grid next to right-side button panels in `dc22`.

## Notes on unverified assumptions

Visual similarity in level-0 frames does not guarantee shared transition rules inside the hidden game code. Before committing full evaluation budgets to a chain, run `T1` alone under `nothing` and inspect `runs/<run_id>/steps.jsonl` alongside the extracted knowledge in `data/knowledge/` to verify whether `T1` facts generalize to `T2` without leaking task-specific hints.
