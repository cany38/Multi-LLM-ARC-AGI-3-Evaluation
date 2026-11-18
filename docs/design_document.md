# Design document: neurosymbolic transfer learning in ARC-AGI-3

Authors: Can, Nina
Supervisor: Matteo Magnini (University of Luxembourg, DCS)

## 1. Concept

Interactive puzzle environments in ARC-AGI-3 give an agent raw 2D color grids and a restricted discrete action set (`ACTION1` to `ACTION7`, with optional click coordinates `x`, `y` on `ACTION6`) without exposing game rules or level goals in advance. An AI agent plays a source task `T1` (`A1 -> T1`), extracts generalizable mechanics `K1` from its experience, and supplies `K1` to a second agent solving a related target task `T2` (`A2(K1) -> T2`).

The system follows a neurosymbolic separation. A symbolic grounder converts each 64x64 color matrix into connected components and frame-to-frame object diffs (`moved`, `recolored`, `appeared`, `disappeared`, `changed_pixels`). A subsymbolic policy model plans short action sequences from that JSON state representation. `T1` runs once per seed, and `Runner` derives the transferred knowledge for all four conditions from that same source run and the same verified fact set:

| Condition | Input provided to `A2` on `T2` | Purpose |
| --- | --- | --- |
| `nothing` | No prior context | Baseline without transfer |
| `trajectory` | Compact `(action, diff, game_state)` log from `T1` | Raw context control |
| `notes` | Plain-language rules extracted and verified from `T1` | Free-text symbolic transfer |
| `ontology` | OWL ontology in Turtle syntax derived from verified rules | Formal symbolic transfer |

The project evaluates whether `K1` reduces actions, model calls, tokens, and elapsed time on `T2`, and whether the ontology representation outperforms free-text notes and raw history. Solving all ARC-AGI-3 games is outside the scope of this work.

## 2. Requirements elicitation and analysis

### 2.1 Domain glossary

| Term | Definition |
| --- | --- |
| Task `(game_id, level)` | A single playable level of an offline or hosted ARC-AGI-3 puzzle |
| Chain | An ordered list of tasks `(T1, T2, ...)` evaluated sequentially with accumulated knowledge |
| Oracle | Provider-agnostic interface exposing `query`, `generate`, and `choose` over sub-symbolic models |
| Grounding | Deterministic pixel-to-object segmentation and frame diff extraction |
| Knowledge (`K1`) | Transferred rules stored per chain and condition under `data/knowledge/` |
| Judge | Verification component that filters ungrounded facts and blocks solution leaks before storage |

### 2.2 Functional requirements

1. The system shall load ARC-AGI-3 games cleanly via the `arc_agi` package, start from arbitrary levels when supported offline, and read per-level baseline action counts.
2. The system shall abstract LLM providers behind a common `Oracle` base class covering Google AI Studio (`GeminiOracle`), OpenAI-compatible endpoints (`OpenAICompatOracle` for NVIDIA and HPC vLLM Qwen servers), multi-provider fallback (`FallbackOracle`), and deterministic tests (`MockOracle`).
3. The grounder shall transform `frame` and `previous_frame` arrays into JSON objects listing dimensions, palette colors, connected objects (`id`, `color`, `area`, `bbox`, `center`), and transitions (`changed_pixels`, `moved`, `recolored`, `appeared`, `disappeared`).
4. The agent shall plan up to `plan_steps` actions per model call, clamp `ACTION6` coordinates inside frame bounds, replay valid queues step by step, and clear its queue to replan immediately when an action produces `changed_pixels == 0`.
5. After `T1` completes under `notes` or `ontology`, `KnowledgeWriter` shall extract JSON facts, `Judge` shall verify each fact against the compact trajectory (`check_fact`), filter out target leaks (`check_leak`), and persist only verified outputs (`Store`).
6. The `ontology` path shall validate generated Turtle syntax and basic OWL class/property consistency (`validate_turtle`) before saving `.ttl` artifacts.

### 2.3 Non-functional requirements

1. Model providers, endpoints, fallback chains, agent hyperparameters, file paths, and prompt templates must live outside Python source files in `configs/models.toml`, `configs/experiment.toml`, `.env`, and `prompts/*.txt`.
2. Every run must record `game_id`, `level`, `condition`, `seed`, `agent_config`, `exact_model`, `provider`, `model_settings`, step observations, actions, individual model calls, token counts (`prompt_tokens`, `completion_tokens`, `thoughts_tokens`, `total_tokens`), elapsed duration, and outcome (`won`, `game_over`, `budget_exhausted`).
3. Unit and module suites must run deterministically without external network calls using `MockOracle` and synthetic toy environments, plus optional offline `arc_agi` verification against local `ls20-9607627b`.
4. Continuous integration shall run tests across Linux and Windows in `.github/workflows/test.yml` and report coverage metrics.

### 2.4 Constraints and implementation choices

Google AI Studio's OpenAI-compatible endpoint fails with HTTP 500 on `gemma-4-26b-a4b-it` at `temperature=0.0`, and Gemma 4 counts reasoning tokens (`thoughtsTokenCount`) inside `maxOutputTokens` while returning HTTP 500 on large game prompts if `maxOutputTokens` is raised to `16384`. `GeminiOracle` therefore uses the native `:generateContent` REST endpoint, separates `thought` parts into `OracleResponse.reasoning`, and uses `max_tokens = 4096` for agent planning (`8192` for knowledge synthesis, `2048` for judging) in `configs/experiment.toml`, which also keeps total context comfortably within Qwen's `32768` vLLM window on the HPC cluster. `ARCBaseGame.next_level()` advances `levels_completed` without setting `GameState.WIN` until the final level of a game, so level completion checks `obs.state == "WIN" or obs.level > start_level or obs.levels_completed > start_completed`.

## 3. Architecture and design

### 3.1 Package layout

```text
configs/
  models.toml           Provider endpoints, env var names, retry settings, model profiles
  experiment.toml       Agent, grounding, knowledge, judge, path, and condition settings
  chains.toml           Curated within-game and cross-game chains and evaluation suites
prompts/
  agent_system.txt      JSON planning contract
  agent_step.txt        Per-step observation and transferred-knowledge template
  extract_facts.txt     Structured rule extraction from trajectory JSON
  notes_from_facts.txt  Free-text summary generation
  owl_from_facts.txt    OWL/Turtle ontology generation using arc:, owl:, rdf:, rdfs:
  judge_fact.txt        Fact-vs-trajectory verification prompt
  judge_leak.txt        Solution-leak classification prompt
bsp3/
  config.py             TOML loaders, path resolution, single-pass {var} prompt rendering
  oracle.py             Oracle hierarchy, key rotation, cooldowns, thinking parsing, factory
  games.py              arc_agi wrapper, GameObservation, Action translation, level jumps
  grounding.py          NumPy BFS object extraction and frame diff computation
  knowledge.py          JSON/Turtle sanitization, OWL consistency check, KnowledgeWriter
  judge.py              Fact verification and leak filter over Oracle.choose
  agent.py              Multi-step planning agent and trajectory execution loop
  store.py              Chain-scoped storage for trajectories, notes, and .ttl ontologies
  logger.py             Per-run steps.jsonl and summary.json writer
  runner.py             Experiment orchestration across single tasks and multi-task chains
  experiment.py         Batch grid runner, per-seed store isolation, resume logic, health checks
  summary.py            Run aggregation into per-condition and per-task CSV/Markdown tables
  cli.py                Single-task CLI entrypoint (python -m bsp3.cli)
scripts/
  survey.py             Surveys local games into data/survey/games.csv and PNG frames
  run_experiment.py     Batch suite runner and --check-models endpoint ping CLI
  summarize_runs.py     Generates runs/results_summary.csv and runs/results_summary.md
```

### 3.2 Component specifications

#### Config (`bsp3/config.py`)

`load_models_config(path) -> dict`, `load_experiment_config(path) -> dict`, and `render_prompt(name, prompts_dir, **values) -> str`. `render_prompt` performs single-pass placeholder matching with `_PLACEHOLDER_RE` so literal JSON braces inside prompt templates stay untouched while missing template variables raise `KeyError`.

#### Oracle (`bsp3/oracle.py`)

Abstract class `Oracle` stores `provider: str`, `model: str`, and `last_response: OracleResponse | None`, and exposes `query(prompt, *, system=None, temperature=0.0, max_tokens=1024) -> OracleResponse`, `generate(...) -> str`, and `choose(prompt, options, ...) -> str`. Concrete implementations are `MockOracle` (canned replies for offline tests), `GeminiOracle` (`x-goog-api-key`, native `:generateContent`), `OpenAICompatOracle` (`Bearer` token, `extra_body` passthrough such as `enable_thinking=false` for HPC Qwen, `<thought>`/`<think>` stripping), and `FallbackOracle` (sequential failover across configured providers).

Key management inside `_KeyedHttpOracle` rotates across multiple comma-separated keys (`GEMINI_API_KEYS`), retires keys permanently on HTTP 401/402/403, honors `Retry-After` or configured `cooldown_s` on HTTP 429, enforces `min_interval_s` pacing (such as `2.0s` on NVIDIA with `cooldown_s = 20.0`, `max_wait_s = 45.0`, `timeout_s = 300.0`, and `stream = true`), and bounds stall time with `max_wait_s`. Instantiation uses `oracle_from_config(profile, config_path, env_path, extra_body)` plus convenience wrappers `gemma_oracle_from_env` and `qwen_oracle_from_env`.

#### Games (`bsp3/games.py`)

`Games(offline=True, arcade=None)` maintains `_arcade`, `_env`, `_game_id`, and `_start_level`. `reset(game_id, level=0, seed=0) -> GameObservation` creates an environment and invokes `env._game.set_level(level)` when `level > 0`. `step(action, data=None) -> GameObservation` translates string or enum actions via `GameAction.from_id` / `GameAction.__members__`, enforces `(x, y)` coordinates on complex actions, and raises `GameError` on invalid calls. `baseline_for_level(level) -> int | None` returns `env.info.baseline_actions[level]` when available.

#### Grounding (`bsp3/grounding.py`)

`Grounding(connectivity=4, max_objects=64)` delegates to `describe(frame, previous_frame, connectivity, max_objects) -> dict`. Object extraction treats the modal pixel value as background color, labels 4-connected or 8-connected components via breadth-first search, sorts objects deterministically, and assigns stable per-color rank IDs (`c<color>_<rank>`). `_diff_objects` first matches identical bounding boxes and colors, then equal bounding boxes with altered colors (`recolored`), then remaining equal `(color, area)` objects by nearest centroid distance (`moved`), reporting leftover entries as `appeared` or `disappeared`.

#### KnowledgeWriter and Judge (`bsp3/knowledge.py`, `bsp3/judge.py`)

`KnowledgeWriter(oracle, prompts_dir, temperature, max_tokens)` converts compact trajectories into JSON rule dicts (`extract_facts`), free-text prose (`as_notes`), and OWL Turtle (`as_owl`). `validate_turtle(ttl)` verifies declared prefixes, masked string literals, matching `< >`, `[ ]`, and `( )` delimiters, complete trailing `.` statements, declared-prefix QNames, presence of RDF/RDFS/OWL vocabulary, absence of symbols typed as both class and property, and absence of `owl:disjointWith` self-loops. `Judge(oracle, prompts_dir, temperature, max_tokens)` verifies individual facts (`check_fact` returning `YES`/`NO`) and checks rendered knowledge for task-specific spoilers (`check_leak`).

#### Agent, Store, Logger, and Runner (`bsp3/agent.py`, `bsp3/store.py`, `bsp3/logger.py`, `bsp3/runner.py`)

`Agent` computes the action budget from `budget_multiplier * baseline` (falling back to `default_max_steps`), builds prompts from the current grounded state and the last `max_history_steps` steps, and falls back to `Oracle.choose` if JSON parsing yields an empty plan. `Logger` creates timestamped folders `runs/<stamp>_<slug>_L<level>_<condition>_s<seed>/` holding per-step lines in `steps.jsonl` and aggregated token/config metadata in `summary.json`. `Store` appends per-run knowledge under `data/knowledge/<chain>/<condition>.json`, caps `trajectory` text at `max_trajectory_steps` and `max_trajectory_chars` by dropping the oldest steps of the largest record first, so every earlier agent's record stays in the prompt as valid JSON, and exports combined ontology blocks to `.ttl` files. `Runner.from_config` builds the player and judge oracles from `model_profile` and `judge.model_profile`. The knowledge writer uses the oracle of the agent that played the step, so each step writes its own knowledge, and an explicit `knowledge_profile` pins one writer oracle for the whole chain. Step models can still differ through `--model a,b,c` or a per-task `model` key. `Runner` extracts `trajectory`, `notes`, and `ontology` from a single `T1` run (`save_knowledge` / `run_all_conditions`).

#### Batch runner and summary (`bsp3/experiment.py`, `bsp3/summary.py`, `configs/chains.toml`)

`configs/chains.toml` holds named within-game and cross-game task chains plus evaluation suites (`pilot`, `within`, `cross`, `all`), and its `[defaults]` run every step with the same `minimax` profile so a chain measures knowledge transfer instead of model changes. `bsp3/experiment.py` isolates `Store` namespaces per `(chain, model_profile, seed)` so seeds and models never leak knowledge across runs, supports `--resume` using `complete = true` markers in `summary.json`, and provides `check_model_profiles` for quick endpoint validation before long runs. `bsp3/summary.py` reads `runs/*/summary.json` and exports condition-level and task-level comparisons (wins, win rate, mean actions, model calls, token counts, elapsed time) to CSV and Markdown tables.
