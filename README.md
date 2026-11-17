# Transfer Learning Between Interactive Reasoning Agents in ARC-AGI-3

Bachelor Semester Project 3 (BSP3) at the University of Luxembourg, by Can Yıldız and Nina, supervised by Matteo Magnini.

## The idea

In ARC-AGI-3 an agent starts a game without knowing the rules or the goal. It has to try actions, watch what changes and work the game out as it plays. When the next game starts, everything it figured out is usually gone.

We want to find out whether that knowledge can be passed on. A first agent plays a game and writes down what it learned: which objects matter, how they move, what seems to finish a level. The next agent reads those notes before it plays a related game, then adds what it learns itself. Over 3 to 10 games this becomes a chain of agents, each starting from what the previous ones left behind.

Two questions drive the project. Do later agents need fewer actions and model calls? And does it matter how the knowledge is written down?

## How we test it

Each later agent plays the same game under four conditions:

| Condition | What the agent gets |
|---|---|
| Baseline | nothing |
| Full trajectory | every observation and action of the earlier agents |
| Notes | notes on the game's rules, in plain language |
| Ontology | the same knowledge as facts in a small OWL ontology |

The full trajectory is a control. Without it, an improvement could simply mean the model had a longer prompt. Notes and facts are limited to general rules, so an agent can't simply copy a solution. For every run we record success, actions, model calls, tokens and time.

We work on a small set of related games and leave the full benchmark aside.

## Neurosymbolic setup

A grounding step turns each game frame into symbols (objects, positions, what changed) and passes them on as JSON. A language model reads that state and picks the next action. Any other subsymbolic model could take its place, so the agent only talks to an oracle interface and never knows which model sits behind it.

What moves between agents is kept in symbolic form. Before a candidate fact enters the ontology, a judge checks it against what the agent actually observed. The judge also flags notes that give away a concrete solution.

## Models

We run the main experiments with MiniMax M3.1 Flash. Gemma 4 26B and Qwen 3.5 9B repeat a smaller set of runs, so we can see whether the results also hold for other models, including a small one.

MiniMax runs on NVIDIA's free endpoint, and Gemma runs on Google AI Studio with OpenRouter as a fallback. Qwen runs on the university's HPC cluster, and we reach it from our own laptops through an SSH tunnel. The judge is a language model as well.

## Timeline

| When | Phase |
|---|---|
| October | Design |
| November | Implementation, with unit, module and system tests |
| December | Experiments |
| January | Report and slides, due 31 January 2027 |

## Where we are

Updated 17 November 2026.

We surveyed the 25 offline public games (`data/survey/`) and picked ten candidate chains: five that move between levels of one game and five that move between games sharing a mechanic. They are described in `docs/candidate_chains.md` and defined in `configs/chains.toml`.

The implementation is in place. The `bsp3/` package has the agent, the oracle interface with a mock oracle for tests, grounding, knowledge extraction into notes and an OWL ontology, the judge, run logging and the experiment runner. The test suite runs on Ubuntu and Windows through GitHub Actions and uses the mock oracle, so it needs no API keys.

MiniMax and Gemma answer through their APIs. One live pilot has run: the `within_ls20` chain (level 0, then level 1 of `ls20`) under all four conditions, with MiniMax as agent and judge. Each task had a budget of one action, so the pilot shows that the pipeline runs from start to finish and says nothing about transfer yet.

Qwen is waiting for the HPC server. The full experiments start in December.

## Repository layout

| Path | Contents |
|---|---|
| `bsp3/` | agent, oracle, games, grounding, knowledge, judge, logging, runner, experiment, summary and monitor |
| `scripts/` | game survey, experiment runner, run summary and live monitor |
| `configs/` | model settings, experiment defaults and task chains |
| `prompts/` | prompt templates for the agent, knowledge extraction and the judge |
| `tests/` | unit, module and system tests |
| `docs/` | design document, design plan, candidate chains and HPC setup |
| `data/survey/` | survey table and the first frame of each public game |

Run logs (`runs/`) and knowledge stores (`data/knowledge/`) stay on our machines and are not in the repository.

## Setup

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Add your ARC-AGI-3 key (from [arcprize.org/platform](https://arcprize.org/platform)) and model API keys to `.env`. Running Qwen also needs an account on the university's HPC (see `docs/HPC.md`) and `QWEN_API_KEY` set in `.env` over an open SSH tunnel (`ssh -N -L 8000:<node>:8000 <user>@<login host>`).

## Usage

Run the test suite (uses mock answers and no API keys):

```bash
uv run python -m unittest discover -s tests -v
```

The same tests with a coverage report, as CI runs them:

```bash
uv run --with pytest --with pytest-cov pytest --cov=bsp3 --cov-report=term-missing tests
```

Check configured model endpoints before running a batch:

```bash
uv run python scripts/run_experiment.py --check-models gemma,minimax,qwen
```

Run the public game survey (`data/survey/games.csv` and `data/survey/frames/`):

```bash
uv run python scripts/survey.py
```

Run a single task with the agent:

```bash
uv run python -m bsp3.cli --game ls20-9607627b --level 0 --model gemma --condition nothing
```

Preview or run an experiment suite across conditions and seeds. The suites (`pilot`, `within`, `cross`, `all`) are listed in `configs/chains.toml`, and `--resume` skips runs that already finished:

```bash
uv run python scripts/run_experiment.py --suite pilot --model qwen --judge minimax_low --dry-run
uv run python scripts/run_experiment.py --suite cross --model qwen --judge minimax_low --resume --continue-on-error
```

Collect logged runs into `runs/results_summary.csv` and `runs/results_summary.md` (`--complete-only` leaves out runs that did not finish):

```bash
uv run python scripts/summarize_runs.py
```

Watch the active run's board and API event log live in a browser:

```bash
uv run python scripts/monitor.py --open
```

Model settings live in `configs/models.toml`, experiment defaults live in `configs/experiment.toml`, task chains live in `configs/chains.toml`, and prompt templates live in `prompts/`.
