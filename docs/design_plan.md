# Design plan

What we do from now until the deadline on 31 January 2027, the design choices we propose, and who does what.

## Design choices we propose

These shape everything else, so we want the supervisor's opinion on them early.

1. Grounding is plain code. Each frame is a grid of colors. The grounding step finds regions of the same color (the objects), compares them with the previous frame and lists what moved, appeared, disappeared or changed color. The result is JSON. No model is involved, so this part is cheap, exact and easy to test. It is the symbolic side of the system.
2. The model plans a few moves at a time. One model call per action would be slow: a single game has hundreds of actions, a thinking model needs several seconds per answer, and NVIDIA allows 40 requests a minute. So the model returns a short plan of up to about five actions, and we ask it again when the plan is done, when something unexpected happens or when a level ends.
3. Every level has an action budget, for example three times the human baseline. If the agent goes over it, the level counts as failed. This keeps runs comparable and their length bounded.
4. We use the first few levels of each game. Later levels are long, and we need many runs.
5. The same knowledge comes in two forms. After a run, a model extracts candidate facts in a structured form. The judge checks each fact against what the agent actually saw. The verified facts are then written out twice: as OWL facts for the ontology condition and as plain sentences for the notes condition. Both conditions then carry the same content and differ only in form, which is what the research question asks about. The full trajectory condition gets the grounded history itself.
6. Every run is logged step by step (state, plan, action, tokens, time) together with the game, level, condition, model and seed, so any result can be traced back and rerun.

## Questions for the supervisor

- Related tasks: levels of one game, different games, or both?
- Is choice 5 (same knowledge, two forms) acceptable?
- How large should the action budget be?
- Which model should act as the judge? It runs on MiniMax by default, the same model as the main player, and can be set to another model per run.

## The game survey

A script opens every public game (25 at the moment) without using any model and records:

- the number of levels and the actions available
- frame size, colors used and the human baseline per level
- which actions change the frame
- whether a game can start at a chosen level, so a later agent can begin directly on the next level, and whether replaying the same actions gives the same frames
- whether a game runs without internet once downloaded, which matters for the tests on GitHub

It writes `data/survey/games.csv` and saves the first frame of each game as a picture in `data/survey/frames/`.

## Components

| Component | Main functions |
|---|---|
| Games | `reset(game_id, level) → Frame`, `step(action) → Frame` |
| Grounding | `describe(frame, previous_frame) → State` |
| Oracle | `generate(prompt) → text`, `choose(prompt, options) → option` |
| Agent | `play(game, level, knowledge, budget) → Trajectory` |
| Knowledge writer | `extract_facts(trajectory) → Facts`, `as_notes(facts) → Notes`, `as_owl(facts) → Ontology` |
| Judge | `check_fact(fact, trajectory) → bool`, `check_leak(text) → bool` |
| Store | `add(chain, knowledge)`, `load(chain) → Knowledge` |
| Logger | `record_step(step)`, `record_run(summary)` |
| Runner | `run_task(chain, game, level, condition, seed)`, `run_all_conditions(chain, tasks)` |

The Oracle has providers behind one interface: Google AI Studio (Gemma 4 26B with key rotation), NVIDIA (MiniMax M3.1 Flash, kept under 40 requests a minute), the HPC server (Qwen 3.5 9B) and a mock with fixed answers for tests.

## Experiment

| | |
|---|---|
| Conditions | nothing, full trajectory, notes, ontology facts |
| Main model | MiniMax M3.1 Flash, every chain and condition |
| Other models | Gemma 4 26B and Qwen 3.5 9B, a smaller set of runs |
| Metrics | levels solved, actions, model calls, tokens, time |
| Repeats | set after the pilot, once we know how long a run takes |

## Tests

Unit tests for every function, module tests for each component, and one system test that runs a whole chain on one game with the mock oracle. GitHub Actions runs them on every push and reports coverage. The tests use no API keys.

## Schedule

| When | Milestone |
|---|---|
| Wed 18 Nov | survey done, games played by hand, chains picked |
| Fri 20 Nov | design document sent to the supervisor |
| Fri 27 Nov | first working version: MiniMax plays level 1 of one game through grounding and the oracle; pilot measurements |
| Fri 11 Dec | knowledge writer, judge, runner, logger, store and tests done; HPC server connected |
| 14 to 20 Dec | experiments; long runs can continue over the winter break |
| Fri 8 Jan | results table and plots |
| Fri 29 Jan | report and slides submitted, two days before the deadline |

## Who does what

Can takes the parts that everything else depends on. Nina's parts can run in parallel, so neither of us waits for the other. Sizes are rough: S is about a day, M a few days, L about a week.

| Can | Size | Nina | Size |
|---|---|---|---|
| Survey script | M | Play the candidate games by hand and describe each one's rules and goal | M |
| Grounding | L | Qwen 3.5 9B server on the HPC and a short guide for the tunnel | M |
| Oracle with NVIDIA provider, rate limit and mock | M | HPC provider for the oracle | S |
| Agent loop and prompts | L | Logger and store | M |
| Pilot and budget decision | S | Ontology draft in Protégé | M |
| Fact extraction and judge | L | Unit, module and system tests, CI with coverage | L |
| Runner for the four conditions | M | Runs with Gemma and Qwen | S |
| Main experiments with MiniMax | M | Results table and plots | M |
| Design document: architecture and design choices | M | Design document: system description and functional requirements | M |
| Report: method, results, discussion | L | One page on related work | M |
| | | Report: introduction, related work, requirements, setup; first draft of the slides | L |

Together: picking the chains, preparing the meetings, a final read of the report and slides.

Nina can start the related work and the game descriptions right away, since they don't depend on any code.
