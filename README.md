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
| Notes | free-text notes on the game's rules |
| Ontology | the same knowledge as facts in a small OWL ontology |

The full trajectory is a control. Without it, an improvement could simply mean the model had a longer prompt. Notes and facts are limited to general rules, so an agent can't copy a ready-made solution. For every run we record success, actions, model calls, tokens and time.

We work on a small set of related games and leave the full benchmark aside.

## Neuro-symbolic setup

A grounding step turns each game frame into symbols (objects, positions, what changed) and passes them on as JSON. A language model reads that state and picks the next action. Any other sub-symbolic model could take its place, so the agent only talks to an oracle interface and never knows which model sits behind it.

What moves between agents is kept in symbolic form. Before a candidate fact enters the ontology, a judge checks it against what the agent actually observed. The judge also flags notes that give away a concrete solution.

## Models

The language models run on NVIDIA's free endpoints:

- MiniMax M3.1 Flash, our main model
- Gemma 4 31B
- Step 5 Preview, to be replaced by Muse Glimmer 30B if the preview is withdrawn

The judge is Jev from TypeSafe AI. It answers with typed values, such as a choice from a list or the probability that a statement is true, so its output can go straight into the checks.

## Timeline

| When | Phase |
|---|---|
| October | Design |
| November | Implementation, with unit, module and system tests |
| December | Experiments |
| January | Report and slides, due 31 January 2027 |

We are in the design phase, and the code comes next.
