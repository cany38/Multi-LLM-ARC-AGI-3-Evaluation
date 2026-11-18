from collections.abc import Sequence
import json
from pathlib import Path
import re
from typing import Any

from bsp3.config import load_experiment_config, render_prompt
from bsp3.oracle import Oracle

_PREFIX_RE = re.compile(
    r"^(?:@prefix\s+([A-Za-z_][\w-]*):\s*<[^>\s]+>\s*\.$|PREFIX\s+([A-Za-z_][\w-]*):\s*<[^>\s]+>$)",
    re.IGNORECASE,
)
_QNAME_RE = re.compile(r"(?<![<\w:/])([A-Za-z_][\w-]*):[A-Za-z_][\w.-]*")
_CLASS_TYPES = frozenset({"owl:Class", "rdfs:Class"})
_PROP_TYPES = frozenset({"owl:ObjectProperty", "owl:DatatypeProperty", "rdf:Property"})


def strip_code_fences(text: str) -> str:
    raw = text.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_payload(raw: str) -> Any:
    cleaned = strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        for open_ch, close_ch in (("[", "]"), ("{", "}")):
            start = cleaned.find(open_ch)
            end = cleaned.rfind(close_ch)
            if start != -1 and end > start:
                return json.loads(cleaned[start : end + 1])
        raise


def _strip_literals(text: str) -> str | None:
    out: list[str] = []
    in_str = False
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if in_str and ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_str = not in_str
            out.append('"')
            continue
        if not in_str:
            out.append(ch)
    if in_str:
        return None
    return "".join(out)


def validate_turtle(ttl: str) -> bool:
    text = strip_code_fences(ttl)
    if not text or "```" in text:
        return False

    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    declared: set[str] = set()
    body_lines: list[str] = []
    for ln in lines:
        if ln.startswith("@prefix") or ln.upper().startswith("PREFIX "):
            match = _PREFIX_RE.match(ln)
            if not match:
                return False
            declared.add(match.group(1) or match.group(2))
        else:
            body_lines.append(ln)

    if not declared or not body_lines:
        return False

    body = "\n".join(body_lines)
    masked = _strip_literals(body)
    if masked is None or not masked.rstrip().endswith("."):
        return False
    for open_ch, close_ch in (("<", ">"), ("[", "]"), ("(", ")")):
        if masked.count(open_ch) != masked.count(close_ch):
            return False

    used_prefixes = _QNAME_RE.findall(re.sub(r"<[^>]*>", " ", masked))
    if any(p != "_" and p not in declared for p in used_prefixes):
        return False
    if not re.search(r"\b(?:a|owl:\w+|rdfs:\w+|rdf:\w+)\b", masked):
        return False

    classes: set[str] = set()
    props: set[str] = set()
    for stmt in masked.split("."):
        tokens = stmt.strip().split()
        if len(tokens) < 3:
            continue
        subj = tokens[0]
        if tokens[1] in ("a", "rdf:type"):
            objs = {t.rstrip(",;") for t in tokens[2:]}
            if objs & _CLASS_TYPES:
                classes.add(subj)
            if objs & _PROP_TYPES:
                props.add(subj)
        if "owl:disjointWith" in tokens[1:]:
            idx = tokens.index("owl:disjointWith")
            if idx + 1 < len(tokens) and tokens[idx + 1].rstrip(",;") == subj:
                return False

    return not (classes & props)


class KnowledgeWriter:
    """Extracts facts from a run and writes them as plain notes and OWL Turtle."""

    def __init__(
        self,
        oracle: Oracle,
        *,
        prompts_dir: Path | str = "prompts",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self._oracle = oracle
        self._prompts_dir = prompts_dir
        self._temperature = temperature
        self._max_tokens = max_tokens

    @classmethod
    def from_config(
        cls,
        oracle: Oracle,
        config_path: Path | str = "configs/experiment.toml",
    ) -> "KnowledgeWriter":
        exp = load_experiment_config(config_path)
        kcfg = exp.get("knowledge", {})
        prompts_dir = exp.get("paths", {}).get("prompts_dir", "prompts")
        return cls(
            oracle,
            prompts_dir=prompts_dir,
            temperature=float(kcfg.get("temperature", 0.0)),
            max_tokens=int(kcfg.get("max_tokens", 1024)),
        )

    def extract_facts(
        self,
        trajectory: Sequence[dict[str, Any]],
        *,
        game_id: str = "",
        level: int = 0,
        outcome: str = "",
    ) -> list[dict[str, Any]]:
        prompt = render_prompt(
            "extract_facts",
            prompts_dir=self._prompts_dir,
            game_id=game_id or "unknown",
            level=level,
            outcome=outcome or "unknown",
            steps_count=len(trajectory),
            trajectory_json=json.dumps(list(trajectory), ensure_ascii=False),
        )
        reply = self._oracle.generate(
            prompt, temperature=self._temperature, max_tokens=self._max_tokens
        )
        parsed = parse_json_payload(reply)
        if not isinstance(parsed, list) or not all(isinstance(x, dict) for x in parsed):
            raise ValueError(f"Expected JSON array of fact objects, got {type(parsed).__name__}")
        return parsed

    def as_notes(self, facts: Sequence[dict[str, Any]]) -> str:
        if not facts:
            return ""
        prompt = render_prompt(
            "notes_from_facts",
            prompts_dir=self._prompts_dir,
            facts_json=json.dumps(list(facts), ensure_ascii=False),
        )
        return strip_code_fences(
            self._oracle.generate(
                prompt, temperature=self._temperature, max_tokens=self._max_tokens
            )
        )

    def as_owl(self, facts: Sequence[dict[str, Any]]) -> str:
        if not facts:
            return ""
        prompt = render_prompt(
            "owl_from_facts",
            prompts_dir=self._prompts_dir,
            facts_json=json.dumps(list(facts), ensure_ascii=False),
        )
        ttl = strip_code_fences(
            self._oracle.generate(
                prompt, temperature=self._temperature, max_tokens=self._max_tokens
            )
        )
        if not validate_turtle(ttl):
            raise ValueError(f"Invalid Turtle ontology output:\n{ttl}")
        return ttl
