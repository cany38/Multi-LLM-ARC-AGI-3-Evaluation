import json
import unittest

from bsp3.judge import Judge
from bsp3.knowledge import KnowledgeWriter, validate_turtle
from bsp3.oracle import MockOracle


VALID_TTL = """@prefix arc: <http://bsp3.local/arc#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

arc:Player a owl:Class .
arc:MovesWith a owl:ObjectProperty ."""


class KnowledgeAndJudgeTests(unittest.TestCase):
    def test_extract_notes_and_owl(self) -> None:
        facts = [{"subject": "Player", "relation": "moves_up_with", "object": "ACTION1"}]
        oracle = MockOracle(
            [
                f"```json\n{json.dumps(facts)}\n```",
                "ACTION1 moves the player block upward.",
                f"```turtle\n{VALID_TTL}\n```",
            ]
        )
        writer = KnowledgeWriter.from_config(oracle)
        trajectory = [{"action": "ACTION1", "state": {"diff": {"moved": [{"id": "c9_1"}]}}}]

        extracted = writer.extract_facts(trajectory, game_id="ls20", level=0, outcome="won")
        self.assertEqual(extracted, facts)
        self.assertIn("ls20", oracle.calls[0]["prompt"])

        notes = writer.as_notes(extracted)
        self.assertEqual(notes, "ACTION1 moves the player block upward.")

        ttl = writer.as_owl(extracted)
        self.assertTrue(validate_turtle(ttl))
        self.assertIn("arc:Player", ttl)

        self.assertEqual(writer.as_notes([]), "")
        self.assertEqual(writer.as_owl([]), "")

    def test_invalid_turtle_rejected(self) -> None:
        self.assertFalse(validate_turtle("not turtle"))
        self.assertFalse(validate_turtle("@prefix arc: <http://a#> .\narc:A a arc:B"))
        writer = KnowledgeWriter.from_config(MockOracle(["missing prefixes."]))
        with self.assertRaises(ValueError):
            writer.as_owl([{"subject": "A", "relation": "r", "object": "B"}])

    def test_owl_consistency_checks(self) -> None:
        rich_ttl = """@prefix arc: <http://bsp3.local/arc#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

arc:Action a owl:Class .
arc:Rule a owl:Class ;
    rdfs:label "Rule with \\"quotes\\" <kept>" ;
    rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty arc:usesAction ;
        owl:someValuesFrom arc:Action
    ] .
arc:usesAction a owl:ObjectProperty ."""
        self.assertTrue(validate_turtle(rich_ttl))

        bad_cases = [
            f"{VALID_TTL}\narc:Truncated a owl:Class",
            f"{VALID_TTL}\nunknown:Thing a owl:Class .",
            f"{VALID_TTL}\narc:Broken rdfs:subClassOf [ a owl:Restriction .",
            "@prefix arc: <http://a#> .\narc:A arc:B arc:C .",
            f"{VALID_TTL}\narc:Player a owl:ObjectProperty .",
            f"{VALID_TTL}\narc:Dual a owl:Class, owl:ObjectProperty .",
            f"{VALID_TTL}\narc:Player owl:disjointWith arc:Player ; rdfs:label \"x\" .",
            "@prefix arc: <http://a#> .\n@prefix owl: <http://www.w3.org/2002/07/owl#> .\narc:A rdfs:label \"unclosed .",
        ]
        for bad in bad_cases:
            with self.subTest(bad=bad):
                self.assertFalse(validate_turtle(bad))

    def test_judge_checks_facts_and_leaks(self) -> None:
        oracle = MockOracle(["YES", "NO", "YES", "NO"])
        judge = Judge.from_config(oracle)
        traj = [{"action": "ACTION1", "moved": True}]
        f1 = {"subject": "Player", "relation": "moves", "object": "ACTION1"}
        f2 = {"subject": "Door", "relation": "teleports", "object": "ACTION5"}

        kept = judge.filter_facts([f1, f2], traj)
        self.assertEqual(kept, [f1])

        self.assertTrue(judge.check_leak("Press ACTION1 3 times then click x=10 y=20."))
        self.assertFalse(judge.check_leak("Collect matching keys before touching locks."))
        self.assertFalse(judge.check_leak(""))


if __name__ == "__main__":
    unittest.main()
