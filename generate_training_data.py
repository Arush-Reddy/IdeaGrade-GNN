"""Generate a reproducible, balanced starter corpus for GNN development.

These labels implement a consistent content rubric. They bootstrap development,
but real anonymized teacher labels should replace them for production evaluation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from settings import DATA_DIR


OUTPUT_PATH = DATA_DIR / "synthetic_training.jsonl"


@dataclass(frozen=True)
class Claim:
    subject: str
    relation: str
    object: str
    synonym: str
    opposite: str


TOPICS: dict[str, list[Claim]] = {
    "photosynthesis": [
        Claim("plants", "absorb", "sunlight", "capture", "block"),
        Claim("plants", "use", "carbon dioxide", "consume", "reject"),
        Claim("plants", "produce", "glucose", "create", "destroy"),
        Claim("plants", "release", "oxygen", "emit", "absorb"),
    ],
    "water_cycle": [
        Claim("sunlight", "heat", "water", "warm", "freeze"),
        Claim("water", "form", "vapor", "create", "destroy"),
        Claim("vapor", "form", "clouds", "create", "remove"),
        Claim("clouds", "release", "rain", "produce", "absorb"),
    ],
    "electric_circuit": [
        Claim("battery", "supply", "energy", "provide", "remove"),
        Claim("wires", "carry", "current", "conduct", "block"),
        Claim("switch", "control", "current", "regulate", "create"),
        Claim("bulb", "produce", "light", "generate", "absorb"),
    ],
    "democracy": [
        Claim("citizens", "elect", "representatives", "choose", "remove"),
        Claim("representatives", "create", "laws", "make", "ignore"),
        Claim("courts", "interpret", "laws", "explain", "write"),
        Claim("constitution", "limit", "government", "restrict", "expand"),
    ],
    "computer_system": [
        Claim("cpu", "execute", "instructions", "run", "store"),
        Claim("memory", "store", "active data", "hold", "delete"),
        Claim("storage", "preserve", "files", "keep", "execute"),
        Claim("operating system", "manage", "hardware", "control", "damage"),
    ],
    "gravity": [
        Claim("gravity", "attract", "objects", "pull", "repel"),
        Claim("mass", "increase", "gravity", "strengthen", "remove"),
        Claim("gravity", "keep", "planets", "hold", "release"),
        Claim("planets", "orbit", "stars", "circle", "escape"),
    ],
    "cell_respiration": [
        Claim("cells", "use", "glucose", "consume", "produce"),
        Claim("cells", "use", "oxygen", "consume", "release"),
        Claim("cells", "produce", "energy", "generate", "destroy"),
        Claim("cells", "release", "carbon dioxide", "emit", "absorb"),
    ],
    "food_chain": [
        Claim("plants", "capture", "energy", "store", "destroy"),
        Claim("herbivores", "eat", "plants", "consume", "produce"),
        Claim("carnivores", "eat", "herbivores", "consume", "protect"),
        Claim("decomposers", "recycle", "nutrients", "return", "remove"),
    ],
    "digestion": [
        Claim("teeth", "break", "food", "crush", "create"),
        Claim("stomach", "digest", "food", "process", "produce"),
        Claim("intestine", "absorb", "nutrients", "take", "release"),
        Claim("blood", "carry", "nutrients", "transport", "destroy"),
    ],
    "immune_system": [
        Claim("pathogens", "cause", "disease", "produce", "prevent"),
        Claim("white cells", "detect", "pathogens", "identify", "hide"),
        Claim("antibodies", "bind", "pathogens", "attach", "release"),
        Claim("immune system", "destroy", "pathogens", "remove", "protect"),
    ],
    "climate_change": [
        Claim("greenhouse gases", "trap", "heat", "retain", "release"),
        Claim("fossil fuels", "release", "carbon dioxide", "emit", "absorb"),
        Claim("carbon dioxide", "increase", "temperature", "raise", "reduce"),
        Claim("warming", "melt", "ice", "thaw", "create"),
    ],
    "plate_tectonics": [
        Claim("mantle", "move", "plates", "shift", "stop"),
        Claim("plates", "create", "earthquakes", "cause", "prevent"),
        Claim("plates", "form", "mountains", "build", "destroy"),
        Claim("magma", "create", "volcanoes", "form", "remove"),
    ],
    "internet": [
        Claim("browser", "request", "webpage", "ask", "store"),
        Claim("dns", "translate", "domain name", "convert", "delete"),
        Claim("server", "send", "data", "transmit", "hide"),
        Claim("router", "forward", "packets", "direct", "destroy"),
    ],
    "database": [
        Claim("table", "store", "records", "hold", "execute"),
        Claim("primary key", "identify", "record", "distinguish", "duplicate"),
        Claim("query", "retrieve", "data", "fetch", "erase"),
        Claim("index", "speed", "search", "accelerate", "block"),
    ],
    "supply_demand": [
        Claim("demand", "increase", "price", "raise", "reduce"),
        Claim("supply", "increase", "price", "lower", "raise"),
        Claim("price", "balance", "demand", "regulate", "destroy"),
        Claim("market", "allocate", "goods", "distribute", "remove"),
    ],
    "newton_laws": [
        Claim("force", "change", "motion", "alter", "preserve"),
        Claim("mass", "resist", "acceleration", "oppose", "increase"),
        Claim("action", "create", "reaction", "produce", "remove"),
        Claim("friction", "oppose", "motion", "resist", "accelerate"),
    ],
    "dna": [
        Claim("dna", "store", "information", "hold", "erase"),
        Claim("genes", "encode", "proteins", "specify", "destroy"),
        Claim("cells", "copy", "dna", "replicate", "remove"),
        Claim("mutations", "change", "genes", "alter", "preserve"),
    ],
    "ecosystem": [
        Claim("producers", "capture", "energy", "store", "destroy"),
        Claim("consumers", "obtain", "energy", "gain", "create"),
        Claim("decomposers", "return", "nutrients", "recycle", "remove"),
        Claim("organisms", "depend", "environment", "rely", "ignore"),
    ],
}


DISTRACTOR_OBJECTS = [
    "plastic", "satellites", "currency", "sand", "music", "engines",
    "mountains", "paint", "telephones", "salt",
]

PLURAL_SUBJECTS = {
    "plants", "clouds", "wires", "citizens", "representatives", "courts",
    "cells", "herbivores", "carnivores", "decomposers", "teeth", "pathogens",
    "white cells", "antibodies", "greenhouse gases", "fossil fuels", "plates",
    "genes", "mutations", "producers", "consumers", "organisms", "objects",
    "planets",
}


def conjugate(subject: str, relation: str) -> str:
    """Use simple-present agreement while preserving the relation lemma."""
    singular = subject not in PLURAL_SUBJECTS
    if relation.startswith("do not "):
        base = relation.removeprefix("do not ")
        return f"{'does' if singular else 'do'} not {base}"
    if not singular:
        return relation
    if relation.endswith("y") and len(relation) > 1 and relation[-2] not in "aeiou":
        return relation[:-1] + "ies"
    if relation.endswith(("s", "sh", "ch", "x", "z", "o")):
        return relation + "es"
    return relation + "s"


def simple_sentence(subject: str, relation: str, object_: str) -> str:
    display_subject = subject[0].upper() + subject[1:]
    return (
        f"The {display_subject.lower()} {conjugate(subject, relation)} "
        f"the {object_}."
    )


def reversed_sentence(claim: Claim) -> str:
    subject = claim.object
    display_subject = subject[0].upper() + subject[1:]
    return (
        f"The {display_subject.lower()} incorrectly "
        f"{conjugate(subject, claim.relation)} the {claim.subject}."
    )


def sentence(claim: Claim, relation: str | None = None, object_: str | None = None) -> str:
    return simple_sentence(
        claim.subject,
        relation or claim.relation,
        object_ or claim.object,
    )


def answer(claims: list[Claim], mode: str = "exact") -> str:
    if mode == "synonym":
        return " ".join(sentence(claim, claim.synonym) for claim in claims)
    return " ".join(sentence(claim) for claim in claims)


def build_records() -> list[dict]:
    rng = random.Random(20260724)
    records: list[dict] = []
    for topic, claims in TOPICS.items():
        reference = answer(claims)
        wrong_object = rng.choice(DISTRACTOR_OBJECTS)
        variants = [
            (answer(claims), 100.0, "exact"),
            (answer(claims, "synonym"), 95.0, "paraphrase"),
            (" ".join(sentence(claim) for claim in reversed(claims)), 100.0, "reordered"),
            (answer(claims[:3]), 75.0, "one_missing"),
            (answer(claims[:2]), 50.0, "two_missing"),
            (answer(claims[:1]), 25.0, "one_correct"),
            (
                " ".join([sentence(claims[0], claims[0].synonym), *[sentence(c) for c in claims[1:3]]]),
                73.0,
                "paraphrase_and_missing",
            ),
            (
                " ".join([sentence(claims[0], claims[0].opposite), *[sentence(c) for c in claims[1:]]]),
                63.0,
                "one_wrong_relation",
            ),
            (
                " ".join([sentence(claims[0], object_=wrong_object), *[sentence(c) for c in claims[1:]]]),
                60.0,
                "one_wrong_object",
            ),
            (
                " ".join([sentence(claims[0]), sentence(claims[1], claims[1].opposite)]),
                30.0,
                "mixed_partial",
            ),
            (
                " ".join(sentence(c, f"do not {c.relation}") for c in claims),
                5.0,
                "negated",
            ),
            (
                " ".join(
                    reversed_sentence(c)
                    for c in claims
                ),
                8.0,
                "reversed",
            ),
            (
                " ".join(sentence(c, c.opposite, rng.choice(DISTRACTOR_OBJECTS)) for c in claims),
                3.0,
                "all_wrong",
            ),
            ("The satellites transmit the music. The engines produce the sand.", 0.0, "unrelated"),
            (
                answer(claims) + f" {sentence(claims[0], claims[0].opposite, wrong_object)}",
                88.0,
                "correct_with_false_extra",
            ),
            (
                answer(claims[:3], "synonym") + f" {sentence(claims[3], claims[3].opposite)}",
                68.0,
                "mostly_correct_one_wrong",
            ),
        ]
        for student, score, variant in variants:
            records.append(
                {
                    "reference": reference,
                    "student": student,
                    "teacher_score": score,
                    "source": "synthetic_rubric_v2",
                    "topic": topic,
                    "variant": variant,
                }
            )

        # Variable-length reference answers are essential: a graph model trained
        # only on four-claim essays generalizes poorly to short responses.
        for size in (1, 2, 3):
            short_claims = claims[:size]
            short_reference = answer(short_claims)
            compact_variants = [
                (answer(short_claims), 100.0, "short_exact"),
                (answer(short_claims, "synonym"), 95.0, "short_paraphrase"),
                (
                    " ".join(sentence(c, f"do not {c.relation}") for c in short_claims),
                    5.0,
                    "short_negated",
                ),
                (
                    " ".join(sentence(c, c.opposite) for c in short_claims),
                    10.0,
                    "short_wrong_relation",
                ),
                (
                    " ".join(sentence(c, object_=wrong_object) for c in short_claims),
                    5.0,
                    "short_wrong_object",
                ),
                (
                    " ".join(
                        reversed_sentence(c)
                        for c in short_claims
                    ),
                    8.0,
                    "short_reversed",
                ),
                ("The satellites transmit the music.", 0.0, "short_unrelated"),
                (
                    answer(short_claims)
                    + f" {sentence(short_claims[0], short_claims[0].opposite, wrong_object)}",
                    85.0,
                    "short_correct_with_false_extra",
                ),
            ]
            if size > 1:
                compact_variants.extend(
                    [
                        (
                            answer(short_claims[:-1]),
                            round(100.0 * (size - 1) / size, 1),
                            "short_partial",
                        ),
                        (
                            answer(short_claims[:-1], "synonym"),
                            round(95.0 * (size - 1) / size, 1),
                            "short_partial_paraphrase",
                        ),
                    ]
                )
            for student, score, variant in compact_variants:
                records.append(
                    {
                        "reference": short_reference,
                        "student": student,
                        "teacher_score": score,
                        "source": "synthetic_rubric_v2",
                        "topic": topic,
                        "variant": f"{variant}_{size}_claims",
                    }
                )
    return records


def main() -> None:
    records = build_records()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} examples across {len(TOPICS)} topics to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
