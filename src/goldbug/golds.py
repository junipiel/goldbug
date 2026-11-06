"""A small, typed set of gold answers in MATH-style LaTeX.

The census does not need problems: every strategy we test is either
answer-independent (it never looks at the gold) or a controlled mutation of the
gold. What matters is coverage of answer *types*, because checker bugs are
type-specific (integers vs. fractions vs. intervals vs. tuples ...).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gold:
    latex: str
    kind: str  # int | neg_int | big_int | frac | decimal | radical | pi | interval | tuple | set | expr | text
    value: float | None = None  # numeric value when the answer is a real number

    @property
    def is_int(self) -> bool:
        return self.kind in {"int", "neg_int", "big_int"}


_INTS = [0, 1, 2, 3, 5, 7, 10, 12, 17, 20, 24, 36, 42, 50, 64, 99, 100, 128, 210, 365, 512, 720, 999]
_NEG = [-1, -3, -8, -12, -45]
_BIG = [1000, 2023, 4096, 12345, 65536, 123456, 1000000]
_FRACS = [(1, 2), (1, 3), (2, 3), (3, 4), (5, 8), (7, 12), (11, 3), (-1, 4), (9, 16), (25, 6)]
_DECIMALS = ["0.5", "2.5", "3.75", "0.125", "12.6"]
_RADICALS = [("\\sqrt{2}", 2 ** 0.5), ("2\\sqrt{3}", 2 * 3 ** 0.5), ("\\frac{\\sqrt{5}}{2}", 5 ** 0.5 / 2),
             ("3+\\sqrt{7}", 3 + 7 ** 0.5), ("\\sqrt[3]{4}", 4 ** (1 / 3))]
_PI = [("2\\pi", 2 * 3.141592653589793), ("\\frac{\\pi}{4}", 3.141592653589793 / 4), ("6\\pi", 6 * 3.141592653589793)]
_INTERVALS = ["(-\\infty, 3)", "[2, 5]", "(0, 1]", "[-1, \\infty)", "(-2, 2)"]
_TUPLES = ["(1, 2)", "(3, -1)", "(0, 5)", "(2, 3, 6)"]
_SETS = ["1, 2", "-3, 3", "2, 5, 7"]
_EXPRS = ["x^2 + 1", "2x - 3", "(x+1)^2", "\\frac{x}{2}", "3a + 2b"]
_TEXT = ["\\text{Monday}", "\\text{(C)}", "\\text{odd}"]


def default_golds() -> list[Gold]:
    golds: list[Gold] = []
    golds += [Gold(str(n), "int", float(n)) for n in _INTS]
    golds += [Gold(str(n), "neg_int", float(n)) for n in _NEG]
    golds += [Gold(str(n), "big_int", float(n)) for n in _BIG]
    for a, b in _FRACS:
        s = f"-\\frac{{{-a}}}{{{b}}}" if a < 0 else f"\\frac{{{a}}}{{{b}}}"
        golds.append(Gold(s, "frac", a / b))
    golds += [Gold(d, "decimal", float(d)) for d in _DECIMALS]
    golds += [Gold(s, "radical", v) for s, v in _RADICALS]
    golds += [Gold(s, "pi", v) for s, v in _PI]
    golds += [Gold(s, "interval") for s in _INTERVALS]
    golds += [Gold(s, "tuple") for s in _TUPLES]
    golds += [Gold(s, "set") for s in _SETS]
    golds += [Gold(s, "expr") for s in _EXPRS]
    golds += [Gold(s, "text") for s in _TEXT]
    return golds
