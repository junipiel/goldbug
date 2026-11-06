"""A deliberately lenient grader for the pre-flight demo: it accepts a response
if ANY \\boxed{...} answer matches the gold. Several hand-written reward
functions in the wild behave like this."""
import re


def any_boxed(response: str, gold: str) -> float:
    return float(gold.strip() in [m.strip() for m in re.findall(r"\\boxed\{([^{}]*)\}", response)])
