"""
Shared deterministic computation cores for prebuilt tools.

These functions implement the LLM-free logic behind deterministic operations
(see ``framework.operations``). MCP servers wrap them in KG-flavored tools,
keeping their own response envelopes and journal handling; the math itself
lives here exactly once.

Extracted verbatim from the previously duplicated ``VerifyNumericCondition``
implementations in kqapro_server.py and sciqa_server.py; output strings are
byte-identical to the originals.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


def parse_numeric(val: str) -> float:
    """Parse numeric value, handling common formats like '150 million', '1.5k', dates."""
    val = val.strip().lower()

    # Handle dates (convert to timestamp for comparison)
    if "-" in val and len(val) >= 10:
        try:
            dt = datetime.fromisoformat(val.split("T")[0])
            return dt.timestamp()
        except Exception:
            pass

    # Handle multipliers
    multipliers = {
        "trillion": 1e12, "billion": 1e9, "million": 1e6,
        "thousand": 1e3, "hundred": 1e2,
        "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12
    }

    # Extract number and multiplier
    match = re.match(r"([+-]?[\d.,]+)\s*([a-z]+)?", val)
    if match:
        num_str = match.group(1).replace(",", "")
        mult_str = match.group(2) or ""
        num = float(num_str)
        mult = multipliers.get(mult_str, 1.0)
        return num * mult

    # Fallback: try direct conversion
    return float(val.replace(",", ""))


@dataclass(frozen=True)
class NumericComparison:
    """Outcome of a deterministic numeric/date comparison."""

    verdict: str  # "TRUE" | "FALSE" | "ERROR"
    explanation: str
    num1: Optional[float] = None
    num2: Optional[float] = None
    error: Optional[str] = None  # str(exception) when verdict == "ERROR"

    @property
    def is_error(self) -> bool:
        return self.verdict == "ERROR"


def compare_numeric(value1: str, operator: str, value2: str, unit: str = "") -> NumericComparison:
    """
    Deterministically compare two values (numbers, dates, values with multipliers).

    Never raises: parse or operator failures are reported as verdict="ERROR".
    """
    try:
        num1 = parse_numeric(value1)
        num2 = parse_numeric(value2)

        comparisons = {
            "<": num1 < num2,
            ">": num1 > num2,
            "<=": num1 <= num2,
            ">=": num1 >= num2,
            "==": abs(num1 - num2) < 1e-9,  # Float equality tolerance
            "!=": abs(num1 - num2) >= 1e-9
        }

        result = comparisons[operator]
        verdict = "TRUE" if result else "FALSE"

        unit_str = f" {unit}" if unit else ""
        explanation = f"{value1}{unit_str} {operator} {value2}{unit_str} → {num1} {operator} {num2} = {verdict}"

        return NumericComparison(verdict=verdict, explanation=explanation, num1=num1, num2=num2)

    except Exception as e:
        return NumericComparison(
            verdict="ERROR",
            explanation=f"Could not compare values: {str(e)}",
            error=str(e),
        )
