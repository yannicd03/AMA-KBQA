"""Tests for the shared deterministic comparison core."""

import pytest

from ama_kbqa.framework.deterministic import compare_numeric, parse_numeric


class TestParseNumeric:
    def test_plain_numbers(self):
        assert parse_numeric("150") == 150.0
        assert parse_numeric("3.14") == 3.14
        assert parse_numeric("-42") == -42.0

    def test_comma_separated(self):
        assert parse_numeric("1,500,000") == 1_500_000.0

    def test_multipliers(self):
        assert parse_numeric("150 million") == 150e6
        assert parse_numeric("1.5k") == 1500.0
        assert parse_numeric("2 billion") == 2e9

    def test_dates_order(self):
        assert parse_numeric("2019-05-15") < parse_numeric("2020-01-01")

    def test_invalid_raises(self):
        with pytest.raises(Exception):
            parse_numeric("not a number")


class TestCompareNumeric:
    def test_true_verdict(self):
        result = compare_numeric("146", "<", "238.9", "minutes")
        assert result.verdict == "TRUE"
        assert not result.is_error
        assert result.num1 == 146.0
        assert result.num2 == 238.9

    def test_false_verdict(self):
        result = compare_numeric("300", "<", "238.9")
        assert result.verdict == "FALSE"

    def test_equality_tolerance(self):
        assert compare_numeric("1.0", "==", "1.0").verdict == "TRUE"
        assert compare_numeric("1.0", "!=", "1.0").verdict == "FALSE"

    def test_date_comparison(self):
        result = compare_numeric("2019-05-15", "<", "2020-01-01")
        assert result.verdict == "TRUE"

    def test_multiplier_comparison(self):
        result = compare_numeric("1500000", ">", "1 million")
        assert result.verdict == "TRUE"

    def test_explanation_format(self):
        result = compare_numeric("146", "<", "238.9", "minutes")
        # Format must stay byte-identical to the original server implementations
        assert result.explanation == "146 minutes < 238.9 minutes → 146.0 < 238.9 = TRUE"

    def test_error_never_raises(self):
        result = compare_numeric("garbage", "<", "100")
        assert result.verdict == "ERROR"
        assert result.is_error
        assert result.error is not None
        assert result.explanation.startswith("Could not compare values:")
