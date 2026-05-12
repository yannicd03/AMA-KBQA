from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent


def _agent() -> KQAProAgent:
    return object.__new__(KQAProAgent)


def test_kqapro_extracts_code_constraint():
    constraints = _agent()._extract_exact_attribute_constraints(
        "Which type of sport has IAB code 543?"
    )

    assert {"attribute_name": "IAB code", "value": "543"} in constraints


def test_kqapro_extracts_official_name_constraint():
    constraints = _agent()._extract_exact_attribute_constraints(
        "I would like to know the army that has official name Land Force Command"
    )

    assert {"attribute_name": "official name", "value": "Land Force Command"} in constraints


def test_kqapro_extracts_birth_date_constraint():
    constraints = _agent()._extract_exact_attribute_constraints(
        "When did John Powell born 1936-03-10 die?"
    )

    assert {"attribute_name": "date of birth", "value": "1936-03-10"} in constraints


def test_kqapro_normalizes_iswc_typo():
    constraints = _agent()._extract_exact_attribute_constraints(
        "What name is Massachusetts known under ISCW T-011.363.815-5?"
    )

    assert {"attribute_name": "ISWC", "value": "T-011.363.815-5"} in constraints
