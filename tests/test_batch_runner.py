"""
Tests for the batch runner functionality.

These tests verify the sampling, file structure, and metadata tracking
without actually running the agent (to avoid API costs).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ama_kbqa.agents.kqapro_agent.batch_runner import (
    load_validation_dataset,
    sample_questions,
    classify_question_type,
    get_next_batch_folder,
    VALIDATION_DATASET_PATH,
    BATCH_RESULTS_BASE_DIR
)


def test_load_validation_dataset():
    """Test that validation dataset loads correctly"""
    print("\n" + "="*80)
    print("Test 1: Load Validation Dataset")
    print("="*80)

    data = load_validation_dataset()

    assert len(data) > 0, "Validation dataset should not be empty"
    assert "question" in data[0], "Each item should have a 'question' field"

    print(f"[PASS] Loaded {len(data)} questions")
    print(f"[PASS] Sample question: {data[0]['question'][:60]}...")

    return True


def test_sampling_reproducibility():
    """Test that sampling with the same seed produces same results"""
    print("\n" + "="*80)
    print("Test 2: Sampling Reproducibility")
    print("="*80)

    data = load_validation_dataset()

    # Sample twice with the same seed
    sample1 = sample_questions(data, 10, seed=123)
    sample2 = sample_questions(data, 10, seed=123)

    # Check they're identical
    questions1 = [q["question"] for q in sample1]
    questions2 = [q["question"] for q in sample2]

    assert questions1 == questions2, "Same seed should produce same sample"

    print(f"[PASS] Sampling is reproducible with seed=123")
    print(f"[PASS] Both samples have {len(sample1)} questions")

    return True


def test_different_seeds():
    """Test that different seeds produce different samples"""
    print("\n" + "="*80)
    print("Test 3: Different Seeds Produce Different Samples")
    print("="*80)

    data = load_validation_dataset()

    # Sample with different seeds
    sample1 = sample_questions(data, 10, seed=111)
    sample2 = sample_questions(data, 10, seed=222)

    questions1 = [q["question"] for q in sample1]
    questions2 = [q["question"] for q in sample2]

    assert questions1 != questions2, "Different seeds should produce different samples"

    print(f"[PASS] seed=111 and seed=222 produce different samples")

    return True


def test_question_type_classification():
    """Test question type classification"""
    print("\n" + "="*80)
    print("Test 4: Question Type Classification")
    print("="*80)

    test_cases = [
        ("How many people live in New York?", None, "Count"),
        ("Is Paris the capital of France?", None, "Verify"),
        ("What is the capital of Germany?", None, "Query"),
        ("Who directed Inception?", None, "Query"),
    ]

    for question, program, expected_type in test_cases:
        qtype = classify_question_type(question, program)
        assert qtype == expected_type, f"Expected {expected_type}, got {qtype} for: {question}"
        print(f"[PASS] '{question[:40]}...' -> {qtype}")

    # Test with program structure
    program_count = [{"function": "Count"}]
    qtype = classify_question_type("Some question", program_count)
    assert qtype == "Count", f"Program with Count should return Count, got {qtype}"
    print(f"[PASS] Program-based classification works")

    return True


def test_batch_folder_creation():
    """Test batch folder numbering"""
    print("\n" + "="*80)
    print("Test 5: Batch Folder Structure")
    print("="*80)

    # Check that batch_results folder exists
    assert BATCH_RESULTS_BASE_DIR.exists(), f"Batch results dir should exist: {BATCH_RESULTS_BASE_DIR}"

    # Get next batch folder
    next_folder = get_next_batch_folder()

    assert next_folder.exists(), "Batch folder should be created"
    assert "Batch" in next_folder.name, "Folder name should contain 'Batch'"
    assert next_folder.name[5:].isdigit(), "Batch number should be numeric"

    print(f"[PASS] Batch folder created: {next_folder}")
    print(f"[PASS] Batch folder structure is correct")

    return True


def test_paths_configuration():
    """Test that all paths are configured correctly"""
    print("\n" + "="*80)
    print("Test 6: Path Configuration")
    print("="*80)

    assert VALIDATION_DATASET_PATH.exists(), f"Validation dataset should exist: {VALIDATION_DATASET_PATH}"
    assert BATCH_RESULTS_BASE_DIR.exists(), f"Batch results dir should exist: {BATCH_RESULTS_BASE_DIR}"

    print(f"[PASS] Validation dataset: {VALIDATION_DATASET_PATH}")
    print(f"[PASS] Batch results dir: {BATCH_RESULTS_BASE_DIR}")
    print(f"[PASS] All paths configured correctly")

    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("BATCH RUNNER TEST SUITE")
    print("="*80)

    tests = [
        test_paths_configuration,
        test_load_validation_dataset,
        test_sampling_reproducibility,
        test_different_seeds,
        test_question_type_classification,
        test_batch_folder_creation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"\n[FAIL] {test.__name__} failed with error: {e}")
            failed += 1

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("="*80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
