import pytest
from typing import Dict

from src.data_processing import (
    load_and_preprocess_data,
    char_to_index,
    index_to_char,
    count_bigrams,
)


@pytest.mark.order(1)
def test_load_and_preprocess_data():
    """
    Test the load_and_preprocess_data function.

    This test ensures that the function correctly reads a file and processes
    its content into bigrams with start and end tokens. It checks whether the
    returned object is a list, the correct format of bigrams, and the bigrams
    themselves.
    """
    test_file_path = "data/test_text.txt"
    start_token = "<S>"
    end_token = "<E>"

    # Create a simple test file
    test_data = ["hello 1 2", "world 3 4"]
    with open(test_file_path, "w") as f:
        for line in test_data:
            f.write(line + "\n")

    # Load and process data
    bigrams = load_and_preprocess_data(test_file_path, start_token, end_token)

    if bigrams is None:
        pytest.skip()

    # Check if the result is a list of tuples
    assert isinstance(bigrams, list), "The result should be a list."
    assert all(
        isinstance(bigram, tuple) for bigram in bigrams
    ), "All elements in the list should be tuples."

    # Generate expected bigrams for the test data
    expected_bigrams = []
    for line in test_data:
        word = " ".join(line.split()[:-2])  # Ignoring the last two elements
        chs = list(start_token) + list(word) + list(end_token)
        for ch1, ch2 in zip(chs, chs[1:]):
            expected_bigrams.append((ch1, ch2))
    # Check if all expected bigrams are in the result
    assert set(bigrams) == set(
        expected_bigrams
    ), "The bigrams in the result do not match the expected bigrams."


@pytest.mark.order(2)
def test_char_to_index():
    """
    Test the char_to_index function.

    This test checks if the function correctly maps each character, including start and end tokens, to an index.
    """
    alphabet = "abcd"
    start_token = "-"
    end_token = "."

    # Expected result
    expected = {start_token: 0, "a": 1, "b": 2, "c": 3, "d": 4, end_token: 5}

    # Actual result
    result = char_to_index(alphabet, start_token, end_token)

    if result is None:
        pytest.skip()

    # Assert the result matches the expected output
    assert result == expected, f"Expected mapping: {expected}, but got: {result}"


@pytest.mark.order(3)
def test_index_to_char():
    """
    Test the index_to_char function.

    This test checks if the function correctly reverses the mapping from the char_to_index function.
    """
    char_to_idx = {".": 0, "a": 1, "b": 2, "c": 3, "d": 4, "-": 5}

    # Expected result
    expected = {0: ".", 1: "a", 2: "b", 3: "c", 4: "d", 5: "-"}

    # Actual result
    result = index_to_char(char_to_idx)

    if result is None:
        pytest.skip()
    # Assert the result matches the expected output
    assert result == expected, f"Expected mapping: {expected}, but got: {result}"


@pytest.mark.order(4)
def test_count_bigrams():
    """
    Test the count_bigrams function.

    This test checks if the function correctly counts the frequency of each bigram in a given list.
    """
    test_bigrams = [("a", "b"), ("b", "c"), ("a", "b"), ("d", "e")]

    # Create a mapping from characters to indices
    char_to_index: Dict[str, int] = {
        ".": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "-": 6,
    }

    bigram_counts = count_bigrams(test_bigrams, char_to_index)

    if bigram_counts is None:
        pytest.skip()
    assert bigram_counts.shape == (
        len(char_to_index.keys()),
        len(char_to_index.keys()),
    ), "The shape of the count tensor is incorrect"
    assert bigram_counts[1, 2] == 2, "The count for bigram ('a', 'b') should be 2"
    assert bigram_counts[2, 3] == 1, "The count for bigram ('b', 'c') should be 1"
    assert bigram_counts[4, 5] == 1, "The count for bigram ('d', 'e') should be 1"
