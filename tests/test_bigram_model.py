import torch
import pytest

from src.bigram_model import (
    bigrams_count_to_probabilities,
    sample_next_character,
    calculate_log_likelihood,
    calculate_neg_mean_log_likelihood,
)


@pytest.mark.order(5)
def test_bigrams_count_to_probabilities():
    """
    Test the bigrams_count_to_probabilities function.

    This test checks if the function correctly converts bigram counts to a probability
    distribution by normalizing each row of the input tensor.
    """
    # Example bigram counts
    bigram_counts = torch.tensor([[2, 2, 0], [0, 1, 1], [3, 0, 0]], dtype=torch.float32)

    # Expected probabilities
    expected_probabilities = torch.tensor(
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0]]
    )

    # Calculate probabilities
    calculated_probabilities = bigrams_count_to_probabilities(bigram_counts)

    if calculated_probabilities is None:
        pytest.skip()
    # Assert each row sums to 1 and the probabilities match the expected values
    assert torch.allclose(calculated_probabilities.sum(dim=1), torch.ones(3))
    assert torch.allclose(calculated_probabilities, expected_probabilities)


@pytest.mark.order(6)
def test_bigrams_count_to_probabilities_smoothed():
    """
    Test the bigrams_count_to_probabilities function.

    This test checks if the function correctly converts bigram counts to a probability
    distribution by normalizing each row of the input tensor.
    """
    # Example bigram counts
    bigram_counts = torch.tensor([[2, 2, 0], [0, 1, 1], [3, 0, 0]], dtype=torch.float32)

    # Expected probabilities
    expected_probabilities = torch.tensor(
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0]]
    )

    # Calculate probabilities
    calculated_probabilities = bigrams_count_to_probabilities(bigram_counts)

    if calculated_probabilities is None:
        pytest.skip()
    # Assert each row sums to 1 and the probabilities match the expected values
    assert torch.allclose(calculated_probabilities.sum(dim=1), torch.ones(3))
    assert torch.allclose(calculated_probabilities, expected_probabilities)

    # Example bigram counts with smoothing factor
    bigram_counts_smoothed = torch.tensor(
        [[2, 2, 0], [0, 1, 1], [3, 0, 0]], dtype=torch.float32
    )

    # Smooth factor
    smooth_factor = 2

    # Expected probabilities with smoothing
    expected_probabilities_smoothed = torch.tensor(
        [[0.4000, 0.4000, 0.2000], [0.2500, 0.3750, 0.3750], [0.5556, 0.2222, 0.2222]]
    )

    # Calculate probabilities with smoothing
    calculated_probabilities_smoothed = bigrams_count_to_probabilities(
        bigram_counts_smoothed, smooth_factor
    )

    if calculated_probabilities_smoothed is None:
        pytest.skip()

    if torch.allclose(calculated_probabilities_smoothed, calculated_probabilities):
        pytest.skip()
    # Assert each row sums to 1 and the probabilities match the expected values with smoothing
    assert torch.allclose(calculated_probabilities_smoothed.sum(dim=1), torch.ones(3))
    assert torch.allclose(
        calculated_probabilities_smoothed, expected_probabilities_smoothed, atol=0.0001
    )


@pytest.mark.order(7)
def test_calculate_log_likelihood():
    """
    Test the calculate_log_likelihood function with a 4x4 bigram probability tensor.

    Args:
        None

    Returns:
        None
    """
    # Example bigram probabilities (4x4 for '-', 'a', 'b', '.')
    bigram_probabilities = torch.tensor(
        [
            [0.1, 0.3, 0.4, 0.2],  # Probabilities starting from '-'
            [0.3, 0.1, 0.4, 0.2],  # Probabilities starting from 'a'
            [0.2, 0.3, 0.2, 0.3],  # Probabilities starting from 'b'
            [0.3, 0.3, 0.2, 0.2],  # Probabilities starting from '.'
        ],
        dtype=torch.float32,
    )

    # Character to index mapping (including start '-' and end '.' characters)
    char_to_index = {"-": 0, "a": 1, "b": 2, ".": 3}

    # Example word
    word = "ab"

    # Calculate log likelihood
    log_likelihood = calculate_log_likelihood(
        word, bigram_probabilities, char_to_index, "-", "."
    )

    # Expected log likelihood calculation
    expected_log_likelihood = (
        torch.log(torch.tensor(0.3))
        + torch.log(torch.tensor(0.4))
        + torch.log(torch.tensor(0.3))
    )
    if log_likelihood is None:
        pytest.skip()
    # Assert the calculated log likelihood matches the expected value
    assert (
        abs(log_likelihood.item() - expected_log_likelihood.item()) < 1e-6
    ), f"Wrong calculated log likelihood. Expected {expected_log_likelihood:.4f}, obtained {log_likelihood:.4f}."


@pytest.mark.order(8)
def test_calculate_mean_log_likelihood():
    """
    Test the calculate_mean_log_likelihood function.

    This test checks if the function correctly computes the mean log likelihood
    for a given list of words, based on predefined bigram probabilities.
    """
    # Example bigram probabilities (simplified for testing)
    bigram_probs = torch.tensor(
        [
            [0.1, 0.8, 0.1],  # Probabilities from '-'
            [0.7, 0.2, 0.1],  # Probabilities from 'a'
            [0.1, 0.3, 0.6],  # Probabilities from '.'
        ],
        dtype=torch.float32,
    )

    # Example character to index mapping
    char_to_index = {"-": 0, "a": 1, ".": 2}

    # List of words to test
    words = ["a"]

    # Expected mean log likelihood
    # Calculated as log(0.9) for '-a' + log(0.2) for 'a.'
    expected_mean_log_likelihood = (
        torch.log(torch.tensor(0.8)) + torch.log(torch.tensor(0.1))
    ).item() / 1

    # Calculate mean log likelihood
    calculated_mean_log_likelihood = calculate_neg_mean_log_likelihood(
        words, bigram_probs, char_to_index, "-", "."
    )

    if calculated_mean_log_likelihood is None:
        pytest.skip()

    # Assert the calculated mean log likelihood matches the expected value
    assert (
        abs(calculated_mean_log_likelihood + expected_mean_log_likelihood) < 1e-6
    ), f"Expected mean log likelihood: {expected_mean_log_likelihood}, but got: {calculated_mean_log_likelihood}"


@pytest.mark.order(9)
def test_deterministic_sampling():
    """Test sampling when the probability distribution is deterministic (one-hot)."""
    prob_dist = torch.tensor([[0.0, 1.0]], dtype=torch.float)
    idx_to_char = {0: 'a', 1: 'b'}
    current_char_index = 0
    result = sample_next_character(current_char_index, prob_dist, idx_to_char)
    assert result == 'b'

@pytest.mark.order(10)
def test_uses_correct_probability_row():
    """Test that the correct row in the probability distribution is used based on current_char_index."""
    prob_dist = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ], dtype=torch.float)
    idx_to_char = {0: 'a', 1: 'b', 2: 'c'}
    
    # Check next character when current_char_index is 1
    assert sample_next_character(1, prob_dist, idx_to_char) == 'c'
    
    # Check next character when current_char_index is 2
    assert sample_next_character(2, prob_dist, idx_to_char) == 'a'

    
@pytest.mark.order(11)
def test_non_zero_index_mapping():
    """Test correct handling when the next character index is non-zero and mapped correctly."""
    prob_dist = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float)
    idx_to_char = {2: 'c'}
    current_char_index = 0
    
    result = sample_next_character(current_char_index, prob_dist, idx_to_char)
    assert result == 'c'
