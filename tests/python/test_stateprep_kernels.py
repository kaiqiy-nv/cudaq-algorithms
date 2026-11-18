# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the state-preparation kernels against dense references.

Each ansatz kernel is checked against an independently computed dense
reference: the ordered product of single-term matrix exponentials
``prod_j exp(i * theta_k * c_kj * P_kj)`` applied to the Hartree-Fock
determinant, with the terms taken from the operator pool in kernel
parameter order. The terms within one generator mutually commute, so the
reference is exact even where the circuits interleave the term
exponentials in a different order; the check pins the circuits, the pool
contents, and the parameter ordering.

Invalid-input tests pin the host-side validation: the error cases the
C++ bindings defined (odd qubit counts, odd electrons at spin 0, odd
spin-orbital counts) plus explicit guards where the C++ had undefined
behavior (counts that are negative, fractional, or mutually
inconsistent).
"""

import numpy as np
import pytest

import cudaq
import cudaq_algorithms as algorithms

from dense_references import dense_matrix
from cudaq_algorithms.stateprep._kernels import _ordered_double_excitation

# ----------------------------------------------------------------------------
# Entry kernels
# ----------------------------------------------------------------------------


@cudaq.kernel
def _uccsd_entry(num_qubits: int, occupied: list[int], thetas: list[float],
                 num_electrons: int, spin: int):
    q = cudaq.qvector(num_qubits)
    for i in range(len(occupied)):
        x(q[occupied[i]])
    algorithms.stateprep.uccsd(q, thetas, num_electrons, spin)


@cudaq.kernel
def _single_excitation_entry(num_qubits: int, occupied: list[int],
                             theta: float, p_occ: int, q_virt: int):
    q = cudaq.qvector(num_qubits)
    for i in range(len(occupied)):
        x(q[occupied[i]])
    algorithms.stateprep.single_excitation(q, theta, p_occ, q_virt)


@cudaq.kernel
def _uccgsd_entry(num_qubits: int, occupied: list[int], thetas: list[float],
                  words: list[list[cudaq.pauli_word]],
                  coeffs: list[list[float]]):
    q = cudaq.qvector(num_qubits)
    for i in range(len(occupied)):
        x(q[occupied[i]])
    algorithms.stateprep.uccgsd(q, thetas, words, coeffs)


@cudaq.kernel
def _upccgsd_entry(num_qubits: int, occupied: list[int], thetas: list[float],
                   words: list[list[cudaq.pauli_word]],
                   coeffs: list[list[float]]):
    q = cudaq.qvector(num_qubits)
    for i in range(len(occupied)):
        x(q[occupied[i]])
    algorithms.stateprep.upccgsd(q, thetas, words, coeffs)


@cudaq.kernel
def _ceo_entry(num_qubits: int, occupied: list[int], thetas: list[float],
               words: list[list[cudaq.pauli_word]], coeffs: list[list[float]]):
    q = cudaq.qvector(num_qubits)
    for i in range(len(occupied)):
        x(q[occupied[i]])
    algorithms.stateprep.ceo(q, thetas, words, coeffs)


# ----------------------------------------------------------------------------
# Dense reference machinery
# ----------------------------------------------------------------------------


def _hf_occupation(num_electrons, spin):
    if spin == 0:
        return list(range(num_electrons))
    n_occ_beta = (num_electrons - spin) // 2
    n_occ_alpha = num_electrons - n_occ_beta
    return sorted([2 * i for i in range(n_occ_alpha)] +
                  [2 * i + 1 for i in range(n_occ_beta)])


def _hf_ket(num_qubits, occupation):
    index = sum(1 << orbital for orbital in occupation)
    ket = np.zeros(1 << num_qubits, dtype=np.complex128)
    ket[index] = 1.0
    return ket


def _thetas(seed, count):
    rng = np.random.default_rng(seed)
    return (0.4 * rng.standard_normal(count)).tolist()


def _uccsd_angles(recipe, num_qubits, num_electrons, count):
    # "random" draws both signs. "arithmetic" keeps every |theta| >= 0.2.
    if recipe == "random":
        return _thetas(num_qubits * 100 + num_electrons, count)
    if recipe == "arithmetic":
        return [0.2 + 0.01 * index for index in range(count)]
    raise ValueError(f"unknown UCCSD angle recipe: {recipe}")


def _pool_term_groups(ops, num_qubits):
    """(word, coefficient) term lists per pool operator, in term order."""
    return [[(str(term.get_pauli_word(num_qubits)),
              float(term.evaluate_coefficient().real))
             for term in cudaq.SpinOperator(op)] for op in ops]


def _dense_product_reference(term_groups, thetas, num_qubits, ket, scale=1.0):
    """Apply prod_k prod_j exp(i scale theta_k c_kj P_kj) to ket, in order.

    exp_pauli(angle, qubits, word) implements exp(i * angle * P), so the
    uccgsd/upccgsd/ceo kernels realize scale = +1. The uccsd CNOT-ladder
    circuit uses rz(0.5 * theta) / rz(0.125 * theta) gadgets, which come
    out as exp(-i * (theta / 2) * c * P) in the pool convention:
    scale = -1/2.
    """
    from scipy.linalg import expm

    for theta, terms in zip(thetas, term_groups):
        for word, coefficient in terms:
            generator = dense_matrix([(1.0, word)], num_qubits)
            ket = expm(1.0j * scale * theta * coefficient * generator) @ ket
    return ket


def _state(kernel, *args):
    return np.array(cudaq.get_state(kernel, *args))


def _assert_close(actual, reference):
    # fp32 targets (e.g. the default `nvidia` simulator) land around 5e-6.
    single_precision = np.dtype(cudaq.complex()) == np.dtype(np.complex64)
    tol = 5e-5 if single_precision else 1e-12
    assert np.max(np.abs(actual - reference)) < tol


# ----------------------------------------------------------------------------
# Dense-exponential validation
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("angles", ["random", "arithmetic"])
@pytest.mark.parametrize(
    "num_qubits,num_electrons,spin",
    # (6, 2, 0) emits swapped and sorted mixed doubles in one ansatz.
    # (10, 5, 1) is open-shell (spin>0) and large enough to emit same-spin
    # alpha *and* beta doubles (3 each) -- the occupancy split the smaller
    # (6, 3, 1) case leaves empty.
    # (8, 3, 3) is fully alpha-polarized, with no occupied beta orbitals.
    # (8, 4, 2) interleaves the occupied alpha and virtual beta indices; the
    # assertion below checks that the direct circuit still matches the pool.
    [(4, 2, 0), (6, 2, 0), (6, 3, 1), (8, 4, 0), (10, 5, 1), (8, 4, 2),
     (8, 3, 3)])
def test_uccsd_kernel_matches_dense_exponential(num_qubits, num_electrons,
                                                spin, angles):
    pool = algorithms.stateprep.make_uccsd_operator_pool(
        num_qubits, num_electrons, spin)
    term_groups = _pool_term_groups(pool, num_qubits)
    thetas = _uccsd_angles(angles, num_qubits, num_electrons, len(pool))
    occupation = _hf_occupation(num_electrons, spin)

    actual = _state(_uccsd_entry, num_qubits, occupation, thetas,
                    num_electrons, spin)
    # One scale for every excitation. The pool carries the swap minus, so
    # swapped mixed doubles use -1/2 as well.
    reference = _dense_product_reference(term_groups,
                                         thetas,
                                         num_qubits,
                                         _hf_ket(num_qubits, occupation),
                                         scale=-0.5)
    _assert_close(actual, reference)


def test_single_excitation_matches_dense_exponential():
    num_qubits, num_electrons, spin = 6, 2, 0
    excitation = [0, 4]
    theta = 0.4
    excitations = algorithms.stateprep.get_uccsd_excitations(
        num_qubits, num_electrons, spin)
    pool = algorithms.stateprep.make_uccsd_operator_pool(
        num_qubits, num_electrons, spin)
    excitation_index = excitations[0].index(excitation)
    occupation = _hf_occupation(num_electrons, spin)

    actual = _state(_single_excitation_entry, num_qubits, occupation, theta,
                    *excitation)
    reference = _dense_product_reference(
        [_pool_term_groups(pool, num_qubits)[excitation_index]], [theta],
        num_qubits,
        _hf_ket(num_qubits, occupation),
        scale=-0.5)
    _assert_close(actual, reference)


def test_uccsd_single_amplitude_exercises_nested_excitation_loops():
    num_qubits, num_electrons, spin = 10, 6, 0
    theta = 0.4
    parameter_count = algorithms.stateprep.get_num_uccsd_parameters(
        num_qubits, num_electrons, spin)
    thetas = [0.0] * parameter_count
    thetas[0] = theta
    occupation = _hf_occupation(num_electrons, spin)

    actual = _state(_uccsd_entry, num_qubits, occupation, thetas,
                    num_electrons, spin)
    expected = _state(_single_excitation_entry, num_qubits, occupation, theta,
                      0, 6)
    _assert_close(actual, expected)


def test_uccsd_interleaved_mixed_double_matches_dense_exponential():
    num_qubits, num_electrons, spin = 8, 4, 2
    excitations = algorithms.stateprep.get_uccsd_excitations(
        num_qubits, num_electrons, spin)
    pool = algorithms.stateprep.make_uccsd_operator_pool(
        num_qubits, num_electrons, spin)
    mixed_offset = len(excitations[0]) + len(excitations[1])
    parameter_index = mixed_offset + excitations[2].index([4, 1, 3, 6])
    thetas = [0.0] * len(pool)
    thetas[parameter_index] = 0.4
    occupation = _hf_occupation(num_electrons, spin)

    actual = _state(_uccsd_entry, num_qubits, occupation, thetas,
                    num_electrons, spin)
    reference = _dense_product_reference(_pool_term_groups(pool, num_qubits),
                                         thetas,
                                         num_qubits,
                                         _hf_ket(num_qubits, occupation),
                                         scale=-0.5)
    # Only the [4, 1] -> [3, 6] amplitude is non-zero, so this assertion
    # isolates the mixed double whose occupied and virtual ranges overlap.
    _assert_close(actual, reference)


def test_uccsd_spaced_open_shell_mixed_double_matches_dense_exponential():
    num_qubits, num_electrons, spin = 10, 4, 2
    excitations = algorithms.stateprep.get_uccsd_excitations(
        num_qubits, num_electrons, spin)
    pool = algorithms.stateprep.make_uccsd_operator_pool(
        num_qubits, num_electrons, spin)
    mixed_offset = len(excitations[0]) + len(excitations[1])
    parameter_index = mixed_offset + excitations[2].index([4, 1, 3, 8])
    thetas = [0.0] * len(pool)
    thetas[parameter_index] = 0.4
    occupation = _hf_occupation(num_electrons, spin)

    actual = _state(_uccsd_entry, num_qubits, occupation, thetas,
                    num_electrons, spin)
    reference = _dense_product_reference(_pool_term_groups(pool, num_qubits),
                                         thetas,
                                         num_qubits,
                                         _hf_ket(num_qubits, occupation),
                                         scale=-0.5)
    _assert_close(actual, reference)


def test_uccsd_spaced_alpha_double_matches_dense_exponential():
    num_qubits, num_electrons, spin = 8, 4, 0
    excitations = algorithms.stateprep.get_uccsd_excitations(
        num_qubits, num_electrons, spin)
    pool = algorithms.stateprep.make_uccsd_operator_pool(
        num_qubits, num_electrons, spin)
    double_offset = sum(len(group) for group in excitations[:3])
    parameter_index = double_offset + excitations[3].index([0, 2, 4, 6])
    thetas = [0.0] * len(pool)
    thetas[parameter_index] = 0.4
    occupation = _hf_occupation(num_electrons, spin)

    actual = _state(_uccsd_entry, num_qubits, occupation, thetas,
                    num_electrons, spin)
    reference = _dense_product_reference(_pool_term_groups(pool, num_qubits),
                                         thetas,
                                         num_qubits,
                                         _hf_ket(num_qubits, occupation),
                                         scale=-0.5)
    _assert_close(actual, reference)


@pytest.mark.parametrize("num_qubits", [4, 6])
def test_uccgsd_kernel_matches_dense_exponential(num_qubits):
    words, coeffs = algorithms.stateprep.get_uccgsd_pauli_lists(num_qubits)
    pool = algorithms.stateprep.make_uccgsd_operator_pool(num_qubits)
    term_groups = _pool_term_groups(pool, num_qubits)
    thetas = _thetas(num_qubits, len(words))
    occupation = list(range(num_qubits // 2))

    actual = _state(_uccgsd_entry, num_qubits, occupation, thetas, words,
                    coeffs)
    reference = _dense_product_reference(term_groups, thetas, num_qubits,
                                         _hf_ket(num_qubits, occupation))
    _assert_close(actual, reference)


def test_uccgsd_minimal_single_matches_dense_exponential():
    num_qubits = 2
    words, coeffs = algorithms.stateprep.get_uccgsd_pauli_lists(
        num_qubits, only_singles=True)
    pool = algorithms.stateprep.make_uccgsd_operator_pool(num_qubits,
                                                          only_singles=True)
    assert len(words) == len(coeffs) == 1
    assert len(words[0]) == len(coeffs[0]) == 2

    thetas = [0.37]
    occupation = [0]
    actual = _state(_uccgsd_entry, num_qubits, occupation, thetas, words,
                    coeffs)
    reference = _dense_product_reference(_pool_term_groups(pool, num_qubits),
                                         thetas, num_qubits,
                                         _hf_ket(num_qubits, occupation))
    _assert_close(actual, reference)


@pytest.mark.parametrize("num_qubits", [4, 8])
def test_upccgsd_kernel_matches_dense_exponential(num_qubits):
    words, coeffs = algorithms.stateprep.get_upccgsd_pauli_lists(num_qubits)
    pool = algorithms.stateprep.make_upccgsd_operator_pool(num_qubits)
    term_groups = _pool_term_groups(pool, num_qubits)
    thetas = _thetas(num_qubits + 7, len(words))
    occupation = list(range(num_qubits // 2))

    actual = _state(_upccgsd_entry, num_qubits, occupation, thetas, words,
                    coeffs)
    reference = _dense_product_reference(term_groups, thetas, num_qubits,
                                         _hf_ket(num_qubits, occupation))
    _assert_close(actual, reference)


def test_upccgsd_kernel_empty_and_zero_term_groups():
    num_qubits = 4
    occupation = list(range(num_qubits // 2))
    hf_ket = _hf_ket(num_qubits, occupation)

    actual = _state(_upccgsd_entry, num_qubits, occupation, [], [], [])
    _assert_close(actual, hf_ket)

    words, coeffs = algorithms.stateprep.get_upccgsd_pauli_lists(num_qubits)
    pool = algorithms.stateprep.make_upccgsd_operator_pool(num_qubits)
    term_groups = _pool_term_groups(pool, num_qubits)
    thetas = [0.2 + 0.01 * index for index in range(len(words))]
    words = [list(group) for group in words]
    coeffs = [list(group) for group in coeffs]
    words[0] = []
    coeffs[0] = []
    term_groups[0] = []

    actual = _state(_upccgsd_entry, num_qubits, occupation, thetas, words,
                    coeffs)
    reference = _dense_product_reference(term_groups, thetas, num_qubits,
                                         hf_ket)
    _assert_close(actual, reference)


# num_orbitals = 4 is the smallest size with same-spin doubles
# (they need four descending spatial orbitals).
@pytest.mark.parametrize("num_orbitals", [2, 3, 4])
def test_ceo_kernel_matches_dense_exponential(num_orbitals):
    num_qubits = 2 * num_orbitals
    words, coeffs = algorithms.stateprep.get_ceo_pauli_lists(num_orbitals)
    pool = algorithms.stateprep.make_ceo_operator_pool(num_orbitals)
    term_groups = _pool_term_groups(pool, num_qubits)
    thetas = _thetas(num_orbitals + 13, len(words))
    occupation = list(range(num_qubits // 2))

    actual = _state(_ceo_entry, num_qubits, occupation, thetas, words, coeffs)
    reference = _dense_product_reference(term_groups, thetas, num_qubits,
                                         _hf_ket(num_qubits, occupation))
    _assert_close(actual, reference)


def test_ceo_kernel_empty_and_nonempty_groups_match_dense():
    num_qubits = 4
    occupation = list(range(num_qubits // 2))
    hf_ket = _hf_ket(num_qubits, occupation)

    actual = _state(_ceo_entry, num_qubits, occupation, [], [], [])
    _assert_close(actual, hf_ket)

    words, coeffs = algorithms.stateprep.get_ceo_pauli_lists(2)
    pool = algorithms.stateprep.make_ceo_operator_pool(2)
    term_groups = _pool_term_groups(pool, num_qubits)
    thetas = [0.3, 0.4]
    grouped_words = [[], list(words[0])]
    grouped_coeffs = [[], list(coeffs[0])]
    grouped_terms = [[], term_groups[0]]

    actual = _state(_ceo_entry, num_qubits, occupation, thetas, grouped_words,
                    grouped_coeffs)
    reference = _dense_product_reference(grouped_terms, thetas, num_qubits,
                                         hf_ket)
    _assert_close(actual, reference)


# ----------------------------------------------------------------------------
# Host-side input validation
# ----------------------------------------------------------------------------


def test_invalid_inputs_raise():
    stateprep = algorithms.stateprep

    # Error cases the C++ bindings also defined (as ValueError, like every
    # other input guard in the package).
    with pytest.raises(ValueError, match="should be even"):
        stateprep.get_uccsd_excitations(5, 2, 0)
    with pytest.raises(ValueError, match="spin multiplicity"):
        stateprep.get_uccsd_excitations(4, 3, 0)
    with pytest.raises(ValueError, match="even number"):
        stateprep.make_upccgsd_operator_pool(7)

    # Guards where the C++ had undefined behavior (unsigned underflow)
    # or rejected the value at the size_t type level.
    with pytest.raises(ValueError, match="num_electrons cannot exceed"):
        stateprep.get_uccsd_excitations(4, 6, 0)
    with pytest.raises(ValueError, match="spin cannot exceed"):
        stateprep.get_uccsd_excitations(4, 1, 3)
    with pytest.raises(ValueError, match="does not fit"):
        stateprep.get_uccsd_excitations(4, 4, 2)
    with pytest.raises(ValueError, match="non-negative integer"):
        stateprep.get_uccsd_excitations(4, 2.5, 0)
    with pytest.raises(ValueError, match="non-negative integer"):
        stateprep.make_uccgsd_operator_pool(-4)
    with pytest.raises(ValueError, match="non-negative integer"):
        stateprep.make_ceo_operator_pool(-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        stateprep.get_ceo_pauli_lists(1.5)
    with pytest.raises(ValueError, match="non-negative integer"):
        stateprep.make_ceo_operator_pool(True)


def test_uccsd_rejects_open_shell_spin_with_odd_electron_difference():
    # spin is 2*S_z, so (num_electrons - spin) must be even when spin > 0.
    # An odd difference floors to spin+1, so (8, 4, 1) aliases (8, 4, 2).
    stateprep = algorithms.stateprep

    # Even electron count and odd spin: the enumerator itself must raise.
    with pytest.raises(ValueError, match="must be even when spin > 0"):
        stateprep.get_uccsd_excitations(8, 4, 1)
    # Odd electron count and even spin > 0. The pool is built by the same
    # enumerator, so the rejection must surface on this entry point too.
    with pytest.raises(ValueError, match="must be even when spin > 0"):
        stateprep.make_uccsd_operator_pool(6, 3, 2)
    # (10, 5, 2) would otherwise be a non-empty pool at the promoted spin 3.
    # The parameter counter uses the same enumerator and must raise first.
    with pytest.raises(ValueError, match="must be even when spin > 0"):
        stateprep.get_num_uccsd_parameters(10, 5, 2)


# ----------------------------------------------------------------------------
# UCCSD ladder / pool angle scale
# ----------------------------------------------------------------------------


@cudaq.kernel
def _one_double_entry(theta: float, p_occ: int, q_occ: int, r_virt: int,
                      s_virt: int):
    qubits = cudaq.qvector(4)
    x(qubits[0])
    x(qubits[1])
    algorithms.stateprep.double_excitation(qubits, theta, p_occ, q_occ, r_virt,
                                           s_virt)


@cudaq.kernel
def _ordered_double_entry(theta: float, p_occ: int, q_occ: int, r_virt: int,
                          s_virt: int):
    qubits = cudaq.qvector(4)
    x(qubits[0])
    x(qubits[1])
    _ordered_double_excitation(qubits, theta, p_occ, q_occ, r_virt, s_virt)


@cudaq.kernel
def _gapped_double_entry(theta: float, p_occ: int, q_occ: int, r_virt: int,
                         s_virt: int):
    qubits = cudaq.qvector(8)
    x(qubits[0])
    x(qubits[2])
    algorithms.stateprep.double_excitation(qubits, theta, p_occ, q_occ, r_virt,
                                           s_virt)


@pytest.mark.parametrize("p_occ,q_occ,r_virt,s_virt,sign",
                         [(0, 1, 2, 3, 1.0), (1, 0, 3, 2, 1.0),
                          (0, 1, 3, 2, -1.0), (1, 0, 2, 3, -1.0)])
def test_double_excitation_normalizes_index_order(p_occ, q_occ, r_virt, s_virt,
                                                  sign):
    # Reversing exactly one pair negates the excitation generator.
    theta = 0.4
    actual = _state(_one_double_entry, theta, p_occ, q_occ, r_virt, s_virt)
    ordered = _state(_one_double_entry, sign * theta, 0, 1, 2, 3)
    # Same state, not merely the same probabilities: the relative phase
    # between the Hartree-Fock determinant and the double excitation is
    # the correlation sign.
    _assert_close(actual, ordered)


@pytest.mark.parametrize(
    "indices,effective_sign",
    [
        pytest.param((0, 2, 4, 6), 1.0, id="ordered"),
        pytest.param((2, 0, 4, 6), -1.0, id="occupied-pair-reversed"),
        pytest.param((0, 2, 6, 4), -1.0, id="virtual-pair-reversed"),
        pytest.param((2, 0, 6, 4), 1.0, id="both-pairs-reversed"),
    ],
)
def test_double_excitation_ordered_permutations_preserve_state(
        indices, effective_sign):
    theta = 0.4
    actual = _state(_gapped_double_entry, theta, *indices)
    canonical = _state(_gapped_double_entry, effective_sign * theta, 0, 2, 4,
                       6)
    _assert_close(actual, canonical)


@pytest.mark.parametrize(
    "indices,effective_sign",
    [
        pytest.param((0, 1, 2, 3), 1.0, id="ordered"),
        pytest.param((1, 0, 3, 2), 1.0, id="both-pairs-reversed"),
        pytest.param((1, 0, 2, 3), -1.0, id="occupied-pair-reversed"),
        pytest.param((0, 1, 3, 2), -1.0, id="virtual-pair-reversed"),
    ],
)
def test_ordered_double_excitation_matches_dense_for_index_orderings(
        indices, effective_sign):
    theta = 0.4
    actual = _state(_ordered_double_entry, theta, *indices)

    pool = algorithms.stateprep.make_uccsd_operator_pool(4, 2, 0)
    reference = _dense_product_reference(_pool_term_groups(pool, 4),
                                         [0.0, 0.0, -effective_sign * theta],
                                         4,
                                         _hf_ket(4, [0, 1]),
                                         scale=-0.5)
    _assert_close(actual, reference)
