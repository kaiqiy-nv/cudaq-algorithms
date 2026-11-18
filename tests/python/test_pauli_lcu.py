# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the PauliLCU block encoding (dense references)."""

import math

import numpy as np
import pytest

import cudaq
from cudaq import spin

from cudaq_algorithms import sim_utils as sim
from cudaq_algorithms import (PauliLCU, PhaseSequence, QSVT, SumEncoding,
                              state_from)
from cudaq_algorithms.common_kernels import reflect_about_zero, signal_phase
from cudaq_algorithms.pauli_lcu import (_prepare_angles, adjoint_walk, apply,
                                        apply_controlled_phase_sequence,
                                        apply_phase_sequence, prepare, select,
                                        unprepare, walk)

from dense_references import dense_matrix, random_ket

FOUR_TERMS = {"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11}


def test_good_subspace_rejects_statevector_with_wrong_dimension():
    enc = PauliLCU(FOUR_TERMS)
    expected = 1 << (enc.num_system + enc.num_ancilla)
    state = np.zeros(expected - 1, dtype=np.complex128)

    with pytest.raises(ValueError) as error:
        sim.good_subspace(enc, state)

    assert str(
        error.value) == (f"expected a statevector of dimension {expected}, "
                         f"got shape {(expected - 1,)}")


def test_action_matches_dense_hamiltonian():
    enc = PauliLCU(FOUR_TERMS)
    assert enc.num_system == 2
    assert enc.num_ancilla == 2
    assert enc.alpha == pytest.approx(1.43)

    ket = random_ket(2, seed=7)
    expected = dense_matrix(list(
        (c, w) for w, c in FOUR_TERMS.items()), 2) @ ket / enc.alpha
    assert np.allclose(sim.action(enc, ket), expected, atol=1e-10)


def test_spin_operator_and_pairs_inputs_agree():
    h = 0.7 * spin.z(0) - 0.43 * spin.z(1) + 0.19 * spin.x(0) * spin.x(1)
    from_op = PauliLCU(h, num_qubits=2)
    from_pairs = PauliLCU([(0.7, "ZI"), (-0.43, "IZ"), (0.19, "XX")])

    assert from_op.alpha == pytest.approx(from_pairs.alpha)
    ket = random_ket(2, seed=11)
    assert np.allclose(sim.action(from_op, ket),
                       sim.action(from_pairs, ket),
                       atol=1e-10)


def test_single_term_negative_coefficient_keeps_sign():
    # Single-term regression: -c * P must encode -c * P, not +c * P.
    # Single-term encodings are normalized to one (idle) ancilla so the
    # full walk/QSVT machinery applies to them unchanged.
    enc = PauliLCU({"XZ": -0.5})
    assert enc.num_ancilla == 1

    ket = random_ket(2, seed=3)
    expected = dense_matrix([(-0.5, "XZ")], 2) @ ket / enc.alpha
    assert np.allclose(sim.action(enc, ket), expected, atol=1e-10)


def test_single_term_walk_keeps_minus_h_over_alpha_sign():
    # walk_kernel's block must be T_power(-H/alpha) for single-term
    # encodings too (regression: the old 0-ancilla path dropped the sign).
    enc = PauliLCU({"X": 1.0})
    ket = np.array([0.6, 0.8], dtype=np.complex128)
    state = np.asarray(
        cudaq.get_state(enc.walk_kernel(power=1), sim.state_from(ket)))
    block = sim.good_subspace(enc, state)
    expected = -dense_matrix([(1.0, "X")], 1) @ ket  # T_1(-H/alpha)|ket>
    assert np.allclose(block, expected, atol=1e-10)


def test_walk_kernel_factories_match_prepared_dense_reference():
    enc = PauliLCU(FOUR_TERMS, backend="exact")

    @cudaq.kernel
    def prep(system: cudaq.qview):
        h(system[0])
        x(system[1])

    expected_ket = np.array([0.0, 0.0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                            dtype=np.complex128)
    prep_walked = enc.walk_kernel(power=1, state_prep=prep)
    walked = enc.walk_kernel(power=1)
    prep_walked_result = np.asarray(cudaq.get_state(prep_walked))
    walked_result = np.asarray(
        cudaq.get_state(walked, sim.state_from(expected_ket)))

    walk_step = enc.walk_step_kernel()
    prepare_ancilla = enc.prepare_kernel()
    unprepare_ancilla = enc.unprepare_kernel()
    n_ancilla = enc.num_ancilla

    @cudaq.kernel
    def composed(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_ancilla)
        prepare_ancilla(ancilla)
        walk_step(ancilla, system)
        unprepare_ancilla(ancilla)

    result = np.asarray(cudaq.get_state(composed,
                                        sim.state_from(expected_ket)))
    assert np.allclose(prep_walked_result, walked_result, atol=1e-10)
    assert np.allclose(result, walked_result, atol=1e-10)

    expected_block = (
        -dense_matrix([(coefficient, word)
                       for word, coefficient in FOUR_TERMS.items()], 2)
        @ expected_ket / enc.alpha)
    assert np.allclose(sim.good_subspace(enc, result),
                       expected_block,
                       atol=1e-10)


def test_identity_term_handling():
    enc = PauliLCU({"II": 0.2, "XI": 0.5, "ZI": 0.3})
    assert enc.constant_term == pytest.approx(0.2)
    assert enc.num_terms == 3
    assert enc.alpha == pytest.approx(1.0)

    excluded = PauliLCU({
        "II": 0.2,
        "XI": 0.5,
        "ZI": 0.3
    },
                        include_identity=False)
    assert excluded.constant_term == pytest.approx(0.2)
    assert excluded.num_terms == 2
    assert excluded.alpha == pytest.approx(0.8)


def test_walk_moments_match_chebyshev():
    # Asymmetric spectrum 0.2 +/- sqrt(0.34): the reflection expectation after
    # k walks must reproduce <T_2k(H/alpha)> with both eigenvalue weights.
    enc = PauliLCU({"I": 0.2, "X": 0.5, "Z": 0.3})
    assert enc.num_ancilla == 2

    lam = math.sqrt(0.34)
    theta = math.atan2(0.5, 0.3)
    prep_angle = 0.7
    delta = prep_angle - theta
    weights = (math.cos(delta / 2)**2, math.sin(delta / 2)**2)
    eigenvalues = (0.2 + lam, 0.2 - lam)

    ket = np.array([math.cos(prep_angle / 2),
                    math.sin(prep_angle / 2)],
                   dtype=np.complex128)
    for k in (1, 2, 3):
        state = cudaq.get_state(enc.walk_kernel(power=k), state_from(ket))
        zero_probability = float(
            np.sum(np.abs(sim.good_subspace(enc, state))**2))
        moment = 2.0 * zero_probability - 1.0

        expected = sum(w * math.cos(2 * k * math.acos(e / enc.alpha))
                       for w, e in zip(weights, eigenvalues))
        assert moment == pytest.approx(expected, abs=1e-10)


def test_validation_errors():
    with pytest.raises(ValueError):
        PauliLCU({})
    with pytest.raises(ValueError):
        PauliLCU({"XI": 0.5, "XII": 0.3})
    with pytest.raises(ValueError):
        PauliLCU({"XQ": 0.5})
    with pytest.raises(ValueError):
        PauliLCU({"XI": 0.5}, num_qubits=3)
    with pytest.raises(TypeError):
        PauliLCU(42)


def test_repr_reads_well():
    text = repr(PauliLCU(FOUR_TERMS))
    assert "terms=4" in text
    assert "ancilla_qubits=2" in text


def test_module_level_phase_sequence_kernels_compose():
    # Regression: apply_phase_sequence once referenced a stale kernel name
    # and could not compile; both escape hatches must stay composable and
    # agree with the QSVT factories.
    enc = PauliLCU({"ZI": 0.7, "IZ": -0.43, "XX": 0.19})
    seq = PhaseSequence([0.4, -0.2, 0.7])
    phases = seq.projector_phases
    directions = list(seq.walk_directions)
    angles, controls, ops, lengths, signs = enc.kernel_args
    n_anc = enc.num_ancilla
    ket = random_ket(2, seed=13)

    @cudaq.kernel
    def composed(state: cudaq.State):
        system = cudaq.qvector(state)
        signal = cudaq.qvector(n_anc)
        apply_phase_sequence(signal, system, phases, directions, angles,
                             controls, ops, lengths, signs)

    reference = np.asarray(
        cudaq.get_state(QSVT(enc).kernel(seq), sim.state_from(ket)))
    state = np.asarray(cudaq.get_state(composed, sim.state_from(ket)))
    assert np.allclose(state, reference, atol=1e-12)

    @cudaq.kernel
    def composed_controlled(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_signal = cudaq.qvector(1 + n_anc)
        x(control_and_signal[0])
        apply_controlled_phase_sequence(control_and_signal, system, phases,
                                        directions, angles, controls, ops,
                                        lengths, signs)

    reference = np.asarray(
        cudaq.get_state(
            QSVT(enc).controlled_kernel(seq, control_state=1),
            sim.state_from(ket)))
    state = np.asarray(
        cudaq.get_state(composed_controlled, sim.state_from(ket)))
    assert np.allclose(state, reference, atol=1e-12)


def test_kernels_tolerate_empty_register_views():
    # Regression: early `return` in kernels is silently ignored
    # (cuda-quantum#4845); the guards must be positive blocks, or an
    # empty-view call executes the body and crashes the process.
    angles = [0.3]

    @cudaq.kernel
    def probe():
        q = cudaq.qvector(2)
        prepare(q.front(0), angles)
        unprepare(q.front(0), angles)
        reflect_about_zero(q.front(0))
        signal_phase(q.front(0), 0.7)

    state = np.asarray(cudaq.get_state(probe))
    expected = np.zeros(4, dtype=np.complex128)
    expected[0] = 1.0
    assert np.allclose(state, expected, atol=1e-12)


def test_identity_only_hamiltonian_encodes_signed_identity():
    # Regression: identity-only Hamiltonians produce an empty term_ops
    # list, which cannot cross the kernel boundary (cuda-quantum#4847);
    # the extraction pads it with a never-dereferenced entry.
    ket = random_ket(2, seed=21)
    positive = PauliLCU({"II": 1.5})
    assert positive.num_ancilla == 1
    assert np.allclose(sim.action(positive, ket), ket, atol=1e-12)

    negative = PauliLCU({"II": -0.5})
    assert np.allclose(sim.action(negative, ket), -ket, atol=1e-12)

    # The walk machinery applies unchanged: T_1(-H/alpha) = -sign * I.
    walk_state = np.asarray(
        cudaq.get_state(negative.walk_kernel(power=1), sim.state_from(ket)))
    assert np.allclose(sim.good_subspace(negative, walk_state),
                       ket,
                       atol=1e-12)


def test_unprepare_inverts_prepare():
    # cudaq.adjoint autogeneration cannot replace unprepare: it fails
    # loudly on prepare's conditionally-conjugated rotations
    # (cuda-quantum#4898) and silently mis-replays loop-carried classical
    # updates elsewhere (cuda-quantum#4897). unprepare stays hand-written;
    # this pins the inverse property directly on an arbitrary state.
    enc = PauliLCU(
        {
            "ZI": 0.70,
            "IZ": -0.43,
            "XX": 0.19,
            "YZ": 0.11,
            "XY": 0.05
        },
        backend="exact")
    prepare_ancilla = enc.prepare_kernel()
    unprepare_ancilla = enc.unprepare_kernel()
    n_anc = enc.num_ancilla
    ket = random_ket(n_anc, seed=17)

    @cudaq.kernel
    def roundtrip(state: cudaq.State):
        q = cudaq.qvector(state)
        prepare_ancilla(q)
        unprepare_ancilla(q)

    state = np.asarray(cudaq.get_state(roundtrip, sim.state_from(ket)))
    assert np.allclose(state, ket, atol=1e-12)


def test_unprepare_inverts_prepare_with_four_ancillas():
    enc = PauliLCU({
        "IX": 0.60,
        "IY": -0.25,
        "IZ": 0.31,
        "XI": -0.47,
        "XX": 0.52,
        "XY": -0.19,
        "XZ": 0.23,
        "YI": -0.36,
        "YX": 0.41
    })
    assert enc.num_ancilla == 4
    angles = enc.kernel_args[0]
    assert len(angles) == 15
    ket = random_ket(enc.num_ancilla, seed=19)

    @cudaq.kernel
    def roundtrip(state: cudaq.State, angles: list[float]):
        q = cudaq.qvector(state)
        prepare(q, angles)
        unprepare(q, angles)

    state = np.asarray(cudaq.get_state(roundtrip, sim.state_from(ket), angles))
    assert np.allclose(state, ket, atol=1e-12)


def test_prepare_four_ancilla_matches_probabilities():
    weights = np.arange(1, 17, dtype=np.float64)
    probabilities = weights / np.sum(weights)
    angles = _prepare_angles(probabilities)
    assert len(angles) == 15

    @cudaq.kernel
    def prepare_four_ancilla(angles: list[float]):
        ancilla = cudaq.qvector(4)
        prepare(ancilla, angles)

    state = np.asarray(cudaq.get_state(prepare_four_ancilla, angles))
    leaf_indices = np.array(
        [int(f"{leaf_index:04b}"[::-1], 2) for leaf_index in range(16)])
    expected = np.zeros(16, dtype=np.float64)
    expected[leaf_indices] = np.sqrt(probabilities)
    assert np.allclose(state, expected, atol=1e-12)


def test_prepare_kernel_matches_padded_exact_probabilities():
    terms = {
        "II": 0.10,
        "XI": -0.20,
        "IZ": 0.30,
        "YX": -0.40,
        "ZZ": 0.50,
    }
    enc = PauliLCU(terms, backend="exact")
    assert enc.num_ancilla == 3

    probabilities = np.array(
        [abs(coefficient)
         for coefficient in terms.values()], dtype=np.float64) / enc.alpha
    probabilities = np.pad(probabilities,
                           (0, (1 << enc.num_ancilla) - probabilities.size))
    prepare_ancilla = enc.prepare_kernel()
    num_ancilla = enc.num_ancilla

    @cudaq.kernel
    def prepared():
        ancilla = cudaq.qvector(num_ancilla)
        prepare_ancilla(ancilla)

    state = np.asarray(cudaq.get_state(prepared), dtype=np.complex128)
    expected = np.zeros(1 << num_ancilla, dtype=np.complex128)
    for leaf_index, probability in enumerate(probabilities):
        reversed_index = int(f"{leaf_index:0{num_ancilla}b}"[::-1], 2)
        expected[reversed_index] = np.sqrt(probability)

    np.testing.assert_allclose(state, expected, atol=1e-12, rtol=1e-12)


def test_complex_coefficients_rejected_uniformly():
    # Every input form must reject complex coefficients with the same
    # ValueError (the mapping path previously raised an opaque TypeError).
    with pytest.raises(ValueError, match="complex"):
        PauliLCU({"XI": 0.5 + 0.3j})
    with pytest.raises(ValueError, match="complex"):
        PauliLCU([(0.5 + 0.3j, "XI")])
    with pytest.raises(ValueError, match="complex"):
        PauliLCU(0.5j * spin.x(0) * spin.y(1))
    # Real-valued complex input is fine everywhere.
    assert PauliLCU({"XI": 0.5 + 0j}).alpha == pytest.approx(0.5)


def test_prepare_angles_keep_tiny_sibling_terms():
    # Regression (review): the padding guard must fire only for exact-zero
    # (padded) subtrees. Two retained sibling terms whose combined
    # probability is below any threshold must still split 50/50 instead of
    # silently zeroing one branch.
    kept = [(2e-12, "XI"), (2e-12, "IX"), (1e6, "ZZ")]
    alpha = sum(abs(c) for c, _ in kept)
    probabilities = [abs(c) / alpha for c, _ in kept] + [0.0]
    angles = _prepare_angles(probabilities)
    assert angles[1] == pytest.approx(2.0 * math.asin(math.sqrt(0.5)))
    # Exact-zero padding subtrees still produce zero rotations.
    assert _prepare_angles([0.5, 0.5, 0.0, 0.0])[2] == 0.0


def test_empty_retained_terms_and_invalid_prepare_angle_size():
    with pytest.raises(ValueError, match="no retained terms"):
        PauliLCU({"X": 0.5}, coefficient_threshold=0.6)
    with pytest.raises(ValueError, match="no retained terms"):
        PauliLCU({"I": 0.5}, include_identity=False)
    with pytest.raises(ValueError, match="power of 2"):
        _prepare_angles([0.25, 0.5, 0.25])


def test_string_hamiltonian_rejected_with_type_error():
    # Parity with Trotter: string-like inputs must get the intended
    # TypeError, not a misleading unpack error from the pair branch.
    with pytest.raises(TypeError, match="SpinOperator"):
        PauliLCU("XZ")


@pytest.mark.parametrize(
    "hamiltonian,expected_width,expected_words",
    [(0.5 * spin.x(1), 2, ["IX"]),
     (0.3 * spin.x(0) + 0.2 * spin.z(3), 4, ["IIIZ", "XIII"])])
def test_spin_operator_infers_register_extent(hamiltonian, expected_width,
                                              expected_words):
    # Off-zero and gapped indices require max_degree + 1 qubits, not the
    # number of distinct indices. The width and padded words pin both facts.
    encoding = PauliLCU(hamiltonian)
    assert encoding.num_system == expected_width
    assert sorted(word for _, word in encoding.terms) == expected_words


def test_spin_operator_rejects_explicit_width_below_extent():
    # Validate the requested width before CUDA-Q renders a Pauli word, so the
    # error identifies the bad argument and the required register extent.
    with pytest.raises(ValueError, match=r"num_qubits=1.*extent 2"):
        PauliLCU(0.5 * spin.x(1), num_qubits=1)


def test_spin_operator_off_zero_dense_reference():
    # Beyond the width and padded words: the encoded block must be the
    # dense operator itself, not just constructible.
    off_zero = PauliLCU(0.5 * spin.x(1))
    assert off_zero.alpha == pytest.approx(0.5)
    ket = random_ket(2, seed=3)
    expected = dense_matrix([(0.5, "IX")], 2) @ ket / off_zero.alpha
    state = np.array(cudaq.get_state(off_zero.encode_kernel(),
                                     state_from(ket)))
    np.testing.assert_allclose(state[:4], expected, atol=1e-12)


def test_spin_operator_wider_explicit_width_pads():
    # An explicit num_qubits wider than the extent is legal padding.
    wider = PauliLCU(spin.x(0) + spin.z(3), num_qubits=6)
    assert wider.num_system == 6


def test_spin_operator_identity_term_extent():
    # Scalar/arithmetic identity terms act on NO degrees, and CUDA-Q's
    # max_degree raises on them rather than returning a sentinel; they
    # must not constrain or crash the register-extent inference. Note
    # spin.i(0) does NOT exercise this path (it explicitly targets degree
    # 0) -- only the scalar form does, which is exactly what the chemistry
    # bridge produces via scalar_offset.
    op = 0.5 * spin.x(1) + 0.25
    encoding = PauliLCU(op)
    assert encoding.num_system == 2
    assert {word for _, word in encoding.terms} == {"II", "IX"}


def test_composable_select_apply_walk_and_adjoint_walk():
    hamiltonian = {"X": 0.7, "Y": -0.4, "Z": 0.5, "I": -0.3}
    enc = PauliLCU(hamiltonian)
    assert enc.num_ancilla == 2

    angles, term_controls, term_ops, term_lengths, term_signs = enc.kernel_args
    n_anc = enc.num_ancilla
    ket = random_ket(1, seed=29)

    @cudaq.kernel
    def direct_select(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        select(ancilla, system, term_controls, term_ops, term_lengths,
               term_signs)

    selected = np.asarray(cudaq.get_state(direct_select, sim.state_from(ket)))
    expected_selected = dense_matrix([(1.0, "X")], 1) @ ket
    np.testing.assert_allclose(selected[:len(ket)],
                               expected_selected,
                               atol=1e-12,
                               rtol=1e-12)

    @cudaq.kernel
    def direct_apply(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        apply(ancilla, system, angles, term_controls, term_ops, term_lengths,
              term_signs)

    applied = np.asarray(cudaq.get_state(direct_apply, sim.state_from(ket)))
    expected_block = dense_matrix(enc.terms, 1) @ ket / enc.alpha
    np.testing.assert_allclose(applied[:len(ket)],
                               expected_block,
                               atol=1e-12,
                               rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(applied),
                               1.0,
                               atol=1e-12,
                               rtol=1e-12)
    initial = np.zeros_like(applied)
    initial[:len(ket)] = ket

    initial = np.zeros_like(applied)
    initial[:len(ket)] = ket

    @cudaq.kernel
    def direct_walk(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        walk(ancilla, system, angles, term_controls, term_ops, term_lengths,
             term_signs)

    walked = np.asarray(cudaq.get_state(direct_walk, sim.state_from(ket)))
    assert not np.allclose(walked, initial, atol=1e-12, rtol=1e-12)

    @cudaq.kernel
    def walk_roundtrip(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        walk(ancilla, system, angles, term_controls, term_ops, term_lengths,
             term_signs)
        adjoint_walk(ancilla, system, angles, term_controls, term_ops,
                     term_lengths, term_signs)

    roundtrip = np.asarray(cudaq.get_state(walk_roundtrip,
                                           sim.state_from(ket)))
    np.testing.assert_allclose(roundtrip, initial, atol=1e-12, rtol=1e-12)


def test_exact_controlled_pauli_lcu_factories():
    enc = PauliLCU(FOUR_TERMS)
    assert enc.backend == "exact"
    assert enc.num_ancilla == 2

    n_anc = enc.num_ancilla
    ket = random_ket(2, seed=43)
    apply_factory = enc.apply_kernel()
    controlled_apply_factory = enc.controlled_apply_kernel()

    @cudaq.kernel
    def apply_uncontrolled(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        apply_factory(ancilla, system)

    @cudaq.kernel
    def apply_controlled_off(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(1 + n_anc)
        controlled_apply_factory(control_and_ancilla, system)

    @cudaq.kernel
    def apply_controlled_on(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(1 + n_anc)
        x(control_and_ancilla[0])
        controlled_apply_factory(control_and_ancilla, system)

    uncontrolled = np.asarray(
        cudaq.get_state(apply_uncontrolled, sim.state_from(ket)))
    controlled_off = np.asarray(
        cudaq.get_state(apply_controlled_off, sim.state_from(ket)))
    expected_off = np.zeros((1 << n_anc, 2, len(ket)), dtype=np.complex128)
    expected_off[0, 0] = ket
    np.testing.assert_allclose(controlled_off.reshape(expected_off.shape),
                               expected_off,
                               atol=1e-12,
                               rtol=1e-12)

    controlled_on = np.asarray(
        cudaq.get_state(apply_controlled_on, sim.state_from(ket)))
    controlled_on = controlled_on.reshape(1 << n_anc, 2, len(ket))
    expected_on = np.zeros_like(controlled_on)
    expected_on[:, 1, :] = uncontrolled.reshape(1 << n_anc, len(ket))
    np.testing.assert_allclose(controlled_on,
                               expected_on,
                               atol=1e-12,
                               rtol=1e-12)

    controlled_walk = enc.controlled_walk_step_kernel()
    controlled_adjoint_walk = enc.controlled_adjoint_walk_step_kernel()

    @cudaq.kernel
    def controlled_walk_roundtrip(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(1 + n_anc)
        x(control_and_ancilla[0])
        controlled_walk(control_and_ancilla, system)
        controlled_adjoint_walk(control_and_ancilla, system)

    roundtrip = np.asarray(
        cudaq.get_state(controlled_walk_roundtrip, sim.state_from(ket)))
    expected_roundtrip = np.zeros((1 << n_anc, 2, len(ket)),
                                  dtype=np.complex128)
    expected_roundtrip[0, 1] = ket
    np.testing.assert_allclose(roundtrip.reshape(expected_roundtrip.shape),
                               expected_roundtrip,
                               atol=1e-12,
                               rtol=1e-12)


def test_exact_controlled_adjoint_walk_superposition_matches_reference():
    enc = PauliLCU({"X": 0.6, "Z": 0.4})
    assert enc.backend == "exact"
    n_anc = enc.num_ancilla
    ket = random_ket(1, seed=61).astype(np.complex128)
    prepare_ancilla = enc.prepare_kernel()
    controlled_adjoint = enc.controlled_adjoint_walk_step_kernel()
    adjoint = enc.adjoint_walk_step_kernel()

    @cudaq.kernel
    def controlled_adjoint_on_superposition(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(1 + n_anc)
        prepare_ancilla(control_and_ancilla.back(n_anc))
        h(control_and_ancilla[0])
        controlled_adjoint(control_and_ancilla, system)

    @cudaq.kernel
    def adjoint_reference(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        prepare_ancilla(ancilla)
        adjoint(ancilla, system)

    @cudaq.kernel
    def prepared_ancilla():
        ancilla = cudaq.qvector(n_anc)
        prepare_ancilla(ancilla)

    actual = np.asarray(
        cudaq.get_state(controlled_adjoint_on_superposition,
                        sim.state_from(ket)))
    actual = actual.reshape(1 << n_anc, 2, len(ket))
    expected_adjoint = np.asarray(
        cudaq.get_state(adjoint_reference, sim.state_from(ket)))
    expected = np.zeros((1 << n_anc, 2, len(ket)), dtype=np.complex128)
    prepare_state = np.asarray(cudaq.get_state(prepared_ancilla))
    expected[:,
             0, :] = (prepare_state[:, None] * ket[None, :] / math.sqrt(2.0))
    expected[:, 1, :] = (expected_adjoint.reshape(1 << n_anc, len(ket)) /
                         math.sqrt(2.0))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    prepared_block = np.einsum("a,acs->cs", prepare_state.conj(), actual)
    dense_hamiltonian = dense_matrix([(0.6, "X"), (0.4, "Z")], 1)
    expected_block = (-dense_hamiltonian @ ket / enc.alpha) / math.sqrt(2.0)
    np.testing.assert_allclose(prepared_block[1],
                               expected_block,
                               atol=1e-12,
                               rtol=1e-12)


@pytest.mark.parametrize(("backend", "options"), [("exact", {}),
                                                  ("alias", {
                                                      "mu": 1
                                                  })],
                         ids=["exact", "alias"])
def test_controlled_walk_superposition_matches_reference(backend, options):
    hamiltonian = {"X": 1.0, "Z": 1.0}
    enc = PauliLCU(hamiltonian, backend=backend, **options)
    n_anc = enc.num_ancilla
    ket = random_ket(1, seed=59).astype(np.complex128)
    prepare = enc.prepare_kernel()
    unprepare = enc.unprepare_kernel()
    controlled_walk = enc.controlled_walk_step_kernel()

    @cudaq.kernel
    def controlled_walk_superposition(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(1 + n_anc)
        h(control_and_ancilla[0])
        prepare(control_and_ancilla.back(n_anc))
        controlled_walk(control_and_ancilla, system)
        unprepare(control_and_ancilla.back(n_anc))

    actual = np.asarray(
        cudaq.get_state(controlled_walk_superposition, sim.state_from(ket)))
    actual = actual.reshape(1 << n_anc, 2, len(ket))

    scale = 1.0 / math.sqrt(2.0)
    expected = np.zeros_like(actual)
    expected[0, 0] = ket * scale
    walked = np.asarray(
        cudaq.get_state(enc.walk_kernel(power=1), sim.state_from(ket)))
    expected[:, 1, :] = walked.reshape(1 << n_anc, len(ket)) * scale
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    dense_hamiltonian = dense_matrix([(1.0, "X"), (1.0, "Z")], 1)
    expected_block = (-dense_hamiltonian @ ket / enc.alpha) * scale
    np.testing.assert_allclose(actual[0, 1],
                               expected_block,
                               atol=1e-12,
                               rtol=1e-12)


def test_exact_controlled_walk_dirty_ancilla_superposition():
    enc = PauliLCU(FOUR_TERMS)
    assert enc.backend == "exact"
    assert enc.num_ancilla == 2

    n_anc = enc.num_ancilla
    ket = random_ket(2, seed=61).astype(np.complex128)
    walk_step = enc.walk_step_kernel()
    controlled_walk = enc.controlled_walk_step_kernel()

    @cudaq.kernel
    def dirty_initial(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        x(ancilla[0])

    @cudaq.kernel
    def dirty_walk(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_anc)
        x(ancilla[0])
        walk_step(ancilla, system)

    @cudaq.kernel
    def controlled_dirty_walk(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(1 + n_anc)
        h(control_and_ancilla[0])
        x(control_and_ancilla[1])
        controlled_walk(control_and_ancilla, system)

    dirty_state = np.asarray(
        cudaq.get_state(dirty_initial,
                        sim.state_from(ket))).reshape(1 << n_anc, len(ket))
    walked_dirty_state = np.asarray(
        cudaq.get_state(dirty_walk,
                        sim.state_from(ket))).reshape(1 << n_anc, len(ket))
    actual = np.asarray(
        cudaq.get_state(controlled_dirty_walk, sim.state_from(ket)))
    actual = actual.reshape(1 << n_anc, 2, len(ket))

    expected = np.zeros_like(actual)
    scale = 1.0 / math.sqrt(2.0)
    expected[:, 0, :] = dirty_state * scale
    expected[:, 1, :] = walked_dirty_state * scale
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_controlled_sequence_and_adjoint_paths():
    enc = PauliLCU({"X": 0.6, "Z": 0.4})
    sequence = PhaseSequence([0.23, -0.41, 0.11], walk_directions=[1, 0])
    phases = sequence.projector_phases
    directions = list(sequence.walk_directions)
    angles, controls, ops, lengths, signs = enc.kernel_args
    n_anc = enc.num_ancilla
    ket = random_ket(1, seed=37)

    @cudaq.kernel
    def composed_controlled_sequence(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_signal = cudaq.qvector(1 + n_anc)
        x(control_and_signal[0])
        apply_controlled_phase_sequence(control_and_signal, system, phases,
                                        directions, angles, controls, ops,
                                        lengths, signs)

    sequence_reference = np.asarray(
        cudaq.get_state(
            QSVT(enc).controlled_kernel(sequence, control_state=1),
            sim.state_from(ket)))
    composed_state = np.asarray(
        cudaq.get_state(composed_controlled_sequence, sim.state_from(ket)))
    np.testing.assert_allclose(composed_state, sequence_reference, atol=1e-12)

    @cudaq.kernel
    def composed_controlled_sequence_off(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_signal = cudaq.qvector(1 + n_anc)
        apply_controlled_phase_sequence(control_and_signal, system, phases,
                                        directions, angles, controls, ops,
                                        lengths, signs)

    off_state = np.asarray(
        cudaq.get_state(composed_controlled_sequence_off, sim.state_from(ket)))
    expected_off = np.zeros((1 << n_anc, 2, len(ket)), dtype=np.complex128)
    expected_off[0, 0] = ket
    np.testing.assert_allclose(off_state, expected_off.reshape(-1), atol=1e-12)

    qsvt = QSVT(enc)
    sequence_reference = np.asarray(
        cudaq.get_state(qsvt.kernel(sequence), sim.state_from(ket)))
    for control_state in (0, 1):
        controlled_state = np.asarray(
            cudaq.get_state(
                qsvt.controlled_kernel(sequence, control_state=control_state),
                sim.state_from(ket)))
        controlled_state = controlled_state.reshape(1 << n_anc, 2, len(ket))
        expected_controlled_state = np.zeros_like(controlled_state)
        if control_state == 1:
            expected_controlled_state[:, 1] = sequence_reference.reshape(
                1 << n_anc, len(ket))
        else:
            expected_controlled_state[0, 0] = ket
        np.testing.assert_allclose(controlled_state,
                                   expected_controlled_state,
                                   atol=1e-12)

    def control_on_input(num_ancilla):
        expected = np.zeros((1 << num_ancilla, 2, len(ket)),
                            dtype=np.complex128)
        expected[0, 1] = ket
        return expected.reshape(-1)

    alias_encoding = PauliLCU({"X": 1.0, "Z": 1.0}, backend="alias", mu=1)
    alias_controlled_walk = alias_encoding.controlled_walk_step_kernel()
    alias_controlled_adjoint = (
        alias_encoding.controlled_adjoint_walk_step_kernel())
    alias_n_anc = alias_encoding.num_ancilla

    @cudaq.kernel
    def alias_walk_roundtrip(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(1 + alias_n_anc)
        x(control_and_ancilla[0])
        alias_controlled_walk(control_and_ancilla, system)
        alias_controlled_adjoint(control_and_ancilla, system)

    alias_state = np.asarray(
        cudaq.get_state(alias_walk_roundtrip, sim.state_from(ket)))
    np.testing.assert_allclose(alias_state,
                               control_on_input(alias_n_anc),
                               atol=1e-12)

    sum_encoding = SumEncoding([(1.0, PauliLCU({"X": 1.0})),
                                (1.0, PauliLCU({"Z": 1.0}))])
    sum_controlled_walk = sum_encoding.controlled_walk_step_kernel()
    sum_controlled_adjoint = sum_encoding.controlled_adjoint_walk_step_kernel()
    sum_n_anc = sum_encoding.num_ancilla

    @cudaq.kernel
    def sum_walk_roundtrip(state: cudaq.State):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(1 + sum_n_anc)
        x(control_and_ancilla[0])
        sum_controlled_walk(control_and_ancilla, system)
        sum_controlled_adjoint(control_and_ancilla, system)

    sum_state = np.asarray(
        cudaq.get_state(sum_walk_roundtrip, sim.state_from(ket)))
    np.testing.assert_allclose(sum_state,
                               control_on_input(sum_n_anc),
                               atol=1e-12)


def test_encode_factories_and_mixed_direction_phase_sequence():
    enc = PauliLCU(FOUR_TERMS)
    assert enc.num_system == 2
    assert enc.num_ancilla == 2

    ket = random_ket(2, seed=47).astype(np.complex128)
    hamiltonian = dense_matrix([(coefficient, word)
                                for word, coefficient in FOUR_TERMS.items()],
                               2)
    expected_block = hamiltonian @ ket / enc.alpha

    encoded = np.asarray(cudaq.get_state(enc.encode_kernel(),
                                         sim.state_from(ket)),
                         dtype=np.complex128)
    np.testing.assert_allclose(encoded[:len(ket)],
                               expected_block,
                               atol=1e-12,
                               rtol=1e-12)

    apply_factory = enc.apply_kernel()
    n_ancilla = enc.num_ancilla

    @cudaq.kernel
    def apply_from_state(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(n_ancilla)
        apply_factory(ancilla, system)

    applied = np.asarray(cudaq.get_state(apply_from_state,
                                         sim.state_from(ket)),
                         dtype=np.complex128)
    np.testing.assert_allclose(applied, encoded, atol=1e-12, rtol=1e-12)

    @cudaq.kernel
    def state_prep(qubits: cudaq.qview):
        h(qubits[0])
        x(qubits[1])

    prepared_ket = np.zeros(4, dtype=np.complex128)
    prepared_ket[2:4] = 1.0 / math.sqrt(2.0)
    prepared = np.asarray(cudaq.get_state(
        enc.encode_kernel(state_prep=state_prep)),
                          dtype=np.complex128)
    prepared_expected = hamiltonian @ prepared_ket / enc.alpha
    np.testing.assert_allclose(prepared[:len(prepared_ket)],
                               prepared_expected,
                               atol=1e-12,
                               rtol=1e-12)

    phase_values = [0.23, -0.41, 0.11]
    sequence = PhaseSequence(phase_values, walk_directions=[1, 0])
    projector_phases = sequence.projector_phases
    directions = list(sequence.walk_directions)
    angles, controls, ops, lengths, signs = enc.kernel_args

    @cudaq.kernel
    def composed_phase_sequence(state: cudaq.State):
        system = cudaq.qvector(state)
        signal = cudaq.qvector(n_ancilla)
        apply_phase_sequence(signal, system, projector_phases, directions,
                             angles, controls, ops, lengths, signs)

    phase_ket = random_ket(2, seed=53).astype(np.complex128)
    composed = np.asarray(cudaq.get_state(composed_phase_sequence,
                                          sim.state_from(phase_ket)),
                          dtype=np.complex128)
    qsvt = np.asarray(cudaq.get_state(
        QSVT(enc).kernel(sequence), sim.state_from(phase_ket)),
                      dtype=np.complex128)
    np.testing.assert_allclose(composed, qsvt, atol=1e-12, rtol=1e-12)


def test_exact_adjoint_walk_step_restores_entangled_register_state():
    enc = PauliLCU(FOUR_TERMS, backend="exact")
    num_system = enc.num_system
    num_ancilla = enc.num_ancilla
    initial = random_ket(num_system + num_ancilla, seed=67)
    walk_step = enc.walk_step_kernel()
    adjoint_walk_step = enc.adjoint_walk_step_kernel()

    @cudaq.kernel
    def roundtrip(state: cudaq.State):
        registers = cudaq.qvector(state)
        ancilla = registers.front(num_ancilla)
        system = registers[num_ancilla:num_ancilla + num_system]
        walk_step(ancilla, system)
        adjoint_walk_step(ancilla, system)

    actual = np.asarray(cudaq.get_state(roundtrip, sim.state_from(initial)),
                        dtype=np.complex128)
    np.testing.assert_allclose(actual, initial, atol=1e-12, rtol=1e-12)
