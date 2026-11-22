import pytest

from tests.reference.p1_pauli_frame import PauliFrame


def test_p1_oracle_clifford_primitives():
    frame = PauliFrame(3)
    frame.error(0, "Y")
    frame.error(1, "Z")
    assert (frame.x, frame.z) == ([1, 0, 0], [1, 1, 0])
    frame.clifford("H", (1,))
    assert (frame.x, frame.z) == ([1, 1, 0], [1, 0, 0])
    frame.clifford("S", (1,))
    assert (frame.x, frame.z) == ([1, 1, 0], [1, 1, 0])
    frame.clifford("CX", (0, 1))
    assert (frame.x, frame.z) == ([1, 0, 0], [0, 1, 0])

    for gate in ("I", "X", "Y", "Z"):
        frame.clifford(gate, (0, 1, 2))
    assert (frame.x, frame.z) == ([1, 0, 0], [0, 1, 0])
    with pytest.raises(ValueError, match="unsupported Clifford"):
        frame.clifford("SWAP", (0, 1))


def test_p1_oracle_measurement_reset_and_pure_record_flip():
    frame = PauliFrame(2)
    frame.error(0, "Y")
    frame.error(1, "Z")
    assert frame.measure_z(0, reset=False, record_flip=True) == 0
    assert (frame.x, frame.z) == ([1, 0], [0, 1])
    assert frame.measure_z(0, reset=True) == 1
    assert frame.measure_z(1, reset=True, record_flip=True) == 1
    assert (frame.x, frame.z) == ([0, 0], [0, 0])
    frame.error(1, "Y")
    frame.reset_z(1)
    assert (frame.x, frame.z) == ([0, 0], [0, 0])
