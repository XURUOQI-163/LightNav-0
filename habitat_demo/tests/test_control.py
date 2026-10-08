import math

import pytest
from habitat_web.control import command_for_path, parse_waypoints


def test_omnivla_units_and_heading():
    assert parse_waypoints([[2, 1, 0, 1]], "omnivla")[0] == pytest.approx((0.2, 0.1, math.pi / 2))


def test_lightnav_nested_protocol():
    assert parse_waypoints([{"actions": [[0.2, -0.1, 0.3]]}], "lightnav") == ((0.2, -0.1, 0.3),)


@pytest.mark.parametrize("value", [[[1, float("nan"), 0]], [[True, 0, 0]], [[1, 2]]])
def test_invalid_lightnav_waypoints_rejected(value):
    with pytest.raises(ValueError):
        parse_waypoints(value, "lightnav")


def test_command_preserves_curvature_when_limited():
    v, w = command_for_path(((1.0, 1.0, 0.0),), 0.333, 0.15, 0.5)
    assert v == pytest.approx(0.15)
    assert w / v == pytest.approx(math.pi / 4)


def test_left_turn_and_zero_position_rotation():
    assert command_for_path(((0.0, 0.0, 0.3),), 0.3, 0.15, 0.5) == (0.0, 0.5)
    assert command_for_path((), 0.3, 0.15, 0.5) == (0.0, 0.0)
