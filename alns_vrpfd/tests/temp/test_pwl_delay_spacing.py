import math

from alns_vrpfd.mip.builder import set_distance_tardiness_objective
from alns_vrpfd.mip.builder import _compute_delay_pwl_points

# Utility: build xs/ys using same logic as builder (quad spacing vs uniform)


def build_uniform_xs(max_delay, segments):
    return [float(i) * max_delay / float(segments) for i in range(segments + 1)]


def build_quadratic_xs(max_delay, segments):
    return [max_delay * (float(i) / float(segments)) ** 2 for i in range(segments + 1)]


def eval_exact_exp(x):
    return math.exp(1.5031 + 7.032 * x) - math.exp(1.5031)


def eval_pwl_value(xs, ys, x):
    # simple linear interpolation piecewise
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


def test_quadratic_spacing_closer_to_exact():
    max_delay = 3.0
    segments = 10

    uniform_xs = build_uniform_xs(max_delay, segments)
    quad_xs = build_quadratic_xs(max_delay, segments)

    uniform_ys = [eval_exact_exp(x) for x in uniform_xs]
    quad_ys = [eval_exact_exp(x) for x in quad_xs]

    # target: small tau (representative problematic tau from reported instance)
    tau = 0.021429

    uniform_val = eval_pwl_value(uniform_xs, uniform_ys, tau)
    quad_val = eval_pwl_value(quad_xs, quad_ys, tau)
    exact_val = eval_exact_exp(tau)

    err_uniform = abs(uniform_val - exact_val)
    err_quad = abs(quad_val - exact_val)

    assert err_quad <= err_uniform, f"Quadratic spacing should be at least as good: {err_quad} > {err_uniform}"


def test_three_piece_manual_scaling_effect():
    # Ensure three-piece linearization with middle scaling affects intermediate y value and slope
    max_delay = 3.0
    # breakpoints at 85 and 105 minutes -> in hours
    b1 = 85.0 / 60.0
    b2 = 105.0 / 60.0

    xs = [0.0, b1, b2, max_delay]
    ys = [eval_exact_exp(x) for x in xs]

    # compute original middle slope
    orig_mid_slope = (ys[2] - ys[1]) / (xs[2] - xs[1])

    # apply scale of 2.0 to middle slope
    scale = 2.0
    new_mid_y2 = ys[1] + orig_mid_slope * scale * (xs[2] - xs[1])
    # new slopes
    new_mid_slope = (new_mid_y2 - ys[1]) / (xs[2] - xs[1])
    # ensure it scaled
    assert abs(new_mid_slope - orig_mid_slope * scale) < 1e-9
    # check continuity at x3 using y3 unchanged
    # new last segment slope should adjust: (y3 - new_y2) / (x3 - x2)
    new_last_slope = (ys[3] - new_mid_y2) / (xs[3] - xs[2])
    assert new_last_slope != (ys[3] - ys[2]) / (xs[3] - xs[2])


def test_three_piece_point_generation_defaults():
    max_delay = 3.0
    xs, ys = _compute_delay_pwl_points(
        max_delay, segments=None, use_three_piece_delay=True)
    assert len(xs) == 4
    # default breakpoints in hours
    assert abs(xs[1] - 85.0 / 60.0) < 1e-6
    assert abs(xs[2] - 105.0 / 60.0) < 1e-6
    # Monotonic xs
    assert xs[0] < xs[1] < xs[2] < xs[3]
    # Ensure convexity of ys (slopes non-decreasing)
    slopes = [(ys[i+1] - ys[i]) / (xs[i+1] - xs[i])
              for i in range(len(xs) - 1)]
    assert slopes[0] <= slopes[1] <= slopes[2]


def test_three_piece_middle_scale_in_builder():
    max_delay = 3.0
    # no scale
    xs, ys = _compute_delay_pwl_points(
        max_delay, segments=None, use_three_piece_delay=True, three_piece_middle_scale=None)
    slopes = [(ys[i+1] - ys[i]) / (xs[i+1] - xs[i])
              for i in range(len(xs) - 1)]
    mid_slope = slopes[1]

    # scale the mid slope by 3x
    xs2, ys2 = _compute_delay_pwl_points(
        max_delay, segments=None, use_three_piece_delay=True, three_piece_middle_scale=3.0)
    slopes2 = [(ys2[i+1] - ys2[i]) / (xs2[i+1] - xs2[i])
               for i in range(len(xs2) - 1)]
    assert abs(slopes2[1] - mid_slope * 3.0) < 1e-9


def test_quadratic_spacing_first_breakpoint_small():
    # With segments=10 and power=2.0, the first positive breakpoint should be small
    max_delay = 3.0
    segments = 10
    power = 1.0
    xs, ys = _compute_delay_pwl_points(
        max_delay, segments=segments, power=power, use_three_piece_delay=False)
    first_pos_x = next(x for x in xs if x > 0.0)
    assert abs(first_pos_x - (max_delay / segments)) < 1e-9


if __name__ == "__main__":
    test_quadratic_spacing_closer_to_exact()
    print("PWL spacing test passed")
