"""
Reference test: the current code must reproduce the reference results
stored in tests/data/reference_baseline.json.

Run from the repository root:

    pytest tests/test_reference.py -v

Integers, labels and strings must match exactly; floating point numbers must
match up to a relative tolerance of 1e-9 (tiny differences in the last digits
can appear on a different machine or numpy version).
"""
import json
import math
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_cases import CASES  # noqa: E402

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'reference_baseline.json')
RTOL = 1e-9


def assert_same(expected, actual, path='result'):
    """Compare two JSON-like objects and report exactly where they differ."""
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, (int, float)) and isinstance(expected, (int, float)), \
            f'{path}: expected {expected!r}, got {actual!r}'
        if math.isnan(expected) and math.isnan(actual):
            return
        assert math.isclose(expected, actual, rel_tol=RTOL, abs_tol=1e-12), \
            f'{path}: expected {expected!r}, got {actual!r}'
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f'{path}: expected a dict, got {type(actual).__name__}'
        assert expected.keys() == actual.keys(), \
            f'{path}: different keys, expected {sorted(expected)}, got {sorted(actual)}'
        for k in expected:
            assert_same(expected[k], actual[k], f'{path}[{k!r}]')
    elif isinstance(expected, list):
        assert isinstance(actual, list), f'{path}: expected a list, got {type(actual).__name__}'
        assert len(expected) == len(actual), \
            f'{path}: expected length {len(expected)}, got {len(actual)}'
        for i, (e, a) in enumerate(zip(expected, actual)):
            assert_same(e, a, f'{path}[{i}]')
    else:
        assert expected == actual, f'{path}: expected {expected!r}, got {actual!r}'


@pytest.fixture(scope='module')
def baseline():
    if not os.path.exists(BASELINE):
        pytest.skip('No baseline found: run python tests/generate_reference_baseline.py first')
    with open(BASELINE) as f:
        return json.load(f)


@pytest.mark.parametrize('name', list(CASES))
def test_reference(name, baseline):
    warnings.filterwarnings('ignore')
    assert name in baseline, f'Case {name!r} missing from baseline: regenerate it'
    actual = json.loads(json.dumps(CASES[name]()))  # same conversion as the baseline
    assert_same(baseline[name], actual, path=name)
