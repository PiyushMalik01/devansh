"""Tests for run_attack's --config merge logic (NumPy only, no torch).

Run from the repo root:
    python tests/test_config.py

Proves the precedence: built-in defaults < --config YAML < explicit CLI flags,
that the config file can supply the image, that unknown keys are ignored, and
that a missing image fails cleanly.
"""

import os
import sys
import argparse
import tempfile
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_attack import resolve_config, DEFAULTS


def _write(d, text):
    path = os.path.join(d, "c.yaml")
    with open(path, "w") as f:
        f.write(text)
    return path


def test_defaults_only():
    args = resolve_config(argparse.ArgumentParser(), {"image": "x.png"})
    assert args.image == "x.png"
    assert args.M == DEFAULTS["M"] == 10
    assert args.perceptual is False and args.mock is False
    print("  [ok] built-in defaults fill in when nothing else is given")


def test_config_fills_values():
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "image: a.png\nM: 7\nperceptual: true\n")
        args = resolve_config(argparse.ArgumentParser(), {"config": path})
    assert args.image == "a.png" and args.M == 7 and args.perceptual is True
    print("  [ok] config file provides values (including the image)")


def test_cli_overrides_config():
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "image: a.png\nM: 7\n")
        args = resolve_config(argparse.ArgumentParser(),
                              {"config": path, "M": 3, "image": "b.png"})
    assert args.M == 3 and args.image == "b.png"
    print("  [ok] explicit CLI flags override the config file")


def test_unknown_keys_ignored():
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "image: a.png\nn_images: 1000\nM: 5\n")
        args = resolve_config(argparse.ArgumentParser(), {"config": path})
    assert args.M == 5 and not hasattr(args, "n_images")
    print("  [ok] unknown config keys (e.g. n_images) are ignored")


def test_missing_image_errors():
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        try:
            resolve_config(argparse.ArgumentParser(), {})   # no image anywhere
        except SystemExit:
            print("  [ok] missing image raises a clean error")
            return
    raise AssertionError("expected SystemExit when no image is provided")


if __name__ == "__main__":
    print("Running config tests...")
    test_defaults_only()
    test_config_fills_values()
    test_cli_overrides_config()
    test_unknown_keys_ignored()
    test_missing_image_errors()
    print("ALL TESTS PASSED")
