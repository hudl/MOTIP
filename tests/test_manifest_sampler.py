"""SAMPLE_MANIFEST: ManifestSampler window legality, weighting, determinism, and the train.py hook."""
import json
import os
import sys
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.manifest_sampler import ManifestSampler, resolve_manifest_path  # noqa: E402


def _source(lengths=None, illegal=None):
    lengths = lengths or {"a": 100, "b": 50}
    illegal = illegal or {}
    infos = {"Hockey": {"train": {s: {"length": n, "is_static": False} for s, n in lengths.items()}}}
    legal = {"Hockey": {"train": {}}}
    for s, n in lengths.items():
        t = torch.ones(n, dtype=torch.bool)
        for f in illegal.get(s, []):
            t[f] = False
        legal["Hockey"]["train"][s] = t
    return SimpleNamespace(sequence_infos=infos, ann_is_legals=legal)


def _m(windows, n=None):
    return {"version": 1, "dataset": "Hockey", "split": "train", "samples_per_epoch": n, "windows": windows}


def test_frames_follow_start_step_length():
    s = ManifestSampler(_source(), _m([{"seq": "a", "start": 10, "length": 5, "step": 4, "tag": "e"}], n=1))
    s.prepare_for_epoch(0)
    (info,) = list(s)
    assert info == {"dataset": "Hockey", "split": "train", "sequence": "a", "frame_idxs": [10, 14, 18, 22, 26]}


def test_drops_outside_illegal_missing_and_zero_weight():
    ws = [
        {"seq": "a", "start": 40, "length": 60, "step": 1, "tag": "ok"},     # 40..99: fits, legal
        {"seq": "b", "start": 0, "length": 20, "step": 4, "tag": "out"},     # last frame 76 >= 50
        {"seq": "a", "start": 30, "length": 10, "step": 1, "tag": "ill"},    # covers illegal frame 35
        {"seq": "zz", "start": 0, "length": 5, "tag": "gone"},
        {"seq": "a", "start": 0, "length": 5, "weight": 0, "tag": "w0"},
        {"seq": "a", "start": -1, "length": 5, "tag": "neg"},
    ]
    s = ManifestSampler(_source(illegal={"a": [35]}), _m(ws))
    assert [w["tag"] for w in s.windows] == ["ok"]
    assert s.dropped == {"out:outside": 1, "ill:illegal": 1, "gone:missing_seq": 1, "w0:zero_weight": 1, "neg:outside": 1}


def test_illegal_frame_only_drops_windows_touching_it():
    ws = [{"seq": "a", "start": 0, "length": 30, "tag": "x"}, {"seq": "a", "start": 40, "length": 30, "tag": "x"}]
    s = ManifestSampler(_source(illegal={"a": [35]}), _m(ws))   # 35 lies between the two windows
    assert [w["frame_idxs"][0] for w in s.windows] == [0, 40] and s.dropped == {}
    s2 = ManifestSampler(_source(illegal={"a": [20]}), _m(ws))
    assert [w["frame_idxs"][0] for w in s2.windows] == [40] and s2.dropped == {"x:illegal": 1}


def test_no_legal_window_raises():
    with pytest.raises(AssertionError):
        ManifestSampler(_source(), _m([{"seq": "b", "start": 45, "length": 10}]))


def test_weights_set_the_mix():
    ws = [{"seq": "a", "start": i, "length": 10, "weight": 3.0, "tag": "event"} for i in range(40)]
    ws += [{"seq": "a", "start": i, "length": 10, "weight": 1.0, "tag": "uniform"} for i in range(40, 80)]
    s = ManifestSampler(_source(), _m(ws, n=20000))   # > candidates: with replacement
    s.prepare_for_epoch(0)
    c = Counter(s.last_epoch_stats["tags"])
    assert s.last_epoch_stats["replacement"] is True
    assert abs(c["event"] / 20000 - 0.75) < 0.02


def test_without_replacement_when_enough_windows():
    ws = [{"seq": "a", "start": i, "length": 10} for i in range(80)]
    s = ManifestSampler(_source(), _m(ws, n=50))
    s.prepare_for_epoch(3)
    starts = [i["frame_idxs"][0] for i in s]
    assert len(starts) == 50 == len(set(starts)) and s.last_epoch_stats["replacement"] is False


def test_deterministic_per_epoch_and_different_across_epochs():
    ws = [{"seq": "a", "start": i, "length": 10} for i in range(80)]
    a, b = ManifestSampler(_source(), _m(ws, n=30), seed=7), ManifestSampler(_source(), _m(ws, n=30), seed=7)
    a.prepare_for_epoch(2), b.prepare_for_epoch(2)
    assert list(a) == list(b)
    e2 = list(a)
    a.prepare_for_epoch(3)
    assert list(a) != e2


def test_stats_keep_naive_sampler_keys_and_count_window_frames():
    ws = [{"seq": "a", "start": 0, "length": 10, "tag": "u"}, {"seq": "a", "start": 0, "length": 20, "step": 2, "tag": "e"}]
    s = ManifestSampler(_source(), _m(ws, n=2))
    s.prepare_for_epoch(0)
    st = s.last_epoch_stats
    for k in ("samples", "sample_length", "sequences", "short", "short_kept", "no_window"):
        assert k in st
    assert st["window_frames"] == {"u": 10, "e": 20} and s.variable_length and st["sample_length"] == "10-20"


def test_samples_per_epoch_override_and_default():
    ws = [{"seq": "a", "start": i, "length": 10} for i in range(5)]
    assert ManifestSampler(_source(), _m(ws)).samples_per_epoch == 5
    assert ManifestSampler(_source(), _m(ws, n=3), samples_per_epoch=4).samples_per_epoch == 4


def test_reads_a_file_and_resolves_relative_to_data_root(tmp_path):
    p = tmp_path / "manifests" / "m.json"
    p.parent.mkdir()
    p.write_text(json.dumps(_m([{"seq": "a", "start": 0, "length": 3}])))
    assert resolve_manifest_path("manifests/m.json", str(tmp_path)) == str(p)
    assert len(ManifestSampler(_source(), str(p)).windows) == 1


def test_train_py_uses_the_manifest_sampler_when_configured():
    src = open(os.path.join(os.path.dirname(__file__), "..", "train.py")).read()
    assert 'config.get("SAMPLE_MANIFEST")' in src and "ManifestSampler(" in src and "NaiveSampler(" in src
