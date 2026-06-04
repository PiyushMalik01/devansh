"""Batch experiment runner with checkpoint/resume.

Runs any attack over a list of (image, true_label) pairs, computes per-image
metrics, and checkpoints after every image so a preempted cloud run resumes by
skipping completed indices. Aggregates into headline statistics.

The runner is model-agnostic: it receives factories, so the same code drives
ScatterCamo, the baselines, and tests against a mock model.
"""

import json
import os

from scattercamo import metrics
from scattercamo.runner.result import normalize_result


class BatchRunner:
    def __init__(self, attack_factory, loss_factory, out_dir,
                 name="attack", checkpoint_every=1, compute_ssim=True):
        """
        Args:
            attack_factory: callable(x) -> attack instance with ``optimise(loss)``.
            loss_factory:   callable(x, true_label) -> loss function.
            out_dir:        directory for the checkpoint file.
            name:           experiment name (checkpoint file is ``<name>.json``).
        """
        self.attack_factory = attack_factory
        self.loss_factory = loss_factory
        self.out_dir = out_dir
        self.name = name
        self.checkpoint_every = checkpoint_every
        self.compute_ssim = compute_ssim
        os.makedirs(out_dir, exist_ok=True)
        self.ckpt_path = os.path.join(out_dir, f"{name}.json")

    def _load_checkpoint(self):
        if os.path.exists(self.ckpt_path):
            with open(self.ckpt_path) as fh:
                return json.load(fh)
        return {"records": {}}

    def _save_checkpoint(self, state):
        tmp = self.ckpt_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh)
        os.replace(tmp, self.ckpt_path)   # atomic: survives mid-write preemption

    def _evaluate_one(self, x, label):
        loss = self.loss_factory(x, label)
        result = normalize_result(self.attack_factory(x).optimise(loss))
        record = {"success": bool(result.success), "queries": int(result.queries)}
        if result.success and result.adv_image is not None:
            adv = result.adv_image
            record["l0"] = metrics.l0(adv, x)
            record["l2"] = metrics.l2(adv, x)
            record["psnr"] = metrics.psnr(adv, x)
            if self.compute_ssim:
                record["ssim"] = metrics.ssim(adv, x)
        return record

    def run(self, dataset):
        """dataset: iterable of (image_array, true_label). Returns aggregate stats."""
        state = self._load_checkpoint()
        records = state["records"]
        dataset = list(dataset)

        for i, (x, label) in enumerate(dataset):
            if str(i) in records:
                continue                                  # already done -> resume
            records[str(i)] = self._evaluate_one(x, label)
            if (i + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(state)
        self._save_checkpoint(state)

        ordered = [records[str(i)] for i in range(len(dataset))]
        summary = metrics.summarize(ordered)
        summary["records"] = ordered
        return summary
