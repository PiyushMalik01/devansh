# Credits & Attribution

This repository builds directly on two works by **Phoenix Neale Williams** and
**Ke Li** (Department of Computer Science, University of Exeter). All credit for
the original methods and reference implementations belongs to them.

## Vendored reference code

### CamoPatch (NeurIPS 2023)
- Folder: [`CamoPatch/`](CamoPatch/)
- Source repo: https://github.com/phoenixwilliams/CamoPatch
- Paper: *CamoPatch: An Evolutionary Strategy for Generating Camouflaged
  Adversarial Patches*, NeurIPS 2023.

```bibtex
@inproceedings{williams2023camopatch,
  title={CamoPatch: An Evolutionary Strategy for Generating Camouflaged Adversarial Patches},
  author={Williams, Phoenix Neale and Li, Ke},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

### SA-MOO — Black-Box Sparse Adversarial Attack (CVPR 2023)
- Folder: [`Black-Box-Sparse-Adversarial-Attack/`](Black-Box-Sparse-Adversarial-Attack/)
- Source repo: https://github.com/phoenixwilliams/Black-Box-Sparse-Adversarial-Attack-via-Multi-Objective-Optimisation
- Paper: *Black-Box Sparse Adversarial Attack via Multi-Objective Optimisation*,
  CVPR 2023, pp. 12291–12301.

```bibtex
@inproceedings{williams2023black,
  title={Black-Box Sparse Adversarial Attack via Multi-Objective Optimisation},
  author={Williams, Phoenix Neale and Li, Ke},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={12291--12301},
  year={2023}
}
```

The two reference folders contain the original authors' public code (their git
histories were removed so this is a single repository). Their original README
files and citation blocks are preserved in place. The bundled paper PDFs are the
authors'/publishers' copyrighted material, included here for research convenience.

## New work in this repository

[`ScatterCamo/`](ScatterCamo/) — *multi-objective scattered camouflaged
adversarial patches* — is new work developed here. It hybridizes the two methods
above: CamoPatch's camouflaged semi-transparent shape representation combined
with SA-MOO's NSGA-II search and prioritized domination relation. See
[`ScatterCamo/docs/design.md`](ScatterCamo/docs/design.md) for the rationale.

## Licensing note

The upstream repositories did not include explicit license files. The vendored
code remains the intellectual property of the original authors and is included
here under fair-use for research and educational purposes. If you intend to
reuse or redistribute any of this material, please contact the original authors
and cite the papers above.
