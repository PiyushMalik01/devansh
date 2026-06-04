"""Model wrappers.

``MockModel`` is torch-free and imported eagerly. ``ImageNetModel`` pulls in
torch/torchvision, so it is imported lazily (PEP 562 module ``__getattr__``):
``from scattercamo.models import ImageNetModel`` still works, but merely
importing this package -- e.g. to use ``MockModel`` -- does not require torch.
"""

from .mock import MockModel

__all__ = ["ImageNetModel", "MockModel"]


def __getattr__(name):
    if name == "ImageNetModel":
        from .imagenet import ImageNetModel
        return ImageNetModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
