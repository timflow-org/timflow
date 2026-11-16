import inspect
from functools import wraps
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def store_input(
    cls: type[T],
    *,
    exclude: frozenset[str] = frozenset({"self", "model", "ml"}),
    store_attr: str = "_init_args",
) -> type[T]:
    """Class decorator that captures normalized ``__init__`` arguments on the instance.

    After instantiation, ``self.<store_attr>`` is a dict mapping each parameter name
    from the original ``__init__`` signature to the value it received, with defaults
    applied. Names in *exclude* (parent-model references) are omitted.

    When applied to both a parent and child class, only the outermost wrapper 
    (e.g. Well) runs and sets the attribute, inner wrappers (e.g. Element) skip.

    Parameters
    ----------
    cls : type
        The class to decorate.
    exclude : frozenset of str, optional
        Parameter names to exclude from capture (default: ``self``, ``model``, ``ml``).
    store_attr : str, optional
        Instance attribute name for the captured dict (default: ``"_init_args"``).

    Returns
    -------
    type
        The same class with a wrapped ``__init__``.

    Examples
    --------
    >>> @store_input
    ... class Well:
    ...     def __init__(self, model, x: float, y: float, Q: float = 1.0):
    ...         self.x, self.y, self.Q = x, y, Q

    >>> w = Well(model_instance, 1.0, 2.0)
    >>> w._init_args
    {'x': 1.0, 'y': 2.0, 'Q': 1.0}   # model excluded by name, defaults applied
    """
    original_init = cls.__init__
    sig = inspect.signature(original_init)

    @wraps(original_init)
    def new_init(self, *args, **kwargs) -> None:
        # Only the outermost (most-derived) wrapper captures; inner ones skip.
        if not hasattr(self, store_attr):
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()

            captured: dict = {}
            for name, value in bound.arguments.items():
                # exclude model from captured kwargs
                if name in exclude:
                    continue
                if name == "kwargs" and isinstance(value, dict):
                    # flatten **kwargs so cls(**captured) reconstructs correctly
                    captured.update(value)
                else:
                    captured[name] = value
            setattr(self, store_attr, captured)

        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    
    return cls


def serialize(value):
    """Convert python objects to exportable types.

    :param value: Object for export.
    :return: Object in exportable form.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def deserialize(value):
    """Convert a dict of values to the right python objects.

    :param value: Imported object
    :return: Object as correct python-type.
    """
    if isinstance(value, list):
        return np.array(value)
    return value


def import_class(class_path: str):
    """Dynamically import a class from its module path.

    Parameters
    ----------
    class_path : str
        Full path to the class, e.g., 'timflow.steady.model.Model'.
    """
    if "." not in class_path:
        raise ValueError(f"Invalid class path: {class_path}")
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


class BaseIO:
    def to_dict(self, exclude=("model", "ml")):
        """Collect the constructor arguments into a dict.

        Parameters
        ----------
        exclude : tuple of str, optional
            Keys to exclude from kwargs. Default: ('model', 'ml')
        """
        return {
            "_type": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "kwargs": {
                k: serialize(v) for k, v in self._init_args.items() if k not in exclude
            },
        }

    @classmethod
    def from_dict(cls, data, model=None, **overrides):
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        target = data.get("_type")
        target_class = import_class(target) if target else cls
        kwargs = {k: deserialize(v) for k, v in data.get("kwargs", {}).items()}
        kwargs.update(overrides)

        # inject model if the class accepts it
        if model is not None:
            sig = inspect.signature(target_class.__init__)
            params = list(sig.parameters.keys())
            if "model" in params:
                kwargs["model"] = model
            elif "ml" in params:
                kwargs["ml"] = model

        return target_class(**kwargs)
