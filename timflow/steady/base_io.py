from __future__ import annotations

import inspect
from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, TypeVar

from numpy import array, ndarray

if TYPE_CHECKING:
    from timflow.steady import Model

T = TypeVar("T")


def store_input(cls: type[T]) -> type[T]:

    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)

        model_instance: Model | None
        if "Model" in self.__class__.__name__:
            model_instance = self
        else:
            if args != ():
                model_instance = args[0]
            else:
                model_instance = kwargs.pop("model", None)  # remove model ref
                if model_instance is None:
                    model_instance = kwargs.pop("ml", None)  # remove model ref
        if model_instance is not None:
            # Prevent the reference to the model object from being stored
            # this is unused and might complicate pickling.
            if len(args) != 0:
                args = args[1:] # model ref always first posarg
                
            model_instance._obj_registry.append(
                {
                    "class": f"{cls.__module__}.{cls.__qualname__}",
                    "args": args,
                    "kwargs": kwargs,
                }
            )

    cls.__init__ = new_init

    return cls


class BaseIO:
    @classmethod
    def to_dict(cls, args: tuple, kwargs: dict):
        """
        Collect the constructor arguments into a dict.

        :return: Dict with the arguments.
        """
        sig = inspect.signature(cls.__init__)
        if "Model" not in cls.__name__:
            if "model" not in kwargs or "ml" not in kwargs:
                args = args + ("model dummy",)  # add dummy for sig.bind
        bound = sig.bind(cls, *args, **kwargs)
        # Reference to class for recreation
        data = {"_type": f"{cls.__module__}.{cls.__qualname__}"}
        data.update(
            {
                k: cls._serialize(v)
                for k, v in bound.arguments.items()
                if k not in ("model", "ml", "self")
            }
        )
        return data

    @classmethod
    def _serialize(cls, value):
        """Convert python objects to exportable types.

        :param value: Object for export.
        :return: Object in exportable form.
        """
        if isinstance(value, list):
            return [cls._serialize(v) for v in value]
        if isinstance(value, dict):
            return {k: cls._serialize(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return {"tuple": [cls._serialize(v) for v in value]}
        if isinstance(value, ndarray):
            return {"ndarray": value.tolist()}
        return value

    @classmethod
    def from_dict(cls, data: dict):
        """Factory method to create an instance of this (sub)class.

        :param data: Dict with parameters
        :return: Instance of this (sub)class.
        """
        type_name: str = data["_type"]
        module_name = ".".join(type_name.split(".")[:-1])
        class_name = type_name.split(".")[-1]
        module = import_module(module_name)
        subclass = getattr(module, class_name)
        sig = inspect.signature(subclass.__init__)
        constructor_args = {}

        for name in sig.parameters:
            if name in ("model", "ml"):
                constructor_args[name] = cls._setup_model
            if name != "self" and name in data:
                constructor_args[name] = cls._deserialize(data.pop(name))
        obj = subclass(**constructor_args)
        if cls._setup_model is None:
            cls._setup_model = obj
        return obj

    @classmethod
    def _deserialize(cls, value):
        """Convert a dict of values to the right python objects.

        :param value: Imported object
        :return: Object as correct python-type.
        """
        if isinstance(value, dict) and "_type" in value:
            return cls.from_dict(value)
        if isinstance(value, dict) and "ndarray" in value:
            return array(value["ndarray"])
        if isinstance(value, dict) and "tuple" in value:
            return tuple(cls._deserialize(v) for v in value["tuple"])
        if isinstance(value, list):
            return [cls._deserialize(v) for v in value]
        if isinstance(value, dict):
            return {k: cls._deserialize(v) for k, v in value.items()}
        return value
