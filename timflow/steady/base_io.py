import inspect
import json

from numpy import array, ndarray
from typing_extensions import Self


class BaseIO:
    # Registry for all subclasses.
    _class_registry = {}
    # Registry for all created objects with their kwargs for storing.
    _obj_registry = {}
    # Registry for model instance for storing.
    _model_registry = {}

    def __init_subclass__(cls) -> None:
        """Add the subclass to the registry on inheritance."""
        cls._class_registry[cls.__name__] = cls

    def __new__(cls, *args, **kwargs) -> Self:
        """Add all newly created object to a registry if they are created directly.

        :return: instance of the (sub)class
        """
        instance = super().__new__(cls)
        frame = inspect.currentframe()
        caller = frame.f_back
        if caller.f_code.co_name == "<module>":
            # If a new Model object create a new list before adding it.
            if "Model" in str(cls.__name__):
                m = f"model{len(cls._obj_registry)}"
                cls._model_registry.update({instance: m})
                cls._obj_registry.update({m: []})
                cls._obj_registry[m].append((instance, args, kwargs))
            # Other objects are added to the list of the model they have been
            # added to.
            else:
                if args != ():
                    m_inst = args[0]
                else:
                    m_inst = kwargs.get("model", None)
                    if m_inst is None:
                        m_inst = kwargs.get("ml")
                cls._obj_registry[cls._model_registry[m_inst]].append(
                    (instance, args, kwargs)
                )
        return instance

    def to_json(self, filepath) -> None:
        """
        Write the constructor arguments to a JSON-file.

        :param filepath: Filepath for the to be created JSON-file.
        """
        data = {}
        i = 0
        for item in self._obj_registry[self._model_registry[self]]:
            obj, args, kwargs = item
            data.update({f"object{i}": obj.to_dict(args, kwargs)})
            i += 1
        with open(filepath, "w") as f:
            f.write(json.dumps(data, indent=4))

    def to_dict(self, args, kwargs):
        """
        Collect the constructor arguments into a dict.

        :return: Dict with the arguments.
        """
        sig = inspect.signature(self.__init__)
        bound = sig.bind(*args, **kwargs)
        # Reference to class for recreation
        data = {"_type": self.__class__.__name__}
        data.update(
            {
                k: self._serialize(v)
                for k, v in bound.arguments.items()
                if k not in ("model", "ml")
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
        if isinstance(value, ndarray):
            return {"ndarray": value.tolist()}
        return value

    @classmethod
    def from_json(cls, filepath):
        """
        Read the constructor arguments and potential addition attributes from a JSON-file.

        :param filepath: Filepath to the to be created JSON-file.
        """
        cls._setup_model = None
        with open(filepath, "r") as f:
            data: dict = json.load(f)
        for k, v in data.items():
            if k == "object0":  # Model object is always first created.
                obj = cls.from_dict(v)
                continue
            if "obj" not in locals():  # No model in json
                raise ImportError("No main model found in the JSON-file.")
            cls.from_dict(v)
        return obj

    @classmethod
    def from_dict(cls, data: dict):
        """Factory method to create an instance of this (sub)class.

        :param data: Dict with parameters
        :return: Instance of this (sub)class.
        """
        type_name: str = data["_type"]
        subclass = cls._class_registry[type_name]
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
        if isinstance(value, list):
            return [cls._deserialize(v) for v in value]
        if isinstance(value, dict):
            return {k: cls._deserialize(v) for k, v in value.items()}
        return value
