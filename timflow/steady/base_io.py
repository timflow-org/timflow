import inspect
import json

from numpy import array, ndarray
from typing_extensions import Self


class BaseIO:
    # Registry for all subclasses.
    _registry = {}
    # Storage for model object
    _model = None
    # Registry for all created objects with their kwargs
    _obj_list = []

    def __new__(cls, *args, **kwargs) -> Self:
        """Register created objects in script.
        
        When a object is created in the script, register this object with the
        constructor kwargs. If the object is made inside of another class or function
        don't register it.

        :return: Created object.
        """        
        instance = super().__new__(cls)
        frame = inspect.currentframe()
        caller = frame.f_back
        if caller.f_code.co_name == "<module>":
            cls._obj_list.append((instance, kwargs))
        return instance

    def __init_subclass__(cls) -> None:
        """Add the subclass to the registry on inheritance."""
        cls._registry[cls.__name__] = cls

    def to_json(self, filepath) -> None:
        """
        Write the constructor arguments to a JSON-file.

        :param filepath: Filepath for the to be created JSON-file.
        """
        data = {}
        i = 0
        for item in self._obj_list:
            obj, kwargs = item
            data.update({f"object{i}": obj.to_dict(**kwargs)})
            i += 1
        with open(filepath, "w") as f:
            f.write(json.dumps(data, indent=4))

    def to_dict(self, **kwargs):
        """
        Collect the constructor arguments into a dict.

        :return: Dict with the arguments.
        """
        sig = inspect.signature(self.__init__)
        data = {"_type": self.__class__.__name__}
        for name in sig.parameters:
            if name == ("model" or "ml"):  # reference to parent object
                continue
            if name in ["aq", "aqin", "aqout"]:
                continue
            if name != "self":
                # For kwargs as inputs
                value = kwargs.get(name, None)
                # If not used as input -> collect from attributes
                if value is None:
                    value = getattr(self, name, None)
                data[name] = self._serialize(value)
        return data

    @classmethod
    def from_json(cls, filepath):
        """
        Read the constructor arguments and potential addition attributes from a JSON-file.

        :param filepath: Filepath to the to be created JSON-file.
        """
        obj = None
        with open(filepath, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            if k == "object0":  # Model object is always first created.
                obj = cls.from_dict(v)
            else:
                cls.from_dict(v)
        if obj is None:
            raise ImportError
        return obj

    @classmethod
    def from_dict(cls, data: dict):
        """Factory method to create an instance of this (sub)class.

        :param data: Dict with parameters
        :return: Instance of this (sub)class.
        """
        type_name = data.pop("_type")
        subclass = cls._registry[type_name]
        sig = inspect.signature(subclass.__init__)
        constructor_args = {}

        for name in sig.parameters:
            if name == ("model" or "ml"):
                constructor_args[name] = cls._model
            if name != "self" and name in data:
                constructor_args[name] = cls._deserialize(data.pop(name))
        obj = subclass(**constructor_args)
        if cls._model is None:
            cls._model = obj
        return obj

    @classmethod
    def _serialize(cls, value):
        """Convert python objects to exportable types.

        :param value: Object for export.
        :return: Object in exportable form.
        """
        if isinstance(value, cls):
            return value.to_dict()
        if isinstance(value, list):
            return [cls._serialize(v) for v in value]
        if isinstance(value, dict):
            return {k: cls._serialize(v) for k, v in value.items()}
        if isinstance(value, ndarray):
            return {"ndarray": value.tolist()}
        return value

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
