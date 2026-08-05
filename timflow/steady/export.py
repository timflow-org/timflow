import inspect
import json


# TODO Print logs for insight to where we get during run.
class ExportBase:
    #  Registry for all subclasses.
    _registry = {}

    def __init_subclass__(cls) -> None:
        """Add the subclass to the registry on creation."""
        cls._registry[cls.__name__] = cls

    def to_json(self, filepath) -> None:
        """
        Write the contructor arguments and potential additional attributes to a JSON-file.

        :param filepath: Filepath to the to be created JSON-file.
        """
        data = self.to_dict()
        print("Test:")
        print(data)
        with open(filepath, "w") as f:
            f.write(json.dumps(data, indent=4))

    def to_dict(self):
        """
        Collect the contructor arguments and potential additional attributes into a dict.

        :return: _description_
        """
        sig = inspect.signature(self.__init__)
        data = {"_type": self.__class__.__name__}
        for name in sig.parameters:
            if name == "model":  # reference to parent object
                continue
            # TODO Reference to other object, inhomogenities need to go to JSON
            if name in ["aq", "aqin", "aqout"]:
                continue
            if name != "self":
                value = getattr(self, name)
                data[name] = self._serialize(value)
        data.update(self.extra_to_dict())
        return data

    def extra_to_dict(self):
        """Add the addition attributes to the dict.

        May be overloaded in the subclass.

        :return: Dict with addition parameters.
        """
        return {}

    @classmethod
    def from_json(cls, filepath) -> dict:
        """
        Read the contructor arguments and potential addition attributes from a JSON-file.

        :param filepath: Filepath to the to be created JSON-file.
        """
        with open(filepath, "r") as f:
            data = json.loads(f)
        return data

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
            if name == "model":
                constructor_args[name] = cls
            if name != "self" and name in data:
                constructor_args[name] = cls._deserialize(data.pop(name))
        obj = subclass(**constructor_args)
        obj.extra_from_dict(data)

        return obj

    def extra_from_dict(self, data) -> None:
        """Add the additional attributes to the (sub)class.

        May be overloaded in the subclasses.

        :param data: Dict with additional parameters.
        """
        pass

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
        return value

    @classmethod
    def _deserialize(cls, value):
        """Convert a dict of values to the right python objects.

        :param value: Imported object
        :return: Object as correct python-type.
        """
        if isinstance(value, dict) and "_type" in value:
            return cls.from_dict(value)
        if isinstance(value, list):
            return [cls._deserialize(v) for v in value]
        if isinstance(value, dict):
            return {k: cls._deserialize(v) for k, v in value.items()}
