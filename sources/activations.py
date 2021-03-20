"""Common activation functions package"""
import abc
import math

from .misc import subclasshook_helper


REQUIRED = (
    '__call__',
    'derivative'
)


class IActivation(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return subclasshook_helper(REQUIRED)

    @abc.abstractmethod
    def __call__(self, value: float) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def derivative(self, value: float) -> float:
        raise NotImplementedError


# Type hinting alias
Activation = IActivation


# str to Activation mapping
activations = dict()


def register(cls):
    """Register class in activations dictionary"""
    activations[cls.__name__] = cls

    return cls


@register
class Identity(IActivation):
    def __call__(self, value: float) -> float:
        return value

    def derivative(self, _) -> float:
        return 1.


# Since Identity does not require __init__
identity = Identity()
