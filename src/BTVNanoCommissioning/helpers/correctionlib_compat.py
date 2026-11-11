"""
correctionlib >= 2.8 compatibility for the evaluation code of this framework.

Up to correctionlib 2.7, ``Correction.evaluate`` returned a numpy array for flat
(numpy-convertible) awkward inputs and an awkward array only for jagged inputs.
From 2.8 on (correctionlib #306) every awkward input gives an awkward output. The
correction code here (e.g. helpers/MuonScaRe.py, utils/correction.py) was written for
the former and uses numpy operations such as in-place assignment on the result.

Code that checks ``"correctionlib" in str(type(obj))`` (utils/correction.py) still
recognises the wrapped sets: the module name contains "correctionlib".

``from_file`` / ``from_string`` return the correction sets wrapped such that
``evaluate`` passes flat awkward inputs as numpy arrays: numpy outputs for flat inputs,
awkward outputs for jagged inputs, as before. Corrections reached via ``cset[name]``,
``cset.get(name)`` and ``cset.compound[name]`` are wrapped.
"""

from collections.abc import Mapping

import awkward as ak
import correctionlib
import numpy as np


def _as_numpy_if_flat(arg):
    if isinstance(arg, ak.Array):
        try:
            return np.asarray(ak.to_numpy(arg))
        except (ValueError, TypeError):
            return arg  # jagged / not numpy-convertible: keep awkward
    return arg


class CompatCorrection:
    """Correction whose evaluate() gives numpy outputs for flat awkward inputs."""

    def __init__(self, correction):
        self.__dict__["_correction"] = correction

    def evaluate(self, *args):
        return self.__dict__["_correction"].evaluate(*map(_as_numpy_if_flat, args))

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__dict__["_correction"], name)

    def __getstate__(self):
        return {"_correction": self.__dict__["_correction"]}

    def __setstate__(self, state):
        self.__dict__["_correction"] = state["_correction"]


class _CompatMapping(Mapping):
    def __init__(self, mapping):
        self.__dict__["_mapping"] = mapping

    def __getitem__(self, key):
        return CompatCorrection(self.__dict__["_mapping"][key])

    def __iter__(self):
        return iter(self.__dict__["_mapping"])

    def __len__(self):
        return len(self.__dict__["_mapping"])

    def __getstate__(self):
        return {"_mapping": self.__dict__["_mapping"]}

    def __setstate__(self, state):
        self.__dict__["_mapping"] = state["_mapping"]


class CompatCorrectionSet(_CompatMapping):  # type name contains "correctionlib" (module)
    """correctionlib CorrectionSet with CompatCorrection items (and compound items)."""

    @property
    def compound(self):
        return _CompatMapping(self.__dict__["_mapping"].compound)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__dict__["_mapping"], name)


def from_file(path):
    return CompatCorrectionSet(correctionlib.CorrectionSet.from_file(path))


def from_string(data):
    return CompatCorrectionSet(correctionlib.CorrectionSet.from_string(data))
