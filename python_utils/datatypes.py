# ======================================================

# This piece of code were copy-pasted from my own github repository

Paths = {
    "https://github.com/code1O/scorpion/blob/main/python/apple.py",     # Data types
    "https://github.com/code1O/scorpion/blob/main/python/Quantica.py",  # Pauli's & dirac's Matresses,
    "https://github.com/code1O/scorpion/blob/main/python/_defdatas.py", # Function `conjugate`
}

Repo = "https://github.com/code1O/scorpion"

import sys

from typing import (
SupportsFloat,
Union, Dict, Tuple, List, Set,
overload
)
from typing_extensions import (
SupportsIndex, TypeAlias
)

if sys.version_info >= (3, 8):
    _TypeNum: TypeAlias = SupportsFloat | SupportsIndex
else:
    _TypeNum: TypeAlias = SupportsFloat

_Typedata: TypeAlias = Union[List[_TypeNum], Set[_TypeNum],
Tuple[_TypeNum], Dict[str, _TypeNum]]

_TdataNum: TypeAlias = _Typedata | _TypeNum

def conjugate(items):
  result = []
  for item in items:
    result.append(item.conjugate())
  return result

class mats:
    """
    Essential maatresses for quantum mechanic
    
    - Pauli's matresses
    
      \\sigma^\\mu
    - gamma matresses
    
      \\gamma^\\mu
    """
    Pauli = {
    "mat_1": [0, 1, 1, 0],
    "mat_2": [0, -1j.imag, 1j.imag, 0],
    "mat_3": [1, 0, 0, -1],
    }
    __Pauli_conjg = {
        "mat_1": conjugate(Pauli["mat_1"]),
        "mat_2": conjugate(Pauli["mat_2"]),
        "mat_3": conjugate(Pauli["mat_3"])
    }
    gamma_mats = {
        "mat_1": [0, Pauli["mat_1"], __Pauli_conjg["mat_1"], 0],
        "mat_2": [0, Pauli["mat_2"], __Pauli_conjg["mat_2"], 0],
        "mat_3": [0, Pauli["mat_3"], __Pauli_conjg["mat_3"], 0]
    }

# ======================================================