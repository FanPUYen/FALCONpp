"""Public API for the readable Falcon++ research implementation.

This package is not a constant-time production cryptographic library.  Its
top-level exports are intentionally small; lower-level numerical and audit
helpers remain available from their respective submodules.
"""

from .keygen import generate_keypair
from .keys import KeyPair, PublicKey, SecretKey, Signature
from .parameters import (
    FALCONPP_1024_117,
    FALCONPP_1024_125,
    FALCONPP_512_117,
    FALCONPP_512_125,
    FALCONPP_I_1245,
    FALCONPP_I_1330,
    FALCONPP_V_1205,
    FALCONPP_V_1330,
    ADDITIONAL_PARAMETER_SETS,
    ORIGINAL_PARAMETER_SETS,
    PARAMETERS,
    PARAMETER_SETS,
    FalconPPParameters,
    get_parameters,
    iter_parameter_sets,
)
from .scheme import FalconPP, FalconPlusPlus, keygen, sign, verify
from .signing import sign_message
from .verification import verify_signature, verify_signature_detailed


__version__ = "0.1.0"


__all__ = [
    "__version__",
    "FalconPP",
    "FalconPlusPlus",
    "FalconPPParameters",
    "FALCONPP_512_117",
    "FALCONPP_512_125",
    "FALCONPP_1024_117",
    "FALCONPP_1024_125",
    "FALCONPP_I_1245",
    "FALCONPP_I_1330",
    "FALCONPP_V_1205",
    "FALCONPP_V_1330",
    "ADDITIONAL_PARAMETER_SETS",
    "ORIGINAL_PARAMETER_SETS",
    "PARAMETERS",
    "PARAMETER_SETS",
    "KeyPair",
    "PublicKey",
    "SecretKey",
    "Signature",
    "generate_keypair",
    "get_parameters",
    "iter_parameter_sets",
    "keygen",
    "sign",
    "sign_message",
    "verify",
    "verify_signature",
    "verify_signature_detailed",
]
