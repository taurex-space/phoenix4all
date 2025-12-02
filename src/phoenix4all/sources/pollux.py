import pathlib
from typing import Optional

POLLUX_BASE_URL = "https://pollux.oreme.org/"

AVAILABLE_MODELS = [
    "AMBRE",
    "BT-Dusty",
    "CMFGEN",
    "CMFGEN-OB-LMC-24",
    "CMFGEN-OB-SMC-24",
    "CMFGEN-VMS-Z0p01ZSun",
    "CMFGEN-VMS-Z0p1ZSun",
    "CMFGEN-VMS-ZSMC",
    "POPSYCLE",
    "RSG",
    "STAGGER",
    "STAGGER-INTENSITY",
    "STAGGER-RVS",
]


def load_spectrum(
    spectrum_pk: str,
    model_name: str,
    base_path: Optional[pathlib.Path] = None,
    base_url: Optional[str] = None,
):
    pass
