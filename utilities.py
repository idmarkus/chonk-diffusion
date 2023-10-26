from __future__ import annotations

from datetime import datetime
from typing import Union, Optional
from pathlib import Path

import errno
import os

import numpy as np
#import cv2 as cv
import torch
from PIL import Image

# Type annotation for path
Pathlike = Union[Path, str]
Imagelike = Union[Image.Image, np.array]
Tensorlike = Union[torch.Tensor, np.array]


# Monkey patch Path methods to work better on MINGW
def sane_path_expanduser(self: Path) -> Path:
    """
    Expand ~ to env["HOME"] if defined, else expanduser().
    To ensure that paths expand correctly on MSYS/MINGW.
    """
    if "HOME" in os.environ:
        return Path(str(self).replace("~", os.environ["HOME"]))
    else:
        # This is set later in this file
        return self.__builtin_expanduser()


def sane_path_str(self: Path, absolute=False, expand=False) -> str:
    """
    Monkey-patch method always giving unix-like '/' str.

    :param absolute: Expand to absolute Path
    :param expand: Alias for :absolute:
    :return: Pathstring with unix slashes '/'
    """
    sep = self._flavour.sep
    return (self.expanduser().absolute() if absolute or expand else self).__builtin__str__().replace(sep, '/')


Path.__builtin_expanduser = Path.expanduser
Path.__builtin__str__ = Path.__str__

Path.expanduser = sane_path_expanduser
Path.sanestr = sane_path_str

Path.__str__ = sane_path_str


def timestamp(strf: str = "%y%m%d_%H%M%S") -> str:
    return datetime.now().strftime(strf)

class InvalidPath(Exception):
    pass


def assert_exists(path: Pathlike, msg="") -> Union[bool, InvalidPath]:
    """
    Raise FileNotFoundError if path does not exist
    """
    if not Path(path).exists():
        msg = str(Path(path)) + (": " + msg if msg else "")
        raise InvalidPath("No such file or directory: " + msg)
        # raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), msg)
    return True


def assert_isdir(path: Pathlike, msg="", exists: bool = False):
    """
    Raise InvalidPath if path exists and is not a directory.
    :param msg: Optional message to append
    :param exists: Also raise if path does not exist
    """
    if exists and not Path(path).exists():
        assert_exists(path, msg=msg)

    if Path(path).exists() and not Path(path).is_dir():
        msg = str(Path(path)) + (": " + msg if msg else "")
        raise InvalidPath("Existing path is not a directory: " + msg)

