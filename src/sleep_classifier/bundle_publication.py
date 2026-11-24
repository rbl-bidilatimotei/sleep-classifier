import ctypes
import os
import sys
from pathlib import Path

_AT_FDCWD = -100  # Linux fcntl.h
_RENAME_NOREPLACE = 1  # Linux fs.h
_RENAME_EXCL = 0x00000004  # macOS sys/stdio.h


def publish_directory(staging: Path, destination: Path) -> None:
    """Atomically move a staged directory to an unused path on the same filesystem.

    Args:
        staging (Path): Private directory with the verified bundle contents.
        destination (Path): Final path with an existing parent directory.

    Raises:
        FileExistsError: The destination is already reserved or published.
        NotImplementedError: The system has no supported atomic exclusive rename API.
        OSError: The filesystem cannot complete the rename.
    """
    if sys.platform not in ("darwin", "linux"):
        raise NotImplementedError("Atomic bundle publication requires macOS or Linux")
    source = os.fsencode(staging)
    target = os.fsencode(destination)
    if b"\0" in source or b"\0" in target:
        raise ValueError("Bundle publication paths must not contain null bytes")

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        if sys.platform == "darwin":
            rename = libc.renamex_np
            rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
            rename.restype = ctypes.c_int
            result = rename(source, target, _RENAME_EXCL)
        else:
            rename = libc.renameat2
            rename.argtypes = (
                ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
            )
            rename.restype = ctypes.c_int
            result = rename(_AT_FDCWD, source, _AT_FDCWD, target, _RENAME_NOREPLACE)
    except AttributeError as error:
        raise NotImplementedError("The system lacks an atomic exclusive rename API") from error

    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(staging), None, str(destination))
