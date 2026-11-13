import os
from pathlib import Path
import stat


MAX_OUTPUT_BYTES = 8 * 1024 * 1024


def read_output_file(outdir: Path, filename: str) -> bytes:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError("Expected a single output filename")

    directory = os.open(outdir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # O_NONBLOCK prevents a miner-created FIFO from blocking before fstat.
        fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"Output is not a regular file: {filename}")
            if info.st_size > MAX_OUTPUT_BYTES:
                raise ValueError(f"Output exceeds {MAX_OUTPUT_BYTES} bytes: {filename}")
            content = source.read(MAX_OUTPUT_BYTES + 1)
            if len(content) > MAX_OUTPUT_BYTES:
                raise ValueError(f"Output exceeds {MAX_OUTPUT_BYTES} bytes: {filename}")
            return content
    finally:
        os.close(directory)
