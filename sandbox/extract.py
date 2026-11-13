from pathlib import Path
import tarfile


def prepare_workspace(archive: Path, scratch: Path, input_path: Path) -> Path:
    extracted = scratch / "submission"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extracted, filter="data")

    root = extracted
    if not (root / "miner.py").is_file():
        candidates = [p for p in extracted.iterdir() if p.is_dir() and (p / "miner.py").is_file()]
        if len(candidates) != 1:
            raise ValueError("Archive must contain one miner.py repository")
        root = candidates[0]

    workspace = scratch / "workspace"
    root.rename(workspace)
    (workspace / "input.json").unlink(missing_ok=True)
    (workspace / "input.json").symlink_to(input_path)
    return workspace


if __name__ == "__main__":
    prepare_workspace(Path("/run/submission.tar.gz"), Path("/tmp"), Path("/run/input.json"))
