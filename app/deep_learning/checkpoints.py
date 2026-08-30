"""Verified downloads and provenance for Phase 3 pretrained checkpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlopen

from app.deep_learning.config import DeepLearningModelConfig


@dataclass(frozen=True)
class CheckpointProvenance:
    model: str
    scale: int
    source_url: str
    original_filename: str
    sha256: str
    size_bytes: int
    architecture: str
    training_dataset: str
    degradation: str
    source_repository: str


_OFFICIAL_IMDN_REPOSITORY = "https://github.com/Zheng222/IMDN"
_OFFICIAL_IMDN_RAW_ROOT = (
    "https://raw.githubusercontent.com/Zheng222/IMDN/master/checkpoints"
)
IMDN_CHECKPOINTS = {
    2: CheckpointProvenance(
        model="imdn",
        scale=2,
        source_url=f"{_OFFICIAL_IMDN_RAW_ROOT}/IMDN_x2.pth",
        original_filename="IMDN_x2.pth",
        sha256="740b39653592bd3f49955a21cd9e16f2d37f2c413368cc09327601b8fbe42ce1",
        size_bytes=2_796_052,
        architecture="IMDN RGB, 64 features, 6 IMD modules",
        training_dataset="DIV2K",
        degradation="bicubic",
        source_repository=_OFFICIAL_IMDN_REPOSITORY,
    ),
    3: CheckpointProvenance(
        model="imdn",
        scale=3,
        source_url=f"{_OFFICIAL_IMDN_RAW_ROOT}/IMDN_x3.pth",
        original_filename="IMDN_x3.pth",
        sha256="11b1716a19bc16e5e353e1db798d476169338e631eb170c82d765559888cb028",
        size_bytes=2_830_672,
        architecture="IMDN RGB, 64 features, 6 IMD modules",
        training_dataset="DIV2K",
        degradation="bicubic",
        source_repository=_OFFICIAL_IMDN_REPOSITORY,
    ),
    4: CheckpointProvenance(
        model="imdn",
        scale=4,
        source_url=f"{_OFFICIAL_IMDN_RAW_ROOT}/IMDN_x4.pth",
        original_filename="IMDN_x4.pth",
        sha256="e660c38221147217a136debeb99eb304d62d9c0f73ac421ed66281b619719714",
        size_bytes=2_879_112,
        architecture="IMDN RGB, 64 features, 6 IMD modules",
        training_dataset="DIV2K",
        degradation="bicubic",
        source_repository=_OFFICIAL_IMDN_REPOSITORY,
    ),
}


_FSRCNN_REPOSITORY = "https://github.com/yjn870/FSRCNN-pytorch"
_FSRCNN_DROPBOX_ROOT = "https://dl.dropboxusercontent.com/s"
FSRCNN_CHECKPOINTS = {
    2: CheckpointProvenance(
        model="fsrcnn",
        scale=2,
        source_url=f"{_FSRCNN_DROPBOX_ROOT}/1k3dker6g7hz76s/fsrcnn_x2.pth",
        original_filename="fsrcnn_x2.pth",
        sha256="1af70e2cbf0d11b942008d6c425e469e9fe275a68f3ac8b6c21e8672d6e67b69",
        size_bytes=54_998,
        architecture="FSRCNN luminance, d=56, s=12, m=4",
        training_dataset="91-image",
        degradation="bicubic",
        source_repository=_FSRCNN_REPOSITORY,
    ),
    3: CheckpointProvenance(
        model="fsrcnn",
        scale=3,
        source_url=f"{_FSRCNN_DROPBOX_ROOT}/pm1ed2nyboulz5z/fsrcnn_x3.pth",
        original_filename="fsrcnn_x3.pth",
        sha256="f65b55819c1258048bc8c1ca96f9e24bd4339f779951d7f928144c08e3868d75",
        size_bytes=54_998,
        architecture="FSRCNN luminance, d=56, s=12, m=4",
        training_dataset="91-image",
        degradation="bicubic",
        source_repository=_FSRCNN_REPOSITORY,
    ),
    4: CheckpointProvenance(
        model="fsrcnn",
        scale=4,
        source_url=f"{_FSRCNN_DROPBOX_ROOT}/vsvumpopupdpmmu/fsrcnn_x4.pth",
        original_filename="fsrcnn_x4.pth",
        sha256="c15150d6787d487f38a68e66be5ec8a964182403af494e6a935fa03eeb56a630",
        size_bytes=54_998,
        architecture="FSRCNN luminance, d=56, s=12, m=4",
        training_dataset="91-image",
        degradation="bicubic",
        source_repository=_FSRCNN_REPOSITORY,
    ),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_official_imdn_checkpoint(
    checkpoint_root: str | Path,
    scale: int,
) -> Path:
    """Download one official IMDN checkpoint and reject altered content."""
    try:
        provenance = IMDN_CHECKPOINTS[scale]
    except KeyError as error:
        raise ValueError(f"IMDN scale must be 2, 3, or 4; received {scale}.") from error

    config = DeepLearningModelConfig("imdn", scale, Path(checkpoint_root))
    target = config.checkpoint_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != provenance.size_bytes:
            raise ValueError(f"Existing IMDN checkpoint has the wrong size: {target}")
        if sha256_file(target) != provenance.sha256:
            raise ValueError(f"Existing IMDN checkpoint failed SHA-256: {target}")
        return target

    temporary = target.with_suffix(f"{target.suffix}.download")
    try:
        digest = hashlib.sha256()
        downloaded_size = 0
        with urlopen(provenance.source_url, timeout=60) as response:
            with temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    downloaded_size += len(block)

        if downloaded_size != provenance.size_bytes:
            raise ValueError(
                "Downloaded IMDN checkpoint has the wrong size: "
                f"expected {provenance.size_bytes}, received {downloaded_size}."
            )
        if digest.hexdigest() != provenance.sha256:
            raise ValueError("Downloaded IMDN checkpoint failed SHA-256 verification.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = target.with_suffix(".provenance.json")
    manifest.write_text(
        json.dumps(asdict(provenance), indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def download_pretrained_fsrcnn_checkpoint(
    checkpoint_root: str | Path,
    scale: int,
) -> Path:
    """Download one published FSRCNN checkpoint and verify its exact content."""
    try:
        provenance = FSRCNN_CHECKPOINTS[scale]
    except KeyError as error:
        raise ValueError(f"FSRCNN scale must be 2, 3, or 4; received {scale}.") from error

    config = DeepLearningModelConfig("fsrcnn", scale, Path(checkpoint_root))
    target = config.checkpoint_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != provenance.size_bytes:
            raise ValueError(f"Existing FSRCNN checkpoint has the wrong size: {target}")
        if sha256_file(target) != provenance.sha256:
            raise ValueError(f"Existing FSRCNN checkpoint failed SHA-256: {target}")
        return target

    temporary = target.with_suffix(f"{target.suffix}.download")
    try:
        digest = hashlib.sha256()
        downloaded_size = 0
        with urlopen(provenance.source_url, timeout=60) as response:
            with temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    downloaded_size += len(block)

        if downloaded_size != provenance.size_bytes:
            raise ValueError(
                "Downloaded FSRCNN checkpoint has the wrong size: "
                f"expected {provenance.size_bytes}, received {downloaded_size}."
            )
        if digest.hexdigest() != provenance.sha256:
            raise ValueError("Downloaded FSRCNN checkpoint failed SHA-256 verification.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = target.with_suffix(".provenance.json")
    manifest.write_text(
        json.dumps(asdict(provenance), indent=2) + "\n",
        encoding="utf-8",
    )
    return target
