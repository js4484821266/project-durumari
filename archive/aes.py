"""AES-256-GCM file encryption and decryption.

Install the only external dependency with::

    python -m pip install cryptography

The encrypted file format is::

    magic (8 bytes) | salt (16) | nonce (12) | ciphertext | GCM tag (16)

The header is authenticated as associated data, so changing it is detected.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import tempfile
from typing import BinaryIO

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


MAGIC = b"DURMAES1"
SALT_SIZE = 16
NONCE_SIZE = 12
TAG_SIZE = 16
KEY_SIZE = 32
CHUNK_SIZE = 1024 * 1024
HEADER_SIZE = len(MAGIC) + SALT_SIZE + NONCE_SIZE


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit key while making password guessing expensive."""
    if not password:
        raise ValueError("암호는 비어 있을 수 없습니다.")
    kdf = Scrypt(salt=salt, length=KEY_SIZE, n=2**15, r=8, p=1)
    return kdf.derive(password.encode("utf-8"))


def _validate_paths(source: Path, destination: Path, overwrite: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("입력과 출력 파일은 같을 수 없습니다.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"출력 파일이 이미 있습니다: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)


def _temporary_output(destination: Path) -> tuple[BinaryIO, Path]:
    temporary = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    return temporary, Path(temporary.name)


def encrypt_file(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    password: str,
    *,
    overwrite: bool = False,
) -> None:
    """Encrypt a file with AES-256-GCM without loading it all into memory."""
    source, destination = Path(input_path), Path(output_path)
    _validate_paths(source, destination, overwrite)

    salt, nonce = os.urandom(SALT_SIZE), os.urandom(NONCE_SIZE)
    header = MAGIC + salt + nonce
    encryptor = Cipher(algorithms.AES(_derive_key(password, salt)), modes.GCM(nonce)).encryptor()
    encryptor.authenticate_additional_data(header)

    output, temporary_path = _temporary_output(destination)
    try:
        with source.open("rb") as source_file, output:
            output.write(header)
            while chunk := source_file.read(CHUNK_SIZE):
                output.write(encryptor.update(chunk))
            output.write(encryptor.finalize())
            output.write(encryptor.tag)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        output.close()
        temporary_path.unlink(missing_ok=True)
        raise


def decrypt_file(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    password: str,
    *,
    overwrite: bool = False,
) -> None:
    """Decrypt a file and reject a wrong password or any modified content."""
    source, destination = Path(input_path), Path(output_path)
    _validate_paths(source, destination, overwrite)
    encrypted_size = source.stat().st_size
    if encrypted_size < HEADER_SIZE + TAG_SIZE:
        raise ValueError("암호화 파일이 너무 짧거나 손상되었습니다.")

    output, temporary_path = _temporary_output(destination)
    try:
        with source.open("rb") as source_file, output:
            header = source_file.read(HEADER_SIZE)
            if header[: len(MAGIC)] != MAGIC:
                raise ValueError("지원하지 않는 암호화 파일 형식입니다.")
            salt = header[len(MAGIC) : len(MAGIC) + SALT_SIZE]
            nonce = header[-NONCE_SIZE:]
            source_file.seek(-TAG_SIZE, os.SEEK_END)
            tag = source_file.read(TAG_SIZE)
            source_file.seek(HEADER_SIZE)

            decryptor = Cipher(
                algorithms.AES(_derive_key(password, salt)), modes.GCM(nonce, tag)
            ).decryptor()
            decryptor.authenticate_additional_data(header)

            remaining = encrypted_size - HEADER_SIZE - TAG_SIZE
            while remaining:
                chunk = source_file.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    raise ValueError("암호화 파일이 손상되었습니다.")
                output.write(decryptor.update(chunk))
                remaining -= len(chunk)
            output.write(decryptor.finalize())
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
    except InvalidTag as error:
        output.close()
        temporary_path.unlink(missing_ok=True)
        raise ValueError("암호가 틀렸거나 파일이 변조되었습니다.") from error
    except BaseException:
        output.close()
        temporary_path.unlink(missing_ok=True)
        raise


def _default_output(command: str, input_path: Path) -> Path:
    if command == "encrypt":
        return input_path.with_name(input_path.name + ".aes")
    if input_path.suffix.lower() == ".aes":
        return input_path.with_suffix("")
    return input_path.with_name(input_path.name + ".decrypted")


def main() -> int:
    parser = argparse.ArgumentParser(description="AES-256-GCM 파일 암·복호화")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("encrypt", "decrypt"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("input", type=Path, help="입력 파일")
        subparser.add_argument("output", type=Path, nargs="?", help="출력 파일")
        subparser.add_argument("--force", action="store_true", help="출력 파일 덮어쓰기")
    args = parser.parse_args()

    output = args.output or _default_output(args.command, args.input)
    password = getpass.getpass("암호: ")
    if args.command == "encrypt":
        if password != getpass.getpass("암호 확인: "):
            parser.error("암호가 서로 다릅니다.")
        encrypt_file(args.input, output, password, overwrite=args.force)
    else:
        decrypt_file(args.input, output, password, overwrite=args.force)
    print(f"완료: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
