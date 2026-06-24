#!/usr/bin/env python3
import hashlib
import sys
from pathlib import Path


def solve(path):
    tokens = Path(path).read_text(encoding="ascii").split()
    count = int(tokens[0])
    limit = int(tokens[1])
    records = [item.encode("ascii") for item in tokens[2:]]
    if len(records) != count:
        raise ValueError("input record count does not match header")
    level = [hashlib.sha256(item).digest() for item in records]
    first = level[0]
    while len(level) > 1:
        level = [
            hashlib.sha256(level[index] + level[index + 1 if index + 1 < len(level) else index]).digest()
            for index in range(0, len(level), 2)
        ]
    root = level[0]
    nonce = next(
        value
        for value in range(limit)
        if hashlib.sha256(root + value.to_bytes(4, "big")).digest()[:2] == b"\0\0"
    )
    block_hash = hashlib.sha256(root + nonce.to_bytes(4, "big")).digest()
    return {
        "sha256_first": first.hex(),
        "merkle_root": root.hex(),
        "nonce": str(nonce),
        "block_hash": block_hash.hex(),
    }


if __name__ == "__main__":
    for key, value in solve(sys.argv[1]).items():
        print(f"{key}={value}")
