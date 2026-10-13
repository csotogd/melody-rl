"""The OOD evaluation must fail before use if a frozen adapter changed."""
import hashlib

import pytest

from scripts.run_generalization import check_frozen_adapters


def test_frozen_policy_change_is_rejected(tmp_path):
    path = tmp_path / 'adapter.safetensors'
    path.write_bytes(b'frozen policy')
    audit = {'frozen_adapter_sha256': {'sft': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}}
    check_frozen_adapters(tmp_path, audit)
    path.write_bytes(b'tuned after test')
    with pytest.raises(ValueError, match='checksum mismatch'):
        check_frozen_adapters(tmp_path, audit)
