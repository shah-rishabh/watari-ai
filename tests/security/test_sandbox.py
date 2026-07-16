"""Red-team the filesystem jail.

These are adversarial unit cases: every escape vector we claim to block must be
demonstrably blocked here. If one of these ever passes, the jail is broken.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from watari.security.sandbox import Sandbox, SandboxError


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "inside").mkdir()
    (root / "inside" / "file.txt").write_text("ok", encoding="utf-8")
    return Sandbox(root)


class TestAllowedPaths:
    def test_relative_path_inside_is_allowed(self, sandbox: Sandbox) -> None:
        resolved = sandbox.resolve("inside/file.txt")
        assert resolved.name == "file.txt"
        assert sandbox.root in resolved.parents

    def test_new_nested_path_is_allowed(self, sandbox: Sandbox) -> None:
        # Writing a not-yet-existing nested path stays contained.
        resolved = sandbox.resolve("a/b/c.txt")
        assert sandbox.root in resolved.parents

    def test_root_itself_is_allowed(self, sandbox: Sandbox) -> None:
        assert sandbox.resolve(".") == sandbox.root


class TestBlockedEscapes:
    def test_dotdot_traversal_blocked(self, sandbox: Sandbox) -> None:
        with pytest.raises(SandboxError):
            sandbox.resolve("../../etc/passwd")

    def test_bare_dotdot_blocked(self, sandbox: Sandbox) -> None:
        with pytest.raises(SandboxError):
            sandbox.resolve("..")

    def test_deep_traversal_blocked(self, sandbox: Sandbox) -> None:
        with pytest.raises(SandboxError):
            sandbox.resolve("inside/../../../..")

    def test_null_byte_blocked(self, sandbox: Sandbox) -> None:
        with pytest.raises(SandboxError):
            sandbox.resolve("foo\x00.txt")

    def test_absolute_path_is_remapped_into_jail_not_escaped(self, sandbox: Sandbox) -> None:
        # An absolute path must never reach the real filesystem root; it is
        # treated as jail-relative and stays contained.
        resolved = sandbox.resolve("/etc/passwd")
        assert sandbox.root in resolved.parents
        assert str(resolved) != "/etc/passwd"

    def test_tilde_is_a_dirname_not_home(self, sandbox: Sandbox) -> None:
        resolved = sandbox.resolve("~/secrets")
        # "~" is a literal directory under the jail, not the real home dir.
        assert sandbox.root in resolved.parents
        assert "~" in resolved.parts

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
    def test_symlink_escape_blocked(self, sandbox: Sandbox, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("leaked", encoding="utf-8")
        os.symlink(outside, sandbox.root / "escape")
        with pytest.raises(SandboxError):
            sandbox.resolve("escape/secret.txt")
