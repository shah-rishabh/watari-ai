"""Filesystem path jail for agent tools.

A pure-Python jail — honestly documented as such (see ``docs/threat-model.md``).
It resolves every requested path (following symlinks) and confirms the result
lies within a configured workspace root. This defeats the common escape vectors
— ``..`` traversal, absolute paths, ``~`` expansion, and symlinks that point
outside the jail — and is fully unit-testable.

It is *not* a container or seccomp sandbox, and we do not claim it is. A tool
that is allowed to write files can still fill the workspace or write malicious
content there; the jail bounds *where*, not *what*. The threat model states this
residual risk explicitly.
"""

from __future__ import annotations

from pathlib import Path


class SandboxError(Exception):
    """Raised when a requested path escapes the workspace jail."""


class Sandbox:
    def __init__(self, root: Path) -> None:
        # Resolve the root once so all comparisons are against a canonical,
        # symlink-free absolute path.
        self._root = root.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, user_path: str | Path) -> Path:
        """Resolve ``user_path`` relative to the jail and verify containment.

        Raises :class:`SandboxError` if the resolved path is outside the
        workspace. Returns the canonical absolute path on success.
        """
        raw = str(user_path)
        if "\x00" in raw:
            raise SandboxError("path contains a null byte")

        candidate = Path(raw)
        # Absolute paths and ~ are treated as jail-relative, never as an escape:
        # we strip any leading root/anchor and join under the workspace.
        if candidate.is_absolute():
            candidate = Path(*candidate.parts[1:]) if len(candidate.parts) > 1 else Path()
        # Expanduser is deliberately NOT called — "~" is just a directory name
        # inside the jail, not the real home directory.

        resolved = (self._root / candidate).resolve()

        # strict containment: resolved must be the root or a descendant of it.
        if resolved != self._root and self._root not in resolved.parents:
            raise SandboxError(f"path escapes workspace: {user_path!r}")
        return resolved
