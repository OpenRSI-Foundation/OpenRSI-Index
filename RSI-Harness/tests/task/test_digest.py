import os

import pytest

from rsi_harness.errors import UnsupportedTaskError
from rsi_harness.task.digest import hash_tree


def test_hash_tree_is_order_independent_and_content_sensitive(tmp_path):
    """Filesystem enumeration order must not alter an identical task digest."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "b.txt").write_text("second")
    (first / "a.txt").write_text("first")
    (second / "a.txt").write_text("first")
    (second / "b.txt").write_text("second")

    assert hash_tree(first) == hash_tree(second)

    (second / "b.txt").write_text("changed")
    assert hash_tree(first) != hash_tree(second)


def test_hash_tree_covers_relative_path_mode_and_entry_type(tmp_path):
    """Renames, permission changes, and file/directory swaps alter semantics."""
    root = tmp_path / "tree"
    root.mkdir()
    item = root / "entry"
    item.write_text("same")
    original = hash_tree(root)

    item.rename(root / "renamed")
    renamed = hash_tree(root)
    assert renamed != original

    renamed_item = root / "renamed"
    renamed_item.chmod(0o755)
    executable = hash_tree(root)
    assert executable != renamed

    renamed_item.unlink()
    renamed_item.mkdir()
    assert hash_tree(root) != executable


def test_hash_tree_excludes_only_named_top_level_roots(tmp_path):
    """An excluded cache must not hide a similarly named nested task file."""
    root = tmp_path / "tree"
    root.mkdir()
    cache = root / ".cache"
    nested_cache = root / "kept" / ".cache"
    cache.mkdir()
    nested_cache.mkdir(parents=True)
    (cache / "ignored").write_text("one")
    (nested_cache / "kept").write_text("one")

    digest = hash_tree(root, excluded_roots=(".cache",))
    (cache / "ignored").write_text("two")
    assert hash_tree(root, excluded_roots=(".cache",)) == digest

    (nested_cache / "kept").write_text("two")
    assert hash_tree(root, excluded_roots=(".cache",)) != digest


def test_hash_tree_rejects_symlink_resolving_outside_root_without_reading_it(tmp_path):
    """An escaping link must not pull host file bytes into a task digest."""
    root = tmp_path / "tree"
    root.mkdir()
    outside = tmp_path / "host-secret"
    outside.write_text("must-not-be-read")
    (root / "escape").symlink_to(outside)

    with pytest.raises(UnsupportedTaskError, match="symlink.*outside"):
        hash_tree(root)


def test_hash_tree_accepts_contained_symlink_and_hashes_its_target_text(tmp_path):
    """Retargeting a contained symlink must invalidate the compiled task."""
    root = tmp_path / "tree"
    root.mkdir()
    (root / "one").write_text("same")
    (root / "two").write_text("same")
    link = root / "current"
    link.symlink_to("one")
    first = hash_tree(root)

    link.unlink()
    link.symlink_to("two")

    assert hash_tree(root) != first
    assert os.path.islink(link)
