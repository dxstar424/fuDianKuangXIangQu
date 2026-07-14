from __future__ import annotations

import ast
import copy
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BLOCK_TABLE = ROOT / "vllm/v1/worker/block_table.py"


class _Buffer:
    def __init__(self) -> None:
        self.np = np.zeros((3, 8), dtype=np.int32)
        self.cpu = _FillRecorder()
        self.gpu = _FillRecorder()
        self.copies: list[int] = []

    def copy_to_gpu(self, num_reqs: int) -> None:
        self.copies.append(num_reqs)


class _FillRecorder:
    def __init__(self) -> None:
        self.values: list[int] = []

    def fill_(self, value: int) -> None:
        self.values.append(value)


def _compile_method(name: str):
    tree = ast.parse(BLOCK_TABLE.read_text(encoding="utf-8"))
    block_table = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BlockTable"
    )
    method = next(
        node
        for node in block_table.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method = copy.deepcopy(method)
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace: dict[str, object] = {}
    exec(compile(module, str(BLOCK_TABLE), "exec"), namespace)
    return namespace[name]


class BlockTableDirtyCopyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.methods = {
            name: _compile_method(name)
            for name in (
                "append_row",
                "add_row",
                "move_row",
                "swap_row",
                "commit_block_table",
                "clear",
                "get_cpu_tensor",
                "get_numpy_array",
            )
        }

    def make_table(self):
        table = types.SimpleNamespace(
            block_table=_Buffer(),
            num_blocks_per_row=np.zeros(3, dtype=np.int32),
            use_hybrid_blocks=False,
            _block_table_dirty=False,
        )
        table.append_row = types.MethodType(self.methods["append_row"], table)
        return table

    def test_unchanged_table_skips_redundant_h2d_copy(self) -> None:
        table = self.make_table()
        self.methods["commit_block_table"](table, 1)
        self.assertEqual(table.block_table.copies, [])

        self.methods["append_row"](table, [4], 0)
        self.methods["commit_block_table"](table, 1)
        self.methods["commit_block_table"](table, 1)
        self.assertEqual(table.block_table.copies, [1])
        self.assertFalse(table._block_table_dirty)

    def test_every_row_mutation_marks_table_dirty(self) -> None:
        mutations = (
            lambda table: self.methods["add_row"](table, [1], 0),
            lambda table: self.methods["move_row"](table, 0, 1),
            lambda table: self.methods["swap_row"](table, 0, 1),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                table = self.make_table()
                table.num_blocks_per_row[0] = 1
                table.block_table.np[0, 0] = 7
                mutate(table)
                self.assertTrue(table._block_table_dirty)

    def test_clear_synchronizes_buffers_and_resets_dirty_state(self) -> None:
        table = self.make_table()
        table._block_table_dirty = True
        self.methods["clear"](table)
        self.assertEqual(table.block_table.gpu.values, [0])
        self.assertEqual(table.block_table.cpu.values, [0])
        self.assertFalse(table._block_table_dirty)

    def test_mutable_cpu_views_conservatively_mark_table_dirty(self) -> None:
        for getter in ("get_cpu_tensor", "get_numpy_array"):
            with self.subTest(getter=getter):
                table = self.make_table()
                self.methods[getter](table)
                self.assertTrue(table._block_table_dirty)


if __name__ == "__main__":
    unittest.main()
