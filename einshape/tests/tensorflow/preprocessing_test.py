# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the TensorFlow-specific preprocessing or einshape ops."""

from absl.testing import absltest
from einshape.src import abstract_ops
from einshape.src.tensorflow import preprocessing


class PreprocessingTest(absltest.TestCase):

  def test_reshapes_and_transposes_preserved(self):
    ops = [
        abstract_ops.Reshape(shape=(2, 3)),
        abstract_ops.Transpose(perm=[1, 0]),
        abstract_ops.Reshape(shape=(6,))]
    preproc_ops = preprocessing.preprocess(ops, [6])

    self.assertLen(preproc_ops, 3)
    for op, preproc_op in zip(ops, preproc_ops):
      self.assertEqual(op, preproc_op)

  def test_broadcast_expands_to_tile_then_reshape(self):
    ops = [abstract_ops.Broadcast(axis_sizes={1: 5})]
    preproc_ops = preprocessing.preprocess(ops, [2, 3])

    self.assertLen(preproc_ops, 2)
    self.assertIsInstance(preproc_ops[0], preprocessing.Tile)
    self.assertEqual(preproc_ops[0].multiples, [1, 5])
    self.assertIsInstance(preproc_ops[1], abstract_ops.Reshape)
    self.assertEqual(preproc_ops[1].shape, [2, 5, 3])

  def test_broadcast_trailing_dimension_generates_initial_reshape(self):
    ops = [abstract_ops.Broadcast(axis_sizes={2: 5})]
    preproc_ops = preprocessing.preprocess(ops, [2, 3])

    self.assertLen(preproc_ops, 3)
    self.assertIsInstance(preproc_ops[0], abstract_ops.Reshape)
    self.assertEqual(preproc_ops[0].shape, [2, 3, 1])
    self.assertIsInstance(preproc_ops[1], preprocessing.Tile)
    self.assertEqual(preproc_ops[1].multiples, [1, 1, 5])
    self.assertIsInstance(preproc_ops[2], abstract_ops.Reshape)
    self.assertEqual(preproc_ops[2].shape, [2, 3, 5])

  def test_broadcast_multiple_dimensions(self):
    ops = [abstract_ops.Broadcast(
        axis_sizes={0: 101, 2: 103, 3: 105, 5: 107, 6: 109})]
    preproc_ops = preprocessing.preprocess(ops, [2, 3])

    self.assertLen(preproc_ops, 3)
    self.assertIsInstance(preproc_ops[0], abstract_ops.Reshape)
    self.assertEqual(preproc_ops[0].shape, [2, 3, 1])
    self.assertIsInstance(preproc_ops[1], preprocessing.Tile)
    self.assertEqual(preproc_ops[1].multiples, [101, 103*105, 107*109])
    self.assertIsInstance(preproc_ops[2], abstract_ops.Reshape)
    self.assertEqual(preproc_ops[2].shape, [101, 2, 103, 105, 3, 107, 109])


if __name__ == '__main__':
  absltest.main()
