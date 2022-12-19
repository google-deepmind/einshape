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

"""Tests for the einshape engine."""

from absl.testing import absltest
from einshape.src import abstract_ops
from einshape.src import engine


class _ShapeExpr(object):
  """Mock class to model a dynamic shape component.

  The einshape engine will still perform simple arithmetic on such shapes
  even though their value is not known at graph-time or JIT-time.
  """

  def __add__(self, x):
    return _ShapeExpr()

  def __radd__(self, x):
    return _ShapeExpr()

  def __sub__(self, x):
    return _ShapeExpr()

  def __rsub__(self, x):
    return _ShapeExpr()

  def __mul__(self, x):
    return _ShapeExpr()

  def __rmul__(self, x):
    return _ShapeExpr()

  def __floordiv__(self, x):
    return _ShapeExpr()

  def __rfloordiv__(self, x):
    return _ShapeExpr()

  def __mod__(self, x):
    return _ShapeExpr()

  def __rmod__(self, x):
    return _ShapeExpr()


class EngineTest(absltest.TestCase):

  def test_rank_zero_noop(self):
    preshape, reshape = engine.generate_ops('->', [])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([], list(preshape.shape))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([], list(reshape.shape))

  def test_rank_one_noop(self):
    preshape, reshape = engine.generate_ops('i->i', [3])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([3], list(preshape.shape))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([3], list(reshape.shape))

  def test_rank_four_noop(self):
    preshape, reshape = engine.generate_ops(
        'ijkl->ijkl', [2, 3, 5, 7])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([2, 3, 5, 7], list(preshape.shape))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([2, 3, 5, 7], list(reshape.shape))

  def test_simple_transpose(self):
    preshape, transpose, reshape = engine.generate_ops('ij->ji', [3, 5])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([3, 5], list(preshape.shape))

    self.assertIsInstance(transpose, abstract_ops.Transpose)
    self.assertEqual([1, 0], list(transpose.perm))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([5, 3], list(reshape.shape))

  def test_nchw_transpose(self):
    preshape, transpose, reshape = engine.generate_ops(
        'nhwc->nchw', [5, 13, 17, 3])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([5, 13, 17, 3], list(preshape.shape))

    self.assertIsInstance(transpose, abstract_ops.Transpose)
    self.assertEqual([0, 3, 1, 2], list(transpose.perm))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([5, 3, 13, 17], list(reshape.shape))

  def test_rejects_lhs_mismatch_with_input_shape(self):
    with self.assertRaises(ValueError):
      engine.generate_ops('i->i', [3, 5])

  def test_rejects_lhs_with_duplicated_index(self):
    with self.assertRaises(ValueError):
      engine.generate_ops('ii->i', [3, 3])

  def test_rejects_rhs_with_missing_index(self):
    with self.assertRaises(ValueError):
      engine.generate_ops('ij->j', [3, 5])

  def test_rejects_rhs_with_duplicated_index(self):
    with self.assertRaises(ValueError):
      engine.generate_ops('ij->ii', [3, 5])

  def test_expand_dim(self):
    preshape, reshape = engine.generate_ops('k->1k', [2])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([2], list(preshape.shape))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([1, 2], list(reshape.shape))

  def test_expand_multiple_dims(self):
    preshape, reshape = engine.generate_ops('n->n111', [5])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([5], list(preshape.shape))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([5, 1, 1, 1], list(reshape.shape))

  def test_transpose_and_expand_dim(self):
    preshape, transpose, reshape = engine.generate_ops('ij->j1i', [5, 7])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([5, 7], list(preshape.shape))

    self.assertIsInstance(transpose, abstract_ops.Transpose)
    self.assertEqual([1, 0], list(transpose.perm))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([7, 1, 5], list(reshape.shape))

  def test_squeeze(self):
    preshape, reshape = engine.generate_ops('j1k->jk', [2, 1, 3])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([2, 3], list(preshape.shape))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([2, 3], list(reshape.shape))

  def test_rejects_squeezing_non_unitary(self):
    with self.assertRaises(ValueError):
      engine.generate_ops('i1->i', [2, 3])

  def test_squeeze_transpose_expand(self):
    preshape, transpose, reshape = engine.generate_ops(
        'ij11k1->1k1ji', [3, 5, 1, 1, 7, 1])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([3, 5, 7], list(preshape.shape))

    self.assertIsInstance(transpose, abstract_ops.Transpose)
    self.assertEqual([2, 1, 0], list(transpose.perm))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([1, 7, 1, 5, 3], list(reshape.shape))

  def test_group_dims(self):
    preshape, reshape = engine.generate_ops('ij->(ij)', [3, 5])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([3, 5], list(preshape.shape))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([15], list(reshape.shape))

  def test_ungroup_dims(self):
    preshape, reshape = engine.generate_ops(
        '(ij)->ij', [15], i=3, j=5)

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([3, 5], list(preshape.shape))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([3, 5], list(reshape.shape))

  def test_ungroup_dims_with_inferred_size(self):
    preshape, reshape = engine.generate_ops(
        'n(hwc)->nhwc', [5, 429], h=11, w=13)

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([5, 11, 13, 3], list(preshape.shape))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([5, 11, 13, 3], list(reshape.shape))

  def test_rejects_input_sizes_conflicting_with_input_shapes(self):
    with self.assertRaises(ValueError):
      engine.generate_ops('(ij)->ij', [15], i=2, j=7)

  def test_rejects_ungroup_with_non_divisor_dimension_size(self):
    with self.assertRaises(ValueError):
      engine.generate_ops('(ij)->ij', [15], j=4)

  def test_rejects_underspecified_ungroup_dimensions(self):
    with self.assertRaises(ValueError):
      engine.generate_ops('(ij)->ij', [15])
    with self.assertRaises(ValueError):
      engine.generate_ops('(ijk)->ijk', [15], i=3)

  def test_regroup_with_transpose(self):
    preshape, transpose, reshape = engine.generate_ops(
        'i(jk)->k(ji)', [3, 35], j=5)

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([3, 5, 7], list(preshape.shape))

    self.assertIsInstance(transpose, abstract_ops.Transpose)
    self.assertEqual([2, 1, 0], list(transpose.perm))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([7, 15], list(reshape.shape))

  def test_wildcard_identity(self):
    preshape, reshape = engine.generate_ops('...->...', [3, 5])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([3, 5], list(preshape.shape))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([3, 5], list(reshape.shape))

  def test_wildcard_transpose(self):
    preshape, transpose, reshape = engine.generate_ops(
        'i...j->j...i', [2, 3, 5, 7])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([2, 3, 5, 7], list(preshape.shape))

    self.assertIsInstance(transpose, abstract_ops.Transpose)
    self.assertEqual([3, 1, 2, 0], list(transpose.perm))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([7, 3, 5, 2], list(reshape.shape))

  def test_rejects_wildcard_lhs_mismatch_with_input_shape(self):
    with self.assertRaises(ValueError):
      engine.generate_ops('ijk...->ijk...', [3, 5])

  def test_wildcard_flatten(self):
    preshape, reshape = engine.generate_ops(
        'n...->n(...)', [2, 3, 5])

    self.assertIsInstance(preshape, abstract_ops.Reshape)
    self.assertEqual([2, 3, 5], list(preshape.shape))

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertEqual([2, 15], list(reshape.shape))

  def test_rejects_ungrouping_wildcard(self):
    with self.assertRaises(ValueError):
      engine.generate_ops('n(...)->n...', [3, 5])

  def test_reshape_unknown_input_shape(self):
    _, reshape = engine.generate_ops('i->1i', [_ShapeExpr()])

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertLen(reshape.shape, 2)
    self.assertEqual(1, reshape.shape[0])
    self.assertIsInstance(reshape.shape[1], _ShapeExpr)

  def test_reshape_unknown_index_size(self):
    _, reshape = engine.generate_ops('i->1i', [3], i=_ShapeExpr())

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertLen(reshape.shape, 2)
    self.assertEqual(1, reshape.shape[0])
    self.assertEqual(3, reshape.shape[1])

  def test_smart_shape_inference(self):
    _, _, reshape = engine.generate_ops('(ij)->(ji)', [6], j=_ShapeExpr())

    self.assertIsInstance(reshape, abstract_ops.Reshape)
    self.assertLen(reshape.shape, 1)
    self.assertEqual(6, reshape.shape[0])

  # Broadcast
  def test_tile_leading_dim(self):
    _, broadcast, _ = engine.generate_ops('j->nj', [5], n=3)

    self.assertIsInstance(broadcast, abstract_ops.Broadcast)
    self.assertDictEqual(broadcast.axis_sizes, {0: 3})

  def test_tile_trailing_dim(self):
    _, broadcast, _ = engine.generate_ops('j->jk', [3], k=2)

    self.assertIsInstance(broadcast, abstract_ops.Broadcast)
    self.assertDictEqual(broadcast.axis_sizes, {1: 2})

  def test_tile_leading_dim_and_flatten(self):
    _, broadcast, reshape = engine.generate_ops('j->(nj)', [5], n=3)

    self.assertIsInstance(broadcast, abstract_ops.Broadcast)
    self.assertDictEqual(broadcast.axis_sizes, {0: 3})
    self.assertLen(reshape.shape, 1)
    self.assertEqual(15, reshape.shape[0])

  def test_tile_trailing_dim_and_flatten(self):
    _, broadcast, reshape = engine.generate_ops('j->(jk)', [3], k=2)

    self.assertIsInstance(broadcast, abstract_ops.Broadcast)
    self.assertDictEqual(broadcast.axis_sizes, {1: 2})
    self.assertLen(reshape.shape, 1)
    self.assertEqual(6, reshape.shape[0])

  def test_tile_rank_two_one_dim(self):
    _, broadcast, _ = engine.generate_ops('ij->inj', [2, 5], n=3)

    self.assertIsInstance(broadcast, abstract_ops.Broadcast)
    self.assertDictEqual(broadcast.axis_sizes, {1: 3})

  def test_tile_rank_two_one_dim_and_flatten(self):
    _, broadcast, reshape = engine.generate_ops('ij->i(nj)', [2, 5], n=3)

    self.assertIsInstance(broadcast, abstract_ops.Broadcast)
    self.assertDictEqual(broadcast.axis_sizes, {1: 3})
    self.assertLen(reshape.shape, 2)
    self.assertEqual(2, reshape.shape[0])
    self.assertEqual(15, reshape.shape[1])

  def test_tile_rank_two_one_dim_with_transpose(self):
    _, transpose, broadcast, _ = engine.generate_ops('ij->nji', [2, 5], n=3)

    self.assertIsInstance(transpose, abstract_ops.Transpose)
    self.assertEqual([1, 0], list(transpose.perm))
    self.assertIsInstance(broadcast, abstract_ops.Broadcast)
    self.assertDictEqual(broadcast.axis_sizes, {0: 3})

  def test_tile_rank_two_two_dims(self):
    _, broadcast, _ = engine.generate_ops('ij->nikj', [2, 5], n=3, k=4)

    self.assertIsInstance(broadcast, abstract_ops.Broadcast)
    self.assertDictEqual(broadcast.axis_sizes, {0: 3, 2: 4})

  def test_rejcts_unknown_tile_multiples(self):
    with self.assertRaises(ValueError):
      engine.generate_ops('j->nj', [15])


if __name__ == '__main__':
  absltest.main()
