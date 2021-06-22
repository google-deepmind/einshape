# coding=utf-8
# Copyright 2021 DeepMind Technologies Limited.
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

"""Tests for the einshape engine's optimiser."""

from absl.testing import absltest
from einshape.src import abstract_ops
from einshape.src import optimizer


class OptimizerTest(absltest.TestCase):

  def test_redundant_reshape_skipped(self):
    ops = [abstract_ops.Reshape(shape=(2, 3)),
           abstract_ops.Reshape(shape=(6,))]
    input_shape = (3, 2)
    opt_ops = optimizer.optimize(ops, input_shape)

    self.assertLen(opt_ops, 1)
    self.assertEqual(opt_ops[0], ops[1])

  def test_nonredundant_reshape_retained(self):
    ops = [abstract_ops.Reshape(shape=(2, 3)),
           abstract_ops.Transpose(perm=(1, 0)),
           abstract_ops.Reshape(shape=(6,))]
    input_shape = (3, 2)
    opt_ops = optimizer.optimize(ops, input_shape)

    self.assertLen(opt_ops, 3)
    self.assertEqual(opt_ops[0], ops[0])
    self.assertEqual(opt_ops[1], ops[1])
    self.assertEqual(opt_ops[2], ops[2])

  def test_noop_reshape_skipped(self):
    ops = [abstract_ops.Reshape(shape=(3, 5))]
    input_shape = (3, 5)
    opt_ops = optimizer.optimize(ops, input_shape)

    self.assertEmpty(opt_ops)

  def test_op_reshape_retained(self):
    ops = [abstract_ops.Reshape(shape=(5, 3))]
    input_shape = (3, 5)
    opt_ops = optimizer.optimize(ops, input_shape)

    self.assertLen(opt_ops, 1)
    self.assertEqual(opt_ops[0], ops[0])

  def test_noop_reshape_after_tanspose_skipped(self):
    ops = [abstract_ops.Transpose(perm=(1, 0)),
           abstract_ops.Reshape(shape=(5, 3))]
    input_shape = (3, 5)
    opt_ops = optimizer.optimize(ops, input_shape)

    self.assertLen(opt_ops, 1)
    self.assertEqual(opt_ops[0], ops[0])

  def test_noop_reshape_after_broadcast_skipped(self):
    ops = [abstract_ops.Broadcast(axis_sizes={1: 3}),
           abstract_ops.Reshape(shape=(5, 3))]
    input_shape = (5,)
    opt_ops = optimizer.optimize(ops, input_shape)

    self.assertLen(opt_ops, 1)
    self.assertEqual(opt_ops[0], ops[0])

  def test_nonstatic_reshape_retained(self):
    ops = [abstract_ops.Reshape(shape=(5, None))]
    input_shape = (5, None)
    opt_ops = optimizer.optimize(ops, input_shape)

    self.assertLen(opt_ops, 1)
    self.assertEqual(opt_ops[0], ops[0])

  def test_transpose_same_shape_retained(self):
    ops = [abstract_ops.Transpose(perm=(1, 0))]
    input_shape = (3, 3)
    opt_ops = optimizer.optimize(ops, input_shape)

    self.assertLen(opt_ops, 1)
    self.assertEqual(opt_ops[0], ops[0])


if __name__ == '__main__':
  absltest.main()
