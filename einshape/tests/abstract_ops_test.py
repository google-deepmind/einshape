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

"""Tests for einshape.abstract_ops."""

from absl.testing import absltest
from einshape.src import abstract_ops


class AbstractOpsTest(absltest.TestCase):

  def test_reshape_transform(self):
    op = abstract_ops.Reshape(shape=[1, 2, 3])
    input_shape = [3, 2]
    output_shape = op.transform_shape(input_shape)
    self.assertListEqual(output_shape, [1, 2, 3])

  def test_transpose_transform(self):
    op = abstract_ops.Transpose(perm=[1, 0, 2])
    input_shape = [4, 5, 6]
    output_shape = op.transform_shape(input_shape)
    self.assertListEqual(output_shape, [5, 4, 6])

  def test_broadcast_transform(self):
    op = abstract_ops.Broadcast(axis_sizes={1: 5, 3: 6})
    input_shape = [2, 2, 2]
    output_shape = op.transform_shape(input_shape)
    self.assertListEqual(output_shape, [2, 5, 2, 6, 2])

if __name__ == '__main__':
  absltest.main()
