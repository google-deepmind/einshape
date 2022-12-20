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

"""Tests for the einshape engine."""

from absl.testing import absltest
from einshape.src.pytorch import pytorch_ops
import torch
import numpy as np


class EngineTest(absltest.TestCase):
  def test_simple_reshape(self):
    x = torch.tensor([3, 5])
    y = pytorch_ops.einshape('i->i1', x)
    np.testing.assert_array_equal(np.array([[3], [5]]), y)

  def test_simple_transpose(self):
    x = torch.tensor([[7, 2, 4], [1, -3, 5]])
    y = pytorch_ops.einshape('ij->ji', x)
    np.testing.assert_array_equal(np.array([[7, 1], [2, -3], [4, 5]]), y)

  def test_ungroup(self):
    x = torch.tensor([1, 4, 7, -2, 3, 2])
    y = pytorch_ops.einshape('(ij)->ij', x, j=3)
    np.testing.assert_array_equal(np.array([[1, 4, 7], [-2, 3, 2]]), y)

  def test_tile_leading_dim(self):
    x = torch.tensor([3, 5])
    y = pytorch_ops.einshape('j->nj', x, n=3)
    np.testing.assert_array_equal(np.array([[3, 5], [3, 5], [3, 5]]), y)

  def test_tile_trailing_dim(self):
    x = torch.tensor([3, 5])
    y = pytorch_ops.einshape('j->jk', x, k=3)
    np.testing.assert_array_equal(np.array([[3, 3, 3], [5, 5, 5]]), y)

  def test_accepts_python_list(self):
    x = [3, 5]  # Python list, not a JAX tensor.
    y = pytorch_ops.einshape('i->i1', x)
    np.testing.assert_array_equal(np.array([[3], [5]]), y)


if __name__ == '__main__':
  absltest.main()
