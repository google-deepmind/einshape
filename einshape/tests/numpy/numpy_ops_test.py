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

from typing import Any, Mapping, Sequence
from absl.testing import absltest
from absl.testing import parameterized
from einshape.src.numpy import numpy_ops
from einshape.tests import test_cases
import numpy as np


class EngineTest(parameterized.TestCase):

  @parameterized.named_parameters(test_cases.TEST_CASES)
  def test_common(self, x: Sequence[Any], equation: str,
                  index_sizes: Mapping[str, int], expected: Sequence[Any]):
    x = np.array(x)
    y = numpy_ops.einshape(equation, x, **index_sizes)
    np.testing.assert_array_equal(np.array(expected), y)

  def test_accepts_python_list(self):
    x = [3, 5]  # Python list, not a JAX tensor.
    y = numpy_ops.einshape('i->i1', x)
    np.testing.assert_array_equal(np.array([[3], [5]]), y)


if __name__ == '__main__':
  absltest.main()
