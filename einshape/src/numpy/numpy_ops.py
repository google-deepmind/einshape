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

"""Einshape implementation for JAX."""

from typing import Any, Union

from einshape.src import abstract_ops
from einshape.src import backend
import numpy as np


class _NumpyBackend(backend.Backend[np.ndarray]):
  """Numpy implementation of reshaping ops."""

  def reshape(self, x: np.ndarray, op: abstract_ops.Reshape) -> np.ndarray:
    return np.reshape(x, op.shape)

  def transpose(
      self, x: np.ndarray, op: abstract_ops.Transpose)-> np.ndarray:
    return np.transpose(x, axes=op.perm)

  def broadcast(
      self, x: np.ndarray, op: abstract_ops.Broadcast) -> np.ndarray:
    desired_shape = op.transform_shape(x.shape)
    new_axis_indices = tuple(sorted(op.axis_sizes.keys()))
    x = np.expand_dims(x, axis=new_axis_indices)
    return np.broadcast_to(x, desired_shape)


def einshape(
    equation: str,
    value: Union[np.ndarray, Any],
    **index_sizes: int
    ) -> np.ndarray:
  """Reshapes `value` according to the given Shape Equation.

  Args:
    equation: The Shape Equation specifying the index regrouping and reordering.
    value: Input tensor, or tensor-like object.
    **index_sizes: Sizes of indices, where they cannot be inferred
      from `input_shape`.

  Returns:
    Tensor derived from `value` by reshaping as specified by `equation`.
  """
  if not isinstance(value, np.ndarray):
    value = np.array(value)
  return _NumpyBackend().exec(equation, value, value.shape, **index_sizes)
