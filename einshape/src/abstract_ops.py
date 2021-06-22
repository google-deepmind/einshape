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

"""Abstract reshape/transpose ops for einshape.

See `generate_ops()` in `engine.py`. An einshape expression is compiled into
a sequence of Reshape/Transpose operations in the abstract, represented by
the types defined here. Framework-specific back-ends will interpret these
operations into concrete ops such as `jnp.reshape`.
"""

import abc
from typing import Any, Dict, Sequence, TypeVar

import dataclasses


T = TypeVar('T')


class ShapeOp(metaclass=abc.ABCMeta):
  """Abstract base class for framework-neutral ops arising in einshape."""

  @abc.abstractmethod
  def transform_shape(self, input_shape: Sequence[Any]) -> Sequence[Any]:
    """Returns output shape of op given input shape."""
    raise NotImplementedError('Should be implemented by subclasses')

  @abc.abstractmethod
  def execute(self, backend, x: T) -> T:
    """Evaluates this op on a framework-specific backend.

    Args:
      backend: Provides framework-specific implementations of all ops.
      x: Tensor to reshape/transpose.

    Returns:
      Reshaped/transposed tensor.
    """
    raise NotImplementedError('Should be implemented by subclasses')


@dataclasses.dataclass
class Reshape(ShapeOp):
  """Operation to reshape a tensor without reordering its elements.

  Attributes:
    shape: New shape for the tensor.
  """
  shape: Sequence[Any]

  def transform_shape(self, input_shape: Sequence[Any]) -> Sequence[Any]:
    del input_shape
    return self.shape

  def execute(self, backend, x: T) -> T:
    return backend.reshape(x, self)


@dataclasses.dataclass
class Transpose(ShapeOp):
  """Operation to reorder the axes of a tensor.

  Attributes:
    perm: Expresses the permutation to be applied to the axes of a tensor.
      For example, `[0, 2, 1]` denotes transposing the inner-most two axes of
      a 3D tensor.
  """
  perm: Sequence[int]

  def transform_shape(self, input_shape: Sequence[Any]) -> Sequence[Any]:
    return [input_shape[i] for i in self.perm]

  def execute(self, backend, x: T) -> T:
    return backend.transpose(x, self)


@dataclasses.dataclass
class Broadcast(ShapeOp):
  """Operation to broadcast (tile) over new axes.

  Attributes:
    axis_sizes: Sizes of the new axes to insert, keyed by the position of the
      axis in the resulting tensor.
      For example, applying `{1: 5, 3: 6}` to a tensor of shape [2, 2, 2]
      results in a tensor of shape [2, 5, 2, 6, 2].
  """
  axis_sizes: Dict[int, Any]

  def transform_shape(self, input_shape: Sequence[Any]) -> Sequence[Any]:
    output_shape = list(input_shape)
    for axis_position, axis_size in sorted(self.axis_sizes.items()):
      output_shape.insert(axis_position, axis_size)
    return output_shape

  def execute(self, backend, x: T) -> T:
    return backend.broadcast(x, self)
