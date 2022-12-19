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

"""TensorFlow-specific preprocessing of the Broadcast op."""

from typing import Any, Sequence, TypeVar

import dataclasses
from einshape.src import abstract_ops


T = TypeVar('T')


@dataclasses.dataclass
class Tile(abstract_ops.ShapeOp):
  """Operation to tile over existing axes.

  Attributes:
    multiples: Multiple by which tile each axis.
  """
  multiples: Sequence[Any]

  @property
  def is_static(self) -> bool:
    return all(isinstance(mult, int) for mult in self.multiples)

  def transform_shape(self, input_shape: Sequence[Any]) -> Sequence[Any]:
    return [mult * size for size, mult in zip(input_shape, self.multiples)]

  def execute(self, backend, x: T) -> T:
    return backend.tile(x, self)


def preprocess(
    ops: Sequence[abstract_ops.ShapeOp], input_shape: Sequence[Any]
    ) -> Sequence[abstract_ops.ShapeOp]:
  """Returns a copy of the shaping op sequence, adjusted for TensorFlow.

  Specifically, a `Broadcast` op are replaced with a `Tile` with `Reshape`s.

  Args:
    ops: Sequence of shaping ops.
    input_shape: Shape of the original inputs.

  Returns: Updated sequence of shaping ops.
  """
  preproc_ops = []

  shape = input_shape
  for op in ops:
    next_shape = op.transform_shape(shape)
    if isinstance(op, abstract_ops.Broadcast):
      # Replace the Broadcast op with an equivalent Tile op (plus Reshapes).

      # Overall we need to insert a dimension for each broadcast axis.
      # However, as far as possible, we tile over "next existing" dimension
      # instead, and reshape to the desired shape at the end.
      # This gives the greatest opportunity to avoid redundant reshapes.
      # Note that multiple new dimensions may share the same "next existing".

      # Special case: If the Broadcast op appends one or more dimensions, then
      # there is no "next existing" dimension. Here we do need to append one.
      if len(next_shape) - 1 in op.axis_sizes:
        # Insert trailing '1' dimension to tile.
        shape = list(shape) + [1]
        preproc_ops.append(abstract_ops.Reshape(shape))

      # Tile the original dimensions to achieve the broadcasting.
      tile_multiples = [1] * len(shape)
      for j, (axis, axis_size) in enumerate(sorted(op.axis_sizes.items())):
        original_axis = axis - j
        tile_multiples[original_axis] *= axis_size
      preproc_ops.append(Tile(tile_multiples))

      # Reshape to the desired shape.
      preproc_ops.append(abstract_ops.Reshape(next_shape))

    else:
      preproc_ops.append(op)

    shape = next_shape

  return preproc_ops
