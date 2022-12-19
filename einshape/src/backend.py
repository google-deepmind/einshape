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

"""Abstract class of backend of einshape."""

import abc

from typing import Any, Generic, Sequence, TypeVar

from einshape.src import abstract_ops
from einshape.src import engine
from einshape.src import optimizer

T = TypeVar('T')


class Backend(Generic[T], metaclass=abc.ABCMeta):
  """The backend exec the einshape expression on a Tensor."""

  @abc.abstractmethod
  def reshape(self, x: T, op: abstract_ops.Reshape) -> T:
    """The realization of `abstract_ops.Reshape` in a defined backend."""

  @abc.abstractmethod
  def transpose(self, x: T, op: abstract_ops.Transpose) -> T:
    """The realization of `abstract_ops.Transpose` in a defined backend."""

  @abc.abstractmethod
  def broadcast(self, x: T, op: abstract_ops.Broadcast) -> T:
    """The realization of `abstract_ops.Broadcast` in a defined backend."""

  def _preprocess(
      self, ops: Sequence[abstract_ops.ShapeOp], input_shape: Sequence[Any]
      ) -> Sequence[abstract_ops.ShapeOp]:
    """Returns an amended copy of the given sequence of shaping ops.

    This default implementation has no effect, but this is an opportunity for
    backend-specific optimisation to be performed prior to the 'main'
    optimisation.

    Args:
      ops: Sequence of shaping ops.
      input_shape: Shape of the original inputs.

    Returns: Amended sequence of shaping ops.
    """
    return ops

  def exec(
      self,
      equation: str,
      value: T,
      shape: Sequence[Any],
      **index_sizes: Any) -> T:
    """Run the ops defined by the einshape equation.

    Args:
      equation: The Shape Equation specifying how to reshape.
      value: Tensor to reshape.
      shape: Shape of `value`. May be a mixture of `int`s (for statically known
        shape components) and tensors (for dynamic shapes).
      **index_sizes: Sizes of indices, where they cannot be
        inferred from `shape`.

    Returns:
      Reshaped tensor.
    """
    ops = engine.generate_ops(equation, shape, **index_sizes)
    ops = self._preprocess(ops, shape)
    ops = optimizer.optimize(ops, shape)
    for op in ops:
      value = op.execute(self, value)
    return value
