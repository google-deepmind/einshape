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

"""Einshape implementation for TensorFlow."""

from typing import Any, Sequence, Union

from einshape.src import abstract_ops
from einshape.src import backend
from einshape.src.tensorflow import preprocessing
import tensorflow as tf


class _TFBackend(backend.Backend[tf.Tensor]):
  """TensorFlow implementation of reshaping ops."""

  def reshape(self, x: tf.Tensor, op: abstract_ops.Reshape) -> tf.Tensor:
    return tf.reshape(x, op.shape)

  def transpose(self, x: tf.Tensor, op: abstract_ops.Transpose) -> tf.Tensor:
    return tf.transpose(x, op.perm)

  def broadcast(self, x: tf.Tensor, op: abstract_ops.Broadcast) -> tf.Tensor:
    # Pre-processing will have removed these.
    raise NotImplementedError()

  def tile(self, x: tf.Tensor, op: preprocessing.Tile) -> tf.Tensor:
    return tf.tile(x, op.multiples)

  def _preprocess(
      self, ops: Sequence[abstract_ops.ShapeOp], input_shape: Sequence[Any]
      ) -> Sequence[abstract_ops.ShapeOp]:
    return preprocessing.preprocess(ops, input_shape)


def einshape(
    equation: str,
    x: Union[tf.Tensor, Any],
    **index_sizes: Union[int, tf.Tensor],
    ) -> tf.Tensor:
  """Reshapes `x` according to the given Shape Equation.

  Args:
    equation: The Shape Equation specifying the index regrouping and reordering.
    x: Input tensor, or tensor-like object.
    **index_sizes: Sizes of indices, where they cannot be inferred
      from `input_shape`.

  Returns:
    Tensor derived from `x` by reshaping as specified by `equation`.
  """
  x = tf.convert_to_tensor(x)
  # Use components of the static shape of `x` where known, falling back on
  # the dynamic shape.
  static_shape = x.shape.as_list()
  dynamic_shape = tf.shape(x)
  known_shape = [
      static_shape[i] if static_shape[i] is not None else dynamic_shape[i]
      for i in range(x.shape.ndims)]
  return _TFBackend().exec(equation, x, known_shape, **index_sizes)
