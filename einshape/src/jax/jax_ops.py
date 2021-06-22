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

"""Einshape implementation for JAX."""

from typing import Any, Union

from einshape.src import abstract_ops
from einshape.src import backend
from jax import lax
import jax.numpy as jnp


class _JaxBackend(backend.Backend[jnp.ndarray]):
  """Jax implementation of reshaping ops."""

  def reshape(self, x: jnp.ndarray, op: abstract_ops.Reshape) -> jnp.ndarray:
    return jnp.reshape(x, op.shape)

  def transpose(
      self, x: jnp.ndarray, op: abstract_ops.Transpose)-> jnp.ndarray:
    return jnp.transpose(x, axes=op.perm)

  def broadcast(
      self, x: jnp.ndarray, op: abstract_ops.Broadcast) -> jnp.ndarray:
    shape = op.transform_shape(x.shape)
    # For each input dimension, lax needs to know which output dimension it
    # corresponds to.
    broadcast_dims = [j for j in range(len(shape)) if j not in op.axis_sizes]
    return lax.broadcast_in_dim(x, shape, broadcast_dims)


def einshape(
    equation: str,
    value: Union[jnp.ndarray, Any],
    **index_sizes: int
    ) -> jnp.ndarray:
  """Reshapes `value` according to the given Shape Equation.

  Args:
    equation: The Shape Equation specifying the index regrouping and reordering.
    value: Input tensor, or tensor-like object.
    **index_sizes: Sizes of indices, where they cannot be inferred
      from `input_shape`.

  Returns:
    Tensor derived from `value` by reshaping as specified by `equation`.
  """
  if not isinstance(value, jnp.ndarray):
    value = jnp.array(value)
  return _JaxBackend().exec(equation, value, value.shape, **index_sizes)
