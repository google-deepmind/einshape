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

"""Einshape: DSL-based reshaping library for JAX and other frameworks.

The `jnp.einsum` op provides a DSL-based unified interface to matmul and
tensordot ops.
This `einshape` library is designed to offer a similar DSL-based approach
to unifying reshape, squeeze, expand_dims, and transpose operations.

See `src/jax/jax_ops.py` for the JAX implementation of the `einshape` function.
Alternatively, the parser and engine are exposed in `src/engine.py` allowing
analogous implementations in other frameworks; see TensorFlow implementation
at `src/tensorflow/tf_ops.py`.
"""

import typing
from typing import Any, Union
from einshape.src import abstract_ops
from einshape.src import backend

if typing.TYPE_CHECKING:
  import jax.numpy as jnp
  import numpy as np
  import tensorflow as tf


def jax_einshape(
    equation: str,
    value: Union['jnp.ndarray', Any],
    **index_sizes: int,
) -> 'jnp.ndarray':
  """Reshapes `value` according to the given Shape Equation.

  Args:
    equation: The Shape Equation specifying the index regrouping and reordering.
    value: Input tensor, or tensor-like object.
    **index_sizes: Sizes of indices, where they cannot be inferred from
      `input_shape`.

  Returns:
    Tensor derived from `value` by reshaping as specified by `equation`.
  """
  import einshape.src.jax.jax_ops  # pylint:disable=g-import-not-at-top
  global jax_einshape
  jax_einshape = einshape.src.jax.jax_ops.einshape  # pytype:disable=module-attr
  return jax_einshape(equation, value, **index_sizes)


def numpy_einshape(
    equation: str,
    value: Union['np.ndarray', Any],
    **index_sizes: int,
) -> 'np.ndarray':
  """Reshapes `value` according to the given Shape Equation.

  Args:
    equation: The Shape Equation specifying the index regrouping and reordering.
    value: Input tensor, or tensor-like object.
    **index_sizes: Sizes of indices, where they cannot be inferred from
      `input_shape`.

  Returns:
    Tensor derived from `value` by reshaping as specified by `equation`.
  """
  import einshape.src.numpy.numpy_ops  # pylint:disable=g-import-not-at-top
  global numpy_einshape
  numpy_einshape = einshape.src.numpy.numpy_ops.einshape  # pytype:disable=module-attr
  return numpy_einshape(equation, value, **index_sizes)


def tf_einshape(
    equation: str,
    x: Union['tf.Tensor', Any],
    **index_sizes: Union[int, 'tf.Tensor'],
) -> 'tf.Tensor':
  """Reshapes `x` according to the given Shape Equation.

  Args:
    equation: The Shape Equation specifying the index regrouping and reordering.
    x: Input tensor, or tensor-like object.
    **index_sizes: Sizes of indices, where they cannot be inferred from
      `input_shape`.

  Returns:
    Tensor derived from `x` by reshaping as specified by `equation`.
  """
  import einshape.src.tensorflow.tf_ops  # pylint:disable=g-import-not-at-top
  global tf_einshape
  tf_einshape = einshape.src.tensorflow.tf_ops.einshape  # pytype:disable=module-attr
  return tf_einshape(equation, x, **index_sizes)
