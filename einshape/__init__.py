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

from einshape.src import abstract_ops
from einshape.src import backend
from einshape.src.jax.jax_ops import einshape
from einshape.src.tensorflow.tf_ops import einshape as tf_einshape
