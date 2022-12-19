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

"""Optimises a sequence of abstract `ShapeOp`s."""

from typing import Any, Sequence

from einshape.src import abstract_ops


def _are_contiguous_reshapes(ops):
  return len(ops) > 1 and all(
      isinstance(op, abstract_ops.Reshape) for op in ops)


def _elide_intermediate_reshapes(ops):
  return [op for i, op in enumerate(ops)
          if not _are_contiguous_reshapes(ops[i:i+2])]


def _is_static(seq):
  return all(isinstance(x, int) for x in seq)


def _is_noop_reshape(op, input_shape):
  if isinstance(op, abstract_ops.Reshape) and _is_static(input_shape):
    output_shape = op.transform_shape(input_shape)
    if _is_static(output_shape):
      return list(output_shape) == list(input_shape)
  return False


def _elide_noop_reshapes(ops, input_shape):
  ops_filtered = []
  for op in ops:
    if not _is_noop_reshape(op, input_shape):
      ops_filtered.append(op)
    input_shape = op.transform_shape(input_shape)
  return ops_filtered


def optimize(
    ops: Sequence[abstract_ops.ShapeOp],
    input_shape: Sequence[Any]
    ) -> Sequence[abstract_ops.ShapeOp]:
  """Returns an optimised copy of the given sequence of shaping ops.

  For example, operations that have no effect are dropped.

  Args:
    ops: Sequence of shaping ops.
    input_shape: Shape of tensor going into op sequence. May be a mixture of
      `int`s (for statically known shape components) and tensors (for dynamic
      shapes).

  Returns: Optimised sequence of shaping ops.
  """
  ops = _elide_intermediate_reshapes(ops)
  ops = _elide_noop_reshapes(ops, input_shape)
  return ops
