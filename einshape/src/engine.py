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

"""Einshape engine."""

import re
import string
from typing import Any, Sequence

from einshape.src import abstract_ops


Indices = str


def _group_dimensions(expr: str) -> Sequence[str]:
  """Splits an expression into its separate grouped dimensions.

  An unqualified dimension index is a group by itself.
  Parentheses are interpreted as demarcating a sequence of dimension indices
  to be grouped into a single dimension.
  '1' is an alias for '()', denoting a dimension of size 1 with no indices.
  Nested parentheses are not permitted.

  Examples:
    'ijk' is grouped as ['i', 'j', 'k']
    '(mn)hwc' is grouped as ['mn', 'h', 'w', 'c']
    'n111' is grouped as ['n', '', '', '']
    'n...' is grouped as ['n', '...'], where '...' stands for multiple groups.

  Args:
    expr: Shape expression to group.

  Returns:
    List of simple expressions, each consisting solely of dimension
    indices, specifying the indices that constitute each grouped dimension.
  """
  groups = []
  i = 0
  while i < len(expr):
    if expr[i].isalpha():
      # Top-level dimension index is a group by itself.
      groups.append(expr[i])
      i += 1

    elif expr[i] == '1':
      # Dimension of size 1 with no indices; equivalent to '()'.
      i += 1
      groups.append('')

    elif expr[i] == '(':
      # Sequence of indices to be grouped as a single dimension.
      i += 1
      group_begin = i
      while i < len(expr) and expr[i].isalpha():
        i += 1
      group_end = i
      if not(i < len(expr) and expr[i] == ')'):
        raise ValueError('Unclosed parenthesis')
      i += 1
      groups.append(expr[group_begin:group_end])

    elif expr[i:].startswith('...'):
      # Wildcard sequence of dimensions.
      i += len('...')
      if '...' in groups:
        raise ValueError('Wildcard "..." may only occur once')
      groups.append('...')

    else:
      raise ValueError(f'Illegal character: {ord(expr[i])}')

  return groups


def _expand_wildcard(expr: str, num_wildcard_dims: int) -> str:
  """Replaces a wildcard with the correct number of axes.

  Args:
    expr: Shape expression containing '...'.
    num_wildcard_dims: Number of dimensions captured by the wildcard.

  Returns:
    Copy of `expr` with the wildcard '...' replaced by indices 'A', 'B', ...
  """
  # Introduce indices A, B, C, ... for the wildcard.
  if num_wildcard_dims > len(string.ascii_uppercase):
    raise ValueError('Too many dimensions')
  expanded_wildcard = string.ascii_uppercase[:num_wildcard_dims]
  return expr.replace('...', expanded_wildcard)


def generate_ops(
    equation: str, input_shape: Sequence[Any], **index_sizes: Any
    ) -> Sequence[abstract_ops.ShapeOp]:
  """Compiles a Shape Equation into Reshape/Transpose ops.

  Args:
    equation: The Shape Equation specifying the index regrouping and reordering.
    input_shape: Shape of the input tensor. May be a mixture of `int`s (for
      statically known shape components) and tensors (for dynamic shapes).
    **index_sizes: Sizes of indices, where they cannot be inferred
      from `input_shape`.

  Returns:
    Unoptimised list of `Reshape` and `Transpose` instructions that abstractly
      specify what concrete tf/np/jax operations should be applied.
  """
  # Upper-case indices are reserved for handling '...' wildcards.
  if equation != equation.lower():
    raise ValueError('Shape Equation may not use upper-case indices')

  expression_list = equation.split('->')
  if len(expression_list) != 2:
    raise ValueError('Shape Equation requires a single "->"')
  lhs, rhs = expression_list

  # Understand how the input indices are grouped.
  lhs_grouped = _group_dimensions(lhs)

  if '...' in lhs_grouped:
    if len(input_shape) < len(lhs_grouped) - 1:
      raise ValueError(
          'Input shape must have rank matching LHS. '
          f'LHS expects rank >= {len(lhs_grouped) - 1} '
          f'but actual input has rank {len(input_shape)}')
    num_wildcard_dims = len(input_shape) - (len(lhs_grouped) - 1)
    # Update the shape expressions, and parse afresh.
    lhs = _expand_wildcard(lhs, num_wildcard_dims)
    rhs = _expand_wildcard(rhs, num_wildcard_dims)
    lhs_grouped = _group_dimensions(lhs)

  else:
    if len(input_shape) != len(lhs_grouped):
      raise ValueError(
          'Input shape must have rank matching LHS. '
          f'LHS expects rank {len(lhs_grouped)} '
          f'but actual input has rank {len(input_shape)}')

  # Determine all indices' sizes, inferred from `input_shape` along with
  # with additional hints from `input_sizes`.
  inferred_index_sizes = {**index_sizes}
  for j, group in enumerate(lhs_grouped):
    # There is permitted to be one index in the group without a size specified.
    known_size = 1
    unknown_index = None
    for a in group:
      if a in index_sizes:
        known_size *= index_sizes[a]
      else:
        if unknown_index:
          raise ValueError(
              'All but one of the indices must have their size specified '
              f'when ungrouping dimensions. In group "({group})", '
              f'"{unknown_index}" and "{a}" have unspecified size.')
        unknown_index = a

    if unknown_index:
      # Infer the size of the remaining index, from the input shape information.
      remainder = input_shape[j] % known_size
      if isinstance(remainder, int) and remainder != 0:
        known_indices = group.replace(unknown_index, '')
        raise ValueError(
            'Dimension to ungroup is not divisible by its index sizes. '
            f'Group "({group})" expects size {input_shape[j]}, but its indices '
            f'"{known_indices}" have combined specified size {known_size}.')
      inferred_index_sizes[unknown_index] = input_shape[j] // known_size

    else:
      # All indices are fully specified. Check consistency with input shape,
      # provided both are known statically.
      discrepancy = input_shape[j] - known_size
      if isinstance(discrepancy, int) and discrepancy != 0:
        raise ValueError(
            'Input shape incompatible with index sizes. '
            f'Group "({group})" expects size {input_shape[j]}, '
            f'but its indices have combined specified size {known_size}.')

  ops = []

  # Begin with a reshape op to ungroup everything.
  ungrouped = ''.join(lhs_grouped)
  ungrouped_shape = [inferred_index_sizes[a] for a in ungrouped]
  ops.append(abstract_ops.Reshape(shape=ungrouped_shape))

  # Infer the permutation, from the rhs expression with grouping removed.
  transposed = ''
  broadcast_axis_sizes = {}
  for i, a in enumerate(re.sub('[1()]', '', rhs)):
    if a in ungrouped:
      transposed += a
    else:
      if a not in inferred_index_sizes:
        raise ValueError(f'Broadcast multiples "{a}" must be specified.')
      broadcast_axis_sizes[i] = inferred_index_sizes[a]

  if transposed != ungrouped:
    # Indices in RHS must occur once and only once in LHS, and vice versa.
    if any(ungrouped.count(a) != 1 for a in transposed):
      raise ValueError('Every index in RHS must occur exactly once in LHS')
    if any(rhs.count(a) != 1 for a in ungrouped):
      raise ValueError('Every index in LHS must occur exactly once in RHS')
    perm = [ungrouped.index(a) for a in transposed]
    ops.append(abstract_ops.Transpose(perm=perm))

  if broadcast_axis_sizes:
    ops.append(abstract_ops.Broadcast(axis_sizes=broadcast_axis_sizes))

  # Now regroup as specified by the RHS expression.
  rhs_grouped = _group_dimensions(rhs)

  def alphabetical(group: Indices) -> Indices:
    return ''.join(sorted(group))
  group_size_dict = {alphabetical(group): size
                     for group, size in zip(lhs_grouped, input_shape)}
  def group_size(group):
    prod = 1
    for a in group:
      prod *= inferred_index_sizes[a]
    # If prod is not a static value, check if the group size can be inferred
    # from input_shape by matching LHS groups, e.g.
    #   einshape("(ij)->(ji)", x, j=n),
    # where x.shape is static, but n is not.
    if not isinstance(prod, int) and alphabetical(group) in group_size_dict:
      return group_size_dict[alphabetical(group)]
    return prod

  output_shape = [group_size(group) for group in rhs_grouped]
  ops.append(abstract_ops.Reshape(shape=output_shape))

  # Return the ops in this "ungroup - transpose? - broadcast? - regroup"
  # structure.
  # Note that in many cases this will be less than optimal; the expectation
  # is that a separate optimisation pass will be applied to this abstract
  # representation.
  return ops
