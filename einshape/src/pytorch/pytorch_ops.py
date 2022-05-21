# coding=utf-8

"""Einshape implementation for PyTorch."""

from typing import Any, Union

from einshape.src import abstract_ops
from einshape.src import backend
import torch


class _JaxBackend(backend.Backend[torch.Tensor]):
  """Jax implementation of reshaping ops."""

  def reshape(self, x: torch.Tensor, op: abstract_ops.Reshape) -> torch.Tensor:
    return x.reshape(op.shape)

  def transpose(
      self, x: torch.Tensor, op: abstract_ops.Transpose)-> torch.Tensor:
    return x.permute(op.perm)

  def broadcast(
      self, x: torch.Tensor, op: abstract_ops.Broadcast) -> torch.Tensor:
    shape = op.transform_shape(x.shape)
    for axis_position in sorted(op.axis_sizes.keys()):
      x = x.unsqueeze(axis_position)
    return x.expand(shape)


def einshape(
    equation: str,
    value: Union[torch.Tensor, Any],
    **index_sizes: int
    ) -> torch.Tensor:
  """Reshapes `value` according to the given Shape Equation.

  Args:
    equation: The Shape Equation specifying the index regrouping and reordering.
    value: Input tensor, or tensor-like object.
    **index_sizes: Sizes of indices, where they cannot be inferred
      from `input_shape`.

  Returns:
    Tensor derived from `value` by reshaping as specified by `equation`.
  """
  if not isinstance(value, torch.Tensor):
    value = torch.tensor(value)
  return _JaxBackend().exec(equation, value, value.shape, **index_sizes)
