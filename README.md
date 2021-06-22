# Einshape: DSL-based reshaping library for JAX and other frameworks.

The `jnp.einsum` op provides a DSL-based unified interface to matmul and
tensordot ops.
This `einshape` library is designed to offer a similar DSL-based approach
to unifying reshape, squeeze, expand_dims, and transpose operations.

Some examples:

* einshape("n->n111", x) is equivalent to expand_dims(x, axis=1) three times
* einshape("a1b11->ab", x) is equivalent to squeeze(x, axis=[1,3,4])
* einshape("nhwc->nchw", x) is equivalent to transpose(x, perm=[0,3,1,2])
* einshape("mnhwc->(mn)hwc", x) is equivalent to a reshape combining
  the two leading dimensions
* einshape("(mn)hwc->mnhwc", x, n=batch_size) is equivalent to a reshape
  splitting the leading dimension into two, using kwargs (m or n or both) to
  supply the necessary additional shape information
* einshape("mn...->(mn)...", x) combines the two leading dimensions without
  knowing the rank of x
* einshape("n...->n(...)", x) performs a 'batch flatten'
* einshape("ij->ijk", x, k=3) inserts a trailing dimension and tiles along it
* einshape("ij->i(nj)", x, n=3) tiles along the second dimension

See `jax_ops.py` for the JAX implementation of the `einshape` function.
Alternatively, the parser and engine are exposed in `engine.py` allowing
analogous implementations in TensorFlow or other frameworks.

![Einshape Logo](einshape-logo.png)