# Backend API

Backends choose execution strategy, not alternate mathematics. CuPy and Torch
remain optional capabilities; the base documentation build does not treat a
mocked optional import as runtime evidence.

```{eval-rst}
.. py:currentmodule:: celnn.backends

.. autoclass:: celnn.backends.ArrayBackend
   :members:

.. autoclass:: celnn.backends.StencilBackend
   :members:

.. autoclass:: celnn.backends.NumPyBackend
   :members:

.. autodata:: celnn.backends.NUMPY_BACKEND

.. autofunction:: celnn.backends.get_default_backend

.. autoclass:: celnn.backends.CuPyBackend
   :members:

.. autoclass:: celnn.backends.TorchBackend
   :members:

.. autofunction:: celnn.backends.get_backend
```
