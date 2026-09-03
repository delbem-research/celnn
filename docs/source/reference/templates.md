# Templates and registry API

`celnn.templates` exposes reusable template abstractions, a registry, and the built-in template constants intentionally supported by the package.

```{eval-rst}
.. py:currentmodule:: celnn.templates

.. autoclass:: celnn.templates.Template
   :members:

.. autoclass:: celnn.templates.TemplateRegistry
   :members:

.. autofunction:: celnn.templates.builtin_templates

.. autodata:: celnn.templates.EDGE_DETECTION

.. autodata:: celnn.templates.INVERSION

.. autodata:: celnn.templates.CORNER_DETECTION

.. autodata:: celnn.templates.DIAGONAL_LINE_DETECTION

.. autodata:: celnn.templates.SMOOTHING_DEMO

.. autodata:: celnn.templates.SHARPENING_DEMO

.. autodata:: celnn.templates.NOT_DEMO

.. autodata:: celnn.templates.THRESHOLD_DEMO

.. autodata:: celnn.templates.DIFFUSION_LIKE

.. autodata:: celnn.templates.LOCAL_EXCITATION_GLOBAL_DAMPING_DEMO

.. autodata:: celnn.templates.ONE_D_DIFFUSION

.. autodata:: celnn.templates.TWO_D_LAPLACIAN_DEMO
```
