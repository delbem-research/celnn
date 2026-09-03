# Templates and registry API

`celnn.templates` exposes reusable template abstractions, a registry, and the built-in template constants intentionally supported by the package.

`Template` and `TemplateRegistry` are public aliases of the same classes exposed at the top level. Their full generated class contracts live in {doc}`core`; this page indexes the `celnn.templates` aliases without creating duplicate implementation targets.

```{eval-rst}
.. py:currentmodule:: celnn.templates

.. py:class:: Template

   Public alias of :class:`celnn.Template`.

.. py:class:: TemplateRegistry

   Public alias of :class:`celnn.TemplateRegistry`.

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
