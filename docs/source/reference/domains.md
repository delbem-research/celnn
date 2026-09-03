# Domain helpers API

`celnn.domains` exposes lightweight grid and signal helpers. Image-specific helpers live in `celnn.domains.image` and are intentionally imported from that module because Pillow remains optional.

```{eval-rst}
.. py:currentmodule:: celnn.domains

.. autofunction:: celnn.domains.checkerboard_grid

.. autofunction:: celnn.domains.coordinate_grid

.. autofunction:: celnn.domains.impulse_grid

.. autofunction:: celnn.domains.random_grid

.. autofunction:: celnn.domains.generate_sine_wave

.. autofunction:: celnn.domains.generate_noisy_sine

.. autofunction:: celnn.domains.normalize_signal

.. autofunction:: celnn.domains.plot_signal

.. autofunction:: celnn.domains.plot_signal_comparison
```

## Optional image helpers

The following helpers are documented from the direct `celnn.domains.image` module. They are not re-exported through `celnn.domains.__all__`, so they are rendered without adding Python-domain targets to the exported-package inventory. Pillow is required only when image conversion or file I/O needs it.

```{eval-rst}
.. py:currentmodule:: celnn.domains.image

.. autofunction:: celnn.domains.image.normalize_image
   :no-index:

.. autofunction:: celnn.domains.image.denormalize_image
   :no-index:

.. autofunction:: celnn.domains.image.image_to_array
   :no-index:

.. autofunction:: celnn.domains.image.array_to_image
   :no-index:

.. autofunction:: celnn.domains.image.load_grayscale
   :no-index:

.. autofunction:: celnn.domains.image.save_grayscale
   :no-index:
```
