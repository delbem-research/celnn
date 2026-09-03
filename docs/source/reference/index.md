# API Reference

The reference is generated from the installed CELNN package. It documents deliberate public export surfaces rather than arbitrary implementation modules.

A symbol may be available from more than one deliberate public module—for example `Template` is exported both at `celnn.Template` and `celnn.templates.Template`. Each module surface is verified independently. Private implementation names are not promoted to API simply because autodoc can import them.

```{toctree}
:maxdepth: 1

core
activations
templates
differentiable
plasticity
associative
training
backends
domains
io
```
