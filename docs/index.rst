skfem-native: native assembly for scikit-fem workflows
=======================================================

.. meta::
   :description: skfem-native is an independent native finite-element assembly engine for scikit-fem-style Python workflows, with accelerated assembly, quadrature, geometry, and sparse scatter.

**Native assembly. Python formulations.**

``skfem-native`` is an independent native finite-element assembly engine for
`scikit-fem <https://scikit-fem.readthedocs.io/>`_-style Python workflows.  It
accelerates reusable assembly, geometry, quadrature, and sparse-scatter kernels
without giving up the clarity and adaptability of Python weak forms.

The distribution is imported as ``skfemntv``.  It is not an official
scikit-fem project, distribution, or backend.  See the
:doc:`scikit-fem compatibility boundary <scikit-fem-compatibility>` before
selecting it for an existing application.

.. code-block:: bash

   python -m pip install skfem-native

.. grid:: 1 2 2 3

   .. grid-item-card:: Start assembling
      :link: getting-started
      :link-type: doc

      Install the package and assemble a first bilinear form.

   .. grid-item-card:: Understand the API
      :link: assembly
      :link-type: doc

      Select skfem or skfemntv without moving formulations out of Python.

   .. grid-item-card:: Check compatibility
      :link: scikit-fem-compatibility
      :link-type: doc

      Compare the supported API subset, DOF ordering, and external solver policy.

   .. grid-item-card:: Inspect performance
      :link: scikit-fem-performance
      :link-type: doc

      Read the reproducible scikit-fem assembly benchmark methodology.

   .. grid-item-card:: Read the rationale
      :link: development
      :link-type: doc

      Learn what belongs in the backend and what remains in the application.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting-started
   assembly
   scikit-fem-compatibility
   scikit-fem-performance
   gallery
   development
