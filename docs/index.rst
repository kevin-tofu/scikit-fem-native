skfem-native
============

**Native assembly. Python formulations.**

``skfem-native`` provides the ``skfemntv`` backend for finite-element
applications that want native assembly performance without giving up the
clarity and adaptability of Python weak forms.

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

   .. grid-item-card:: Read the rationale
      :link: development
      :link-type: doc

      Learn what belongs in the backend and what remains in the application.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting-started
   assembly
   development
