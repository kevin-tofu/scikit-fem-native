Getting started
===============

Installation
------------

Install the released package from PyPI:

.. code-block:: bash

   python -m pip install skfem-native

The distribution name is ``skfem-native`` and the import package is
``skfemntv``.  Python 3.10 or newer is required.

First assembly
--------------

.. code-block:: python

   import skfemntv
   from skfemntv.helpers import dot, grad

   mesh = skfemntv.MeshTet()
   basis = skfemntv.Basis(mesh, skfemntv.ElementTetP1())

   @skfemntv.BilinearForm
   def diffusion(u, v, w):
       return dot(grad(u), grad(v))

   matrix = skfemntv.asm(diffusion, basis)

Development installation
------------------------

.. code-block:: bash

   git clone https://github.com/kevin-tofu/skfem-native.git
   cd skfem-native
   python -m pip install -e '.[test,docs]'
