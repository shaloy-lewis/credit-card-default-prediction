"""Governed scientific modelling package.

The package initializer deliberately stays lightweight. Import public contracts
and dataset interfaces from their defining modules so coverage instrumentation
does not preload NumPy or pandas.
"""
