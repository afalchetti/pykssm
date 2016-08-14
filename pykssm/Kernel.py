#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# Kernel.py
# Reproducing Kernel Hilbert Space descriptor
# Part of pykssm
# 
# Copyright (c) 2016, Angelo Falchetti
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np

class Kernel(object):
	"Tag class for kernel descriptors."
	
	def __init__(self, setparams):
		"""Construct a kernel space descriptor.
		
		Args:
			setparams: if true, the kernel will be deemed
			           completely parametrized.
		"""
		
		self._setparams = setparams
	
	def complete_params(self):
		"True if all parameters have been correctly set."
		return self._setparams
	
	def __call__(self, a, b):
		""" Calculate the kernel product between two vectors a and b.
		
		Args:
			a: first vector.
			b: second vector.
		Returns:
			Kernel product K(a, b).
		"""
		
		raise NotImplementedError
	
	def deviation(self):
		"Estimate a step size for MCMC given the parameters of the kernel"
		
		raise NotImplementedError
	
	def mixture_eval(self, weights, components, point):
		"Evaluate a kernel mixture at a given point."
		
		return sum(weight * self(component, point) for (weight, component) in zip(weights, components))
	
	def estimate_params(self, vectors):
		"""Propose the kernel hilbert space parameters from sample vectors.
		
		Args:
			vectors: List of sample vectors in state-space.
		"""
		
		self._setparams = True

class LinearKernel(Kernel):
	"Linear kernel descriptor."
	
	def __init__(self):
		"Construct a new linear kernel descriptor."
		
		setparams = True
		
		super().__init__(setparams)
	
	def __call__(self, a, b):
		""" Calculate the linear kernel product between two vectors a and b.
		
		Args:
			a: first vector.
			b: second vector.
		Returns:
			Linear kernel product K(a, b) = a^T b.
		"""
		
		return a.dot(b)
	
	def deviation(self):
		"Estimate a step size for MCMC given the parameters of the kernel"
		
		return 1.0
		
	def estimate_params(self, vectors):
		"""Propose the kernel hilbert space parameters from sample vectors.
		
		Does nothing as this kernel does not have any parameters.
		
		Args:
			vectors: List of sample vectors in state-space.
		"""
		
		super().estimate_params(vectors)

class GaussianKernel(Kernel):
	"Gaussian kernel descriptor."
	
	def __init__(self, sigma=None):
		"""Construct a new gaussian kernel descriptor.
		
		Args:
			sigma: kernel width.
		"""
		
		setparams       = sigma is not None
		self._sigma     = sigma
		self._invsigma2 = 1.0 / sigma**2 if sigma is not None else None
		
		super().__init__(setparams)
	
	@property
	def sigma(self):
		"Kernel width."
		return self._sigma
	
	@sigma.setter
	def ftransition(self, sigma):
		self._sigma = sigma
	
	def __call__(self, a, b):
		""" Calculate the gaussian kernel product between two vectors a and b.
		
		Args:
			a: first vector.
			b: second vector.
		Returns:
			Gaussian kernel product K(a, b) = exp(-|a-b|^2/sigma).
		"""
		
		return np.exp(-self._invsigma2 * np.linalg.norm(a - b)**2)
	
	def deviation(self):
		"Estimate a step size for MCMC given the parameters of the kernel"
		
		return self._sigma
		
	def estimate_params(self, vectors):
		"""Propose the width sigma for a gaussian kernel from sample vectors.
		
		Fills the sigma parameter with the proposal.
		
		Args:
			vectors: List of sample vectors in state-space.
		"""
	
		diffs = [np.linalg.norm(vectors[i] - vectors[k])
		            for i in range(len(vectors))
		            for k in range(len(vectors)) if i < k]
		
		self._sigma     = np.std(diffs)
		self._invsigma2 = 1.0 / self._sigma**2
		
		super().estimate_params(vectors)
