#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# kssm.py
# Core KSSM functions
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

def offline(observations, hsensor, kernels):
	"""Estimate state transition from observation matrix.
	
	Given a state-space model
	X_{t+1} = f(X_t) + W_t
	Y_t     = h(X_t) + V_t,
	where both the sensor function h() and a observations Y_{1:t} are known
	and update and observation are corrupted with noises W_t and V_t,
	estimate the probability distribution of the state transition function f()
	parametrized in the reproducing kernel hilbert space spanned by the
	specified kernels.
	
	Args:
		observations: matrix of observations, each row is an observation,
		              as a numpy array.
		hsensor: sensor function, given the internal state,
		         gives the uncorrupted observation.
		kernels: KernelSpace describing the domain of the estimation.
	Returns:
		Tuple ([a_i], s) of mixing parameters and support vectors.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given KernelSpace.
		The particles a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
	"""
	pass

def offline_stream(observations, hsensor, kernels):
	"""Estimate state transition from observation stream.
	
	Given a state-space model
	X_{t+1} = f(X_t) + W_t
	Y_t     = h(X_t) + V_t,
	where both the sensor function h() and a observations Y_{1:t} are known
	and update and observation are corrupted with noises W_t and V_t,
	estimate the probability distribution of the state transition function f()
	parametrized in the reproducing kernel hilbert space spanned by the
	specified kernels.
	
	Args:
		observations: generator of observations, each entry is an observation
		              as a numpy array.
		hsensor: sensor function, given the internal state,
		         gives the uncorrupted observation.
		kernels: KernelSpace describing the domain of the estimation.
	Returns:
		Tuple ([a_i], s) of mixing parameters and support vectors.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given KernelSpace.
		The particles a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
	"""
	
	for 
	pass

def online(observations, hsensor, kernels):
	"""Estimate time-varying state transition from observation matrix.
	
	Given a state-space model
	X_{t+1} = f_t(X_t) + W_t
	Y_t     = h(X_t) + V_t,
	where both the sensor function h() and a observations Y_{1:t} are known
	and update and observation are corrupted with noises W_t and V_t,
	estimate the probability distribution of the state transition function f()
	parametrized in the reproducing kernel hilbert space spanned by the
	specified kernels.
	
	Args:
		observations: matrix of observations, each row is an observation,
		              as a numpy array.
		hsensor: sensor function, given the internal state,
		         gives the uncorrupted observation.
		kernels: KernelSpace describing the domain of the estimation.
	Returns:
		Tuple ([a_i], s) of mixing parameters and support vectors.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given KernelSpace.
		The particles a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
	"""
	pass

def online_stream(observations, hsensor, kernels):
	"""Estimate time-varying state transition from observation stream.
	
	Given a state-space model
	X_{t+1} = f_t(X_t) + W_t
	Y_t     = h(X_t) + V_t,
	where both the sensor function h() and a observations Y_{1:t} are known
	and update and observation are corrupted with noises W_t and V_t,
	estimate the probability distribution of the state transition function f()
	parametrized in the reproducing kernel hilbert space spanned by the
	specified kernels.
	
	Args:
		observations: generator of observations, each entry is an observation
		              as a numpy array.
		hsensor: sensor function, given the internal state,
		         gives the uncorrupted observation.
		kernels: KernelSpace describing the domain of the estimation.
	Returns:
		Tuple ([a_i], s) of mixing parameters and support vectors.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given KernelSpace.
		The particles a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
	"""
	pass

def _getsupportvectors(observations, hsensor):
	"""Propose a set of support vectors for the state from observations Y_{1:t}.
	
	Args:
		observations: matrix of observations, each row is an observation.
		hsensor: sensor function, given the internal state,
		         gives the uncorrupted observation.
	Returns:
		List of support vectors s_i in state-space which adequately cover
		the preimages of the observations.
	"""
	pass

# this may belong inside the gaussian kernel descriptor
# FUTURE generalize into a _estimateparams(Kernel)
def _getwidth(svectors):
	"""Propose the width sigma for a gaussian kernel from the support vectors.
	
	Args:
		svectors: List of support vectors in state-space.
	Returns:
		Scalar kernel width.
	"""
	pass
