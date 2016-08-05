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

def offline(observations, hsensor, invhsensor, kernel = GaussianKernel(),
            nsamples = 400, snstd = 1.0, snoise = None, svectors = None):
	"""Estimate state transition from observation matrix.
	
	Given a state-space model
	X_{t+1} = f(X_t) + W_t
	Y_t     = h(X_t) + V_t,
	where both the sensor function h() and a observations Y_{1:t} are known
	and update and observation are corrupted with zero-mean noises W_t and V_t,
	estimate the probability distribution of the state transition function f()
	parametrized in the reproducing kernel hilbert space spanned by the
	specified kernels.
	
	Args:
		observations: matrix of observations, each row is an observation,
		              as a numpy array.
		hsensor: sensor stochastic model, as a function that takes
		         x and y and returns a value proportional to p(y|x).
		invhsensor: inverse of the sensor function, h^-1(y); equivalently
		            solves argmax_x(p(x|y)).
		kernel: Kernel object describing the domain of the estimation,
		        if it has incomplete parameters, they will be computed
		        from the observations.
		nsamples: number of samples to draw from the MCMC process.
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		snstd: if snoise is None, additive white gaussian noise will be
		       used with this standard deviation. If both arguments are
		       None, the process will present no noise. 
		snoise: state transition known additive noise process W_t, as
		        a function of the state.
	Returns:
		Tuple ([a_i], s) of mixing parameters and support vectors.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given Kernel space.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
	"""
	
	if svectors is None:
		svectors = _getsupportvectors(observations, invhsensor)
		
	ssize = len(svectors)
	
	if snoise is None:
		if snstd is None:
			snoise lambda s: np.zeros(ssize)
		else:
			snoise = lambda s: np.randn(ssize)
	if not kernel.complete_params:
		kernel.estimate_params(svectors)
	
	firststate = invhsensor(observations[0])
	
	sampler = PMCMC(observations[1:],  # the first observation was used for the SMC state prior
	                initial       = np.array.zeros(ssize),
	                prior         = lambda s: 1.0,
	                proposer      = lambda s: s + kernel.dev.dot(np.randn(ssize)),
	                smcprior      = lambda s: firststate,
	                ftransitioner = lambda a: lambda s: sum(a[i] * kernel(svectors[i], s) for i in range(len(a))),
	                hsensor       = hsensor,
	                nsamples      = 400)
	
	samples = [sampler.draw() for i in range(nsamples)]
	
	return (samples, svectors)

def offline_stream(observations, hsensor, invhsensor, kernel = GaussianKernel(),
                   nsamples = 400, snstd = 1.0, snoise = None, svectors = None):
	"""Estimate state transition from observation stream.
	
	Given a state-space model
	X_{t+1} = f(X_t) + W_t
	Y_t     = h(X_t) + V_t,
	where both the sensor function h() and a observations Y_{1:t} are known
	and update and observation are corrupted with zero-mean noises W_t and V_t,
	estimate the probability distribution of the state transition function f()
	parametrized in the reproducing kernel hilbert space spanned by the
	specified kernels.
	
	Args:
		observations: generator of observations, each entry is an observation,
		              as a numpy array.
		hsensor: sensor stochastic model, as a function that takes
		         x and y and returns a value proportional to p(y|x).
		invhsensor: inverse of the sensor function, h^-1(y); equivalently
		            solves argmax_x(p(x|y)).
		kernel: Kernel object describing the domain of the estimation,
		        if it has incomplete parameters, they will be computed
		        from the observations.
		nsamples: number of samples to draw from the MCMC process.
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		snstd: if snoise is None, additive white gaussian noise will be
		       used with this standard deviation. If both arguments are
		       None, the process will present no noise. 
		snoise: state transition known additive noise process W_t, as
		        a function of the state.
	Returns:
		Tuple ([a_i], s) of mixing parameters and support vectors.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given KernelSpace.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
	"""
	
	obsmatrix = [observation for observation in observations]
	
	return offline(np.array(obsmatrix), hsensor, invhsensor,
	               kernel, nsamples, snstd, snoise, svectors)

def online(observations, hsensor, invhsensor, kernel = GaussianKernel(),
           nsamples = 400, snstd = 1.0, snoise = None, svectors = None):
	"""Estimate time-varying state transition from observation matrix.
	
	Given a state-space model
	X_{t+1} = f_t(X_t) + W_t
	Y_t     = h(X_t) + V_t,
	where both the sensor function h() and a observations Y_{1:t} are known
	and update and observation are corrupted with zero-mean noises W_t and V_t,
	estimate the probability distribution of the state transition function f()
	parametrized in the reproducing kernel hilbert space spanned by the
	specified kernels.
	
	Args:
		observations: matrix of observations, each row is an observation,
		              as a numpy array.
		hsensor: sensor stochastic model, as a function that takes
		         x and y and returns a value proportional to p(y|x).
		invhsensor: inverse of the sensor function, h^-1(y); equivalently
		            solves argmax_x(p(x|y)).
		kernel: Kernel object describing the domain of the estimation,
		        if it has incomplete parameters, they will be computed
		        from the observations.
		nsamples: number of samples to draw from the MCMC process.
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		snstd: if snoise is None, additive white gaussian noise will be
		       used with this standard deviation. If both arguments are
		       None, the process will present no noise. 
		snoise: state transition known additive noise process W_t, as
		        a function of the state.
	Returns:
		Tuple ([a_i], s) of mixing parameters and support vectors.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given KernelSpace.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
	"""
	
	return [estimate for estimate in
	           online_stream((row for row in observations),
	                         hsensor, invhsensor, kernel,
	                         nsamples, snstd, snoise, svectors)]

def online_stream(observations, hsensor, invhsensor, kernel = GaussianKernel(),
                  nsamples = 400, snstd = 1.0, snoise = None, svectors = None):
	"""Estimate time-varying state transition from observation stream.
	
	Given a state-space model
	X_{t+1} = f_t(X_t) + W_t
	Y_t     = h(X_t) + V_t,
	where both the sensor function h() and a observations Y_{1:t} are known
	and update and observation are corrupted with zero-mean noises W_t and V_t,
	estimate the probability distribution of the state transition function f()
	parametrized in the reproducing kernel hilbert space spanned by the
	specified kernels.
	
	Args:
		observations: generator of observations, each entry is an observation,
		              as a numpy array.
		hsensor: sensor stochastic model, as a function that takes
		         x and y and returns a value proportional to p(y|x).
		invhsensor: inverse of the sensor function, h^-1(y); equivalently
		            solves argmax_x(p(x|y)).
		kernel: Kernel object describing the domain of the estimation,
		        if it has incomplete parameters, they will be computed
		        from the observations.
		nsamples: number of samples to draw from the MCMC process.
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		snstd: if snoise is None, additive white gaussian noise will be
		       used with this standard deviation. If both arguments are
		       None, the process will present no noise. 
		snoise: state transition known additive noise process W_t, as
		        a function of the state.
	Returns:
		Tuple ([a_i], s) of mixing parameters and support vectors.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given KernelSpace.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
	"""
	
	pass

def _getsupportvectors(observations, invhsensor):
	"""Propose a set of support vectors for the state from observations y_{1:t}.
	
	Args:
		observations: matrix of observations, each row is an observation.
		invhsensor: inverse of the sensor function, h^-1(y); equivalently
		            solves argmax_x(p(x|y)).
	Returns:
		List of support vectors s_i in state-space which adequately cover
		the preimages of the observations.
	"""
	
	dummy = GaussianKernel()
	dummy.estimate_params([observation in observations])
	
	threshold = 2 * dummy.sigma
	svectors  = []
	
	for observation in observations:
		svectors = _getsupportvectors_stream(observation, invhsensor,
		                                     svectors, threshold)
	
	return svectors

def _getsupportvectors_stream(observation, invhsensor, svectors, threshold):
	"""Add a support vectors for the state from observations y_t if necessary.
	
	Args:
		observation: new observation vector.
		invhsensor: inverse of the sensor function, h^-1(y); equivalently
		            solves argmax_x(p(x|y)).
		svectors: current list of support vectors.
		threshold: distance threshold at which the new observation is deemed
		           far enough to any existing support vector to require a new
		           support vector to be added.
	Notes:
		Input variable svectors may be modified.
	Returns:
		New list of support vectors s_i in state-space which adequately cover
		the preimages of the observations, including the last one. It may
		or may not be the same as before depending on the coverage of the
		new observation preimage.
	"""
	
	proposal = invhsensor(observation)
	
	mindistance = float("inf")
	
	for svector in svectors:
		mindistance = min(mindistance, np.linalg.norm(svector, proposal))
	
	if mindistance > threshold:
		svectors.append(proposal)
	
	return svectors

