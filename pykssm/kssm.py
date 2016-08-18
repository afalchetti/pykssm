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
from .SMC import SMC
from .PMCMC import PMCMC
from .RecursivePMCMC import RecursivePMCMC
from .Kernel import GaussianKernel

def offline(observations, hsensor, invhsensor, kernel=GaussianKernel(),
            smcprior=None, nsamples=400, snstd=1.0, snoise=None,
            svectors=None, verbose=False):
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
		smcprior: state prior for the internal particle filter;
		          if None, use a appropriately sized deterministic zero vector.
		nsamples: number of samples to draw from the MCMC process.
		snstd: if snoise is None, additive white gaussian noise will be
		       used with this standard deviation. If both arguments are
		       None, the process will present no noise. 
		snoise: state transition known additive noise process W_t, as
		        a function of the state.
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		verbose: if true, log every time a sample is drawn.
	Returns:
		Tuple ([a_i], s, k) of mixing parameters, support vectors and kernel.
		The kernel defines the form of each component, with all
		the required parameters set.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given Kernel space.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
	"""
	
	if svectors is None:
		(svectors, kernel) = _getsupportvectors(observations, invhsensor, kernel)
	
	if snoise is None:
		if snstd is None:
			snoise = lambda s: np.zeros(len(s))
		else:
			snoise = lambda s: np.random.randn(len(s)) * snstd
	if not kernel.complete_params():
		kernel.estimate_params(svectors)
	
	firststate = invhsensor(observations[0])
	ssize      = len(firststate)
	
	if smcprior is None:
		smcprior = lambda: firststate
	
	sampler = PMCMC(observations,
	                initial       = np.zeros(len(svectors)),
	                prior         = lambda s: 1.0,
	                proposer      = lambda s: np.random.multivariate_normal(s, 0.2**2 * np.exp(-0.2 * _diff2matrix(svectors, svectors))),
	                smcprior      = smcprior,
	                ftransitioner = lambda a: lambda s: kernel.mixture_eval(a, svectors, s) + snoise(s),
	                hsensor       = hsensor,
	                nsamples      = 200)
	
	samples = []
	
	for i in range(nsamples):
		if verbose:
			print("sample", i + 1, "| ratio:", sampler.ratio)
		samples.append(sampler.draw())
	
	return (samples, svectors, kernel)

def offline_stream(observations, hsensor, invhsensor,
                   kernel=GaussianKernel(), smcprior=None, nsamples=400,
                   snstd=1.0, snoise=None, svectors=None, verbose=False):
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
		smcprior: state prior for the internal particle filter;
		          if None, use a appropriately sized deterministic zero vector.
		nsamples: number of samples to draw from the MCMC process.
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		snstd: if snoise is None, additive white gaussian noise will be
		       used with this standard deviation. If both arguments are
		       None, the process will present no noise. 
		snoise: state transition known additive noise process W_t, as
		        a function of the state.
		verbose: if true, log every time a sample is drawn.
	Returns:
		Tuple ([a_i], s, k) of mixing parameters, support vectors and kernel.
		The kernel defines the form of each component, with all
		the required parameters set.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given KernelSpace.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
	"""
	
	obsmatrix = [observation for observation in observations]
	
	return offline(observations=np.array(obsmatrix), hsensor=hsensor,
	               invhsensor=invhsensor, kernel=kernel, nsamples=nsamples,
	               snstd=snstd, snoise=snoise, svectors=svectors,
	               verbose=verbose)

def online(observations, hsensor, invhsensor, theta,
           kernel=GaussianKernel(), smcprior=None, nsamples=400,
           snstd=1.0, snoise=None, svectors=None, verbose=False):
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
		theta: (state transition function) transition function, as a
		       function that takes f_t and f_{t+1} and calculates
		       p(f_{t+1}|f_t). Both functions are represented by their
		       mixing parameters.
		kernel: Kernel object describing the domain of the estimation,
		        if it has incomplete parameters, they will be computed
		        from the observations.
		smcprior: state prior for the internal particle filter;
		          if None, use a appropriately sized deterministic zero vector.
		nsamples: number of samples to draw from the MCMC process.
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		snstd: if snoise is None, additive white gaussian noise will be
		       used with this standard deviation. If both arguments are
		       None, the process will present no noise. 
		snoise: state transition known additive noise process W_t, as
		        a function of the state.
		verbose: if true, log every time a time step has been completed.
	Returns:
		Tuple ([a_i], s, k) of mixing parameters, support vectors and kernel.
		The kernel defines the form of each component, with all
		the required parameters set.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given KernelSpace.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
	"""
	
	if svectors is None:
		(svectorsx, kernel) = _getsupportvectors(observations, invhsensor, kernel)
	
	if not kernel.complete_params():
		kernel.estimate_params(svectors)
	
	return [estimate for estimate in
	           online_stream(observations=(row for row in observations),
	              hsensor=hsensor, invhsensor=invhsensor, theta=theta,
                  kernel=kernel, smcprior=smcprior, nsamples=nsamples,
                  snstd=snstd, snoise=snoise, svectors=svectors,
                  verbose=verbose)]

def online_stream(observations, hsensor, invhsensor, theta,
                  kernel=GaussianKernel(1.0), smcprior=None,
                  nsamples=400, snstd=1.0, snoise=None,
                  svectors=None, verbose=False):
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
		theta: (state transition function) transition function, as a
		       function that takes f_t and f_{t+1} and calculates
		       p(f_{t+1}|f_t). Both functions are represented by their
		       mixing parameters.
		kernel: Kernel object describing the domain of the estimation;
		        it must have all its parameters defined.
		smcprior: state prior for the internal particle filter;
		          if None, use a appropriately sized deterministic zero vector.
		nsamples: number of samples to draw from the MCMC process.
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		snstd: if snoise is None, additive white gaussian noise will be
		       used with this standard deviation. If both arguments are
		       None, the process will present no noise. 
		snoise: state transition known additive noise process W_t, as
		        a function of the state.
		verbose: if true, log every time a timestep has been completed.
	Returns:
		Tuple ([a_i], s, k) of mixing parameters, support vectors and kernel.
		The kernel defines the form of each component, with all
		the required parameters set.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given KernelSpace.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
	"""
	
	observations     = iter(observations)
	threshold        = 0.6 * kernel.deviation() 
	firstobservation = next(observations)
	firststate       = invhsensor(firstobservation)
	
	if svectors is None:
		prevlen  = 0
		svectors = [firststate]
		initial  = []
	else:
		prevlen = len(svectors)
		initial = np.zeros(len(svectors))
	
	if snoise is None:
		if snstd is None:
			snoise = lambda s: np.zeros(len(s))
		else:
			snoise = lambda s: np.random.randn(len(s)) * snstd
	
	if smcprior is None:
		smcprior = lambda: np.zeros(len(firststate))
	
	def proposer(sample, context):
		svectors = context["svectors"]
		psigmas  = context["psigmas"]
		
		if context["augment"]:
			presvectors = svectors[:-1]
			prefix = sample + psigmas[:-1] * np.random.randn(len(presvectors))
			
			prevstate = context["prevstate"]
			state     = context["state"]
			coverage  = kernel.mixture_eval(sample, presvectors, prevstate)
			
			# mean = whatever is necessary to raise the transition function
			# to predict this state from the previous one;
			# NOTE in the current settings of this module, sample is
			# one-dimensional, so mu is a scalar, hence the [0]
			mu = state[0] - coverage
			
			return np.concatenate((prefix, [mu + np.random.randn() *
			                                   kernel.deviation()]))
		else:
			return sample + psigmas * np.random.randn(len(svectors))
	
	def thetactx(previous, sample, context):
		if context["augment"]:
			return theta(previous, sample[:-1])
		else:
			return theta(previous, sample)
	
	def ftransitioner(sample, context):
		svectors = context["svectors"]
		
		return lambda state: (kernel.mixture_eval(sample, svectors, state) +
		                      snoise(state))
	
	def proppdf(previous, sample, context):
		svectors = context["svectors"]
		psigmas  = context["psigmas"]
		sigma    = kernel.deviation()
		
		if context["augment"]:
			# sample[:-1] will be copied from a random previous state
			# so technically this should be multiplied by
			# dirac(previous), but that doesn't change anything
			# in practice (sample always comes from p(sample|previous)).
			last = sample[-1]
			
			presvectors = svectors[:-1]
			
			prevstate = context["prevstate"]
			state     = context["state"]
			coverage  = kernel.mixture_eval(previous, presvectors, prevstate)
			
			# mean = whatever is necessary to raise the transition function
			# to predict this state from the previous one;
			# NOTE in the current settings of this module, sample is
			# one-dimensional, so mu is a scalar, hence the [0]
			mu = state[0] - coverage
			
			return 0.5 / sigma * np.exp(-0.5 *
			                            (np.linalg.norm(last - mu) / sigma)**2)
		else:
			distances = [np.linalg.norm(p - s) for (p, s) in zip(previous, sample)]
			probs     = [0.5 / s * np.exp(-0.5 * (d/s)**2) for
			                (d, s) in zip(distances, psigmas)]
			
			return np.prod(probs)
	
	sampler = RecursivePMCMC(initial          = initial,
	                         prior            = lambda s: 1.0,
	                         proposer         = proposer,
	                         proppdf          = proppdf,
	                         thetatransition  = thetactx,
	                         smcprior         = smcprior,
	                         ftransitioner    = ftransitioner,
	                         hsensor          = hsensor,
	                         firstobservation = firstobservation,
	                         nsamples         = 200)
	
	prevobservation = firstobservation
	
	for (i, observation) in enumerate(observations):
		# svectors contains the support vectors accesible by the RecursivePMCMC;
		# augment should be true iff a new support vector has been added in the
		# current time step;
		# psigmas is a precomputed value for the proposal step;
		# state and prevstate correspond to estimates of the internal states
		# (used for speeding up MCMC convergence)
		sampler.context = {"svectors": svectors,
		                   "augment":  prevlen < len(svectors),
		                   "psigmas":  np.sqrt([hsensor(v, prevobservation) for
		                                           v in svectors]),
		                   "prevstate": invhsensor(prevobservation),
		                   "state":     invhsensor(observation)}
		
		sampler.add_observation(observation)
		
		samples         = [sampler.draw() for i in range(nsamples)]
		prevobservation = observation
		prevlen         = len(svectors)
		
		svectors = _getsupportvectors_stream(observation, invhsensor,
		                                     svectors, threshold)
		
		if verbose:
			print("time", i + 1)
		
		yield (samples, svectors, kernel)

def filter(ftransition, hsensor, sigmax, sigmay, x0, size):
	"""Run a forward nonlinear filter with gaussian noise.
	
	Args:
		ftransition: State transition function, x_{t+1} = ftransition(x_t);
		             (deterministic, disregards stochastic effects).
		hsensor: Sensor function, y = hsensor(x);
		         (deterministic, disregards stochastic effects).
		sigmax: Standard deviation for the state transition noise.
		sigmay: Standard deviation for the sensorn noise.
		x0: Initial state.
		size: Number of time steps the filter will be run for.
	"""
	
	x = np.zeros(size)
	y = np.zeros(size)

	x[0] = x0
	y[0] = hsensor(x[0]) + np.random.randn() * sigmay

	for i in range(1, size):
		x[i] = ftransition(x[i - 1]) + np.random.randn() * sigmax
		y[i] = hsensor(x[i])         + np.random.randn() * sigmay
	
	return (x, y)

def filter_t(ftransition, hsensor, sigmax, sigmay, x0, size):
	"""Run a forward time-varying nonlinear filter with gaussian noise.
	
	Args:
		ftransition: State transition function, x_{t+1} = ftransition(t, x_t);
		             (deterministic, disregards stochastic effects);
		             the time argument grows with the state index.
		hsensor: Sensor function, y = hsensor(x);
		         (deterministic, disregards stochastic effects).
		sigmax: Standard deviation for the state transition noise.
		sigmay: Standard deviation for the sensorn noise.
		x0: Initial state.
		size: Number of time steps the filter will be run for.
	"""
	
	x = np.zeros(size)
	y = np.zeros(size)

	x[0] = x0
	y[0] = hsensor(x[0]) + np.random.randn() * sigmay

	for i in range(1, size):
		x[i] = ftransition(i, x[i - 1]) + np.random.randn() * sigmax
		y[i] = hsensor(x[i])            + np.random.randn() * sigmay
	
	return (x, y)

def kls(x, y, svectors, kernel, regularization):
	"""Estimate a function from known points using kernel linear regression.
	
	Find the mixing parameters {a}_i that minimize the expression
	sum((f(x) - y)**2) + lambda |f|_K^2
	representing the function f as a kernel mix, in the space spanned
	by the support vectors,
	f(x) = sum(a_i * k(s_i, x)),
	where {s}_i correspond to the spport vectors and |f|_K is the norm
	induced by the kernel.
	
	The first term reduces the error in predicting y from x, while
	the second one ensures f is appropiately smooth.
	
	Args:
		x: list of the x axis component for each known point.
		y: list of the y axis component for each known point.
		svectors: support vectors for the estimation.
		kernel: Kernel object describing the support vectors.
		regularization: Tikhonov regularization factor
		                (lambda in the expression above).
	Notes:
		Assumes symmetric kernel k(x, y) = k(y, x)
	Returns:
		Mixing parameters for the support vectors that approximate
		the function y = f(x).
	"""
	
	# In support-vector induced space, the minimization becomes
	# min_c {|Y - K_tr c|_2^2 + lambda c^T K_rr c}
	# where the Ks are the gram matrices: K_ij = k(x_i, s_j);
	# K_rr corresponds to the square matrix using only support vectors and
	# K_tr corresponds to using support vectors and data.
	# 
	# Deriving and equaling to zero gives the expression
	# (K_rt K_tr + lambda K_rr) c = K_rt Y
	# where K_tr uses the inverted arguments for the kernel than K_rt.
	# For performance, the kernel will be assumed symmetric so K_tr = K_rt^T.
	
	K_rt = kernel.gram(svectors, x)
	K_tr = K_rt.T
	K_rr = kernel.gram(svectors, svectors)
	
	left = K_rt.dot(K_tr) + regularization * K_rr
	right = K_rt.dot(y)
	
	# possible optimization: use common cholesky factorization
	# and triangular solver (from scipy), but not worth it right now
	return np.array([(np.linalg.solve(left, row)) for row in right.T])

def _getsupportvectors(observations, invhsensor, kernel):
	"""Propose a set of support vectors for the state from observations y_{1:t}.
	
	Args:
		observations: matrix of observations, each row is an observation.
		invhsensor: inverse of the sensor function, h^-1(y); equivalently
		            solves argmax_x(p(x|y)).
		kernel: Kernel object describing the domain of the estimation,
		        if it has incomplete parameters, they will be computed
		        from the observations.
	Returns:
		Tuple ([s]_i, kernel)
		s is a list of support vectors s_i in state-space which adequately cover
		the preimages of the observations.
		kernel is the updated kernel (parameters estimated from data
		if not already set).
	"""
	
	if not kernel.complete_params():
		kernel.estimate_params([invhsensor(observation) for observation in observations])
	
	threshold = 0.6 * kernel.sigma
	svectors  = []
	
	for observation in observations:
		svectors = _getsupportvectors_stream(observation, invhsensor,
		                                     svectors, threshold)
	
	return (svectors, kernel)

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
		mindistance = min(mindistance, np.linalg.norm(svector - proposal))
	
	if mindistance > threshold:
		svectors.append(proposal)
	
	return svectors

def _diff2matrix(a, b):
	""" Get the squared difference matrix.
	
	Calculate a matrix of the square of the differences between all
	the components of each vector, i.e. M_ij = |a_i - b_j|^2.
	
	Args:
		a: first vector.
		b: second vector.
	Returns:
		Squared difference matrix.
	"""
	matrix = np.zeros((len(a), len(b)))
	
	for (i, ai) in enumerate(a):
		for (k, bk) in enumerate(b):
			matrix[i, k] = np.linalg.norm(ai - bk)**2
	
	return matrix

def test():
	f  = lambda x: 10 * np.sinc(x / 7)
	h  = lambda x: x
	x0 = 0 + np.random.randn() * 10

	sigmax = 2
	sigmay = 2
	size   = 40

	(x, y) = filter(f, h, sigmax, sigmay, x0, size)
	
	(samples, svectors, kernel) = offline(observations = y[np.newaxis].T,
	                                      hsensor      = lambda x, y: 0.5 / sigmay * np.exp(-0.5 * (np.linalg.norm(x - y) / sigmay)**2),
	                                      invhsensor   = lambda y: y,
	                                      kernel       = GaussianKernel(),
	                                      nsamples     = 5,
	                                      snstd        = sigmax)
	
	limit = 10.0 * np.ceil(1/10.0 *2.0 * max(abs(min(x)), abs(max(x))))
	grid  = np.arange(-limit, limit, 2)
	
	smean = np.mean(np.array(samples), 0)
	svar  = np.var(np.array(samples), 0)
	
	print("samples:", samples)
	print("s:", np.array(samples))
	print("mean:", smean)
	print("svar:", svar)
	
	# Real transition function
	real  = [f(i) for i in grid]
	
	# Mean estimate and its marginal deviation (note that
	# since support vectors are constants and the mixture
	# is a linear combination, the variance just requires
	# evaluating the mixture with the weight variances)
	estmean = np.array([kernel.mixture_eval(smean, svectors, [i]) for i in grid])
	estvar  = np.array([kernel.mixture_eval(svar,  svectors, [i]) for i in grid])
	eststd  = np.sqrt(estvar)
	
	like = []

	for sample in samples:
		filt  = SMC(observations = y[np.newaxis].T,
		            prior        = lambda: 0 + np.random.randn() * 10,
		            ftransition  = lambda x: kernel.mixture_eval(sample, svectors, x) + np.random.randn() * sigmax,
		            hsensor      = lambda x, y: 0.5/sigmay * np.exp(-((x - y)/sigmay)**2),
		            nsamples     = 50)
		
		like.append(filt.get_likelihood())

def test2():
	# Time-varying model
	def ft(t, x):
		# time-invariant stable 
		def flow(x):
			return x / 2 + 25 * x / (1 + x * x)
		
		# time-invariant unstable (two accumulation points)
		def fhigh(x):
			return 10 * np.sinc(x / 7)
		
		# linear interpolation between the previous two
		def fmid(t, x):
			return (60 - t) / 30 * flow(x) + (t - 30) / 30 * fhigh(x)
		
		if t < 30:
			return flow(x)
		elif t > 60:
			return fhigh(x)
		else:
			return fmid(t, x)
	
	ht = lambda x: x / 2 + 5
	
	sigmaxt0 = 1
	sigmaxt  = 1
	sigmayt  = np.sqrt(0.5)
	sizet    = 90
	
	xt0 = 0 + np.random.randn() * sigmaxt0
	
	(xt, yt) = filter_t(ft, ht, sigmaxt, sigmayt, xt0, sizet)
	
	# Time-varying KSSM, the core of this notebook
	# (state transition function) transition standard deviation
	sigmaf = 0.2
	
	estimate = online(observations = yt[np.newaxis].T,
	                  hsensor      = lambda x, y: 0.5 / sigmayt *
	                                              np.exp(-0.5 * (np.linalg.norm((x / 2 + 5) - y)/sigmayt)**2),
	                  invhsensor   = lambda y: 2 * (y - 5),
	                  theta        = lambda f1, f2: 0.5 / sigmaf *
	                                                np.exp(-0.5 * (np.linalg.norm(f1 - f2)/sigmaf)**2),
	                  kernel       = GaussianKernel(),
	                  nsamples     = 400,
	                  snstd        = sigmaxt,
	                  smcprior     = lambda: np.array([0 + np.random.randn() * sigmaxt0]),
	                  verbose      = True)
	# estimate is an array of tuples of the form (samples, svectors, kernel), each one corresponding
	# to a time step and similar to the offline case.