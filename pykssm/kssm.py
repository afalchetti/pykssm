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

def offline(observations, hsensor, invhsensor=None, kernel=GaussianKernel(),
            smcprior=None, nsamples=400, sigmax=1.0, sigmay=1.0,
            xnoise=None, ynoise=None, svectors=None, verbose=False):
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
		         x and returns a noiseless y, y = h(x).
		invhsensor: inverse of the sensor function, h^-1(y); equivalently
		            solves argmax_x(p(x|y)). If not specified, it will be
		            solved using scipy's solve system.
		kernel: Kernel object describing the domain of the estimation,
		        if it has incomplete parameters, they will be computed
		        from the observations.
		smcprior: state prior for the internal particle filter;
		          if None, use a appropriately sized deterministic zero vector.
		nsamples: number of samples to draw from the MCMC process.
		sigmax: if xnoise is None, additive white gaussian noise will be
		         used with this standard deviation. If both arguments are
		         None, the process will present no noise. 
		sigmay: if ynoise is None, additive white gaussian noise will be
		         used with this standard deviation. If both arguments are
		         None, the process will present no noise. 
		xnoise: state transition known additive noise process W_t, as
		        a function of the state.
		ynoise: sensor known additive noise process W_t, as a function of
		        the reference measurement and the measurement, P(y|y0),
		        where the reference y0 corresponds to the noiseless operation
		        on the known state, y0 = h(x0).
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		verbose: if true, log every time a sample is drawn.
	Returns:
		Tuple ([a_i], [p_i], s, k) of mixing parameters, likelihoods, support
		vectors and kernel.
		The kernel defines the form of each component, with all
		the required parameters set.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given Kernel space.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
		For each sample, its corresponding likelihood is also saved.
	"""
	
	if invhsensor is None:
		invhsensor = lambda y: scipy.optimize.solve(lambda x: hsensor(x) - y, y)
	
	firststate = invhsensor(observations[0])
	xsize      = len(firststate)
	ysize      = len(observations[0])
	
	if svectors is None:
		(svectors, kernel) = _getsupportvectors(observations, invhsensor, kernel)
	
	vsize = len(svectors)
	
	if xnoise is None:
		if sigmax is None:
			xnoise = lambda s: np.zeros(xsize)
		else:
			xnoise = lambda s: np.dot(sigmax, np.random.randn(xsize))
	
	if ynoise is None:
		if sigmay is None:
			psensor = lambda x0, y: 1.0 if np.abs(hsensor(x0) - y) < 1e-10 else 0.0
		else:
			if np.isscalar(sigmay):
				sigmay = np.array([[sigmay]])
			
			cov        = sigmay.dot(sigmay.T)
			coeff      = 1.0 / np.sqrt((2.0 * np.pi)**ysize * _pdet(cov))
			infomatrix = np.linalg.pinv(cov)
			expcoeff   = -0.5 * infomatrix
			
			def gnoise(x0, y):
				dy = hsensor(x0) - y
				return coeff * np.exp(np.dot(dy, np.dot(expcoeff, dy)))
			
			psensor = gnoise
	else:
		psensor = lambda x, y: ynoise(hsensor(x), y)
	
	if not kernel.complete_params():
		kernel.estimate_params(svectors)
	
	if smcprior is None:
		smcprior = lambda: firststate
	
	pcovariance = 0.2**2 * np.exp(-0.2 * _diff2matrix(svectors, svectors))
	
	def proposer(sample):
		prop = np.empty((xsize, vsize))
		
		for (i, row) in enumerate(sample):
			prop[i, :] = np.random.multivariate_normal(row, pcovariance)
		
		return prop
	
	sampler = PMCMC(observations,
	                initial       = np.zeros((xsize, vsize)),
	                prior         = lambda s: 1.0,
	                proposer      = proposer,
	                smcprior      = smcprior,
	                ftransitioner = lambda a: lambda s: kernel.mixture_eval(a, svectors, s) + xnoise(s),
	                hsensor       = psensor,
	                nsamples      = 200)
	
	samples     = np.empty((nsamples, xsize, vsize))
	likelihoods = np.empty(nsamples)
	
	for i in range(nsamples):
		if verbose:
			print("sample", i + 1, "| ratio:", sampler.ratio)
		samples    [i, :, :] = sampler.draw()
		likelihoods[i]       = sampler.likelihood
	
	return (samples, likelihoods, svectors, kernel)

def offline_stream(observations, hsensor, invhsensor=None,
                   kernel=GaussianKernel(), smcprior=None, nsamples=400,
                   sigmax=1.0, sigmay=1.0, xnoise=None, ynoise=None,
                   svectors=None, verbose=False):
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
		         x and returns a noiseless y, y = h(x).
		invhsensor: inverse of the sensor function, h^-1(y); equivalently
		            solves argmax_x(p(x|y)). If not specified, it will be
		            solved using scipy's solve system.
		kernel: Kernel object describing the domain of the estimation,
		        if it has incomplete parameters, they will be computed
		        from the observations.
		smcprior: state prior for the internal particle filter;
		          if None, use a appropriately sized deterministic zero vector.
		nsamples: number of samples to draw from the MCMC process.
		sigmax: if xnoise is None, additive white gaussian noise will be
		         used with this standard deviation. If both arguments are
		         None, the process will present no noise. 
		sigmay: if ynoise is None, additive white gaussian noise will be
		         used with this standard deviation. If both arguments are
		         None, the process will present no noise. 
		xnoise: state transition known additive noise process W_t, as
		        a function of the state.
		ynoise: sensor known additive noise process W_t, as a function of
		        the reference measurement and the measurement, P(y|y0),
		        where the reference y0 corresponds to the noiseless operation
		        on the known state, y0 = h(x0).
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		verbose: if true, log every time a sample is drawn.
	Returns:
		Tuple ([a_i], [p_i], s, k) of mixing parameters, likelihoods, support
		vectors and kernel.
		The kernel defines the form of each component, with all
		the required parameters set.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given Kernel space.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
		For each sample, its corresponding likelihood is also saved.
	"""
	
	obsmatrix = [observation for observation in observations]
	
	return offline(observations=np.array(obsmatrix), hsensor=hsensor,
	               invhsensor=invhsensor, kernel=kernel, nsamples=nsamples,
	               sigmax=sigmax, sigmay=sigmay, xnoise=xnoise, ynoise=ynoise,
	               svectors=svectors, verbose=verbose)

def online(observations, hsensor, theta, invhsensor=None,
           kernel=GaussianKernel(), smcprior=None, nsamples=400,
           sigmax=1.0, sigmay=1.0, xnoise=None, ynoise=None, svectors=None,
           verbose=False, _autoregressive=False):
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
		         x and returns a noiseless y, y = h(x).
		theta: (state transition function) transition function, as a
		       function that takes f_t and f_{t+1} and calculates
		       p(f_{t+1}|f_t). Both functions are represented by their
		       mixing parameters.
		invhsensor: inverse of the sensor function, h^-1(y); equivalently
		            solves argmax_x(p(x|y)). If not specified, it will be
		            solved using scipy's solve system.
		kernel: Kernel object describing the domain of the estimation,
		        if it has incomplete parameters, they will be computed
		        from the observations.
		smcprior: state prior for the internal particle filter;
		          if None, use a appropriately sized deterministic zero vector.
		nsamples: number of samples to draw from the MCMC process.
		sigmax: if xnoise is None, additive white gaussian noise will be
		         used with this standard deviation. If both arguments are
		         None, the process will present no noise. 
		sigmay: if ynoise is None, additive white gaussian noise will be
		         used with this standard deviation. If both arguments are
		         None, the process will present no noise. 
		xnoise: state transition known additive noise process W_t, as
		        a function of the state.
		ynoise: sensor known additive noise process W_t, as a function of
		        the reference measurement and the measurement, P(y|y0),
		        where the reference y0 corresponds to the noiseless operation
		        on the known state, y0 = h(x0).
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		verbose: if true, log every time a sample is drawn.
		_autoregressive: internal param. Do not use. Make the proposer
		                 and transition evaluator autoregressive-aware.
	Returns:
		List of tuples, one per time step.
		Each tuple ([a_i], [p_i], s, k) is ocmposed of mixing parameters,
		likelihoods, support vectors and kernel.
		The kernel defines the form of each component, with all
		the required parameters set.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given Kernel space.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
		For each sample, its corresponding likelihood is also saved.
	"""
	
	if not kernel.complete_params():
		kernel.estimate_params([invhsensor(observation) for observation in observations])
	
	if not kernel.complete_params():
		kernel.estimate_params(svectors)
	
	return [estimate for estimate in
	           online_stream(observations=(row for row in observations),
	              hsensor=hsensor, theta=theta, invhsensor=invhsensor,
                  kernel=kernel, smcprior=smcprior, nsamples=nsamples,
                  sigmax=sigmax, sigmay=sigmay, xnoise=xnoise, ynoise=ynoise,
                  svectors=svectors, verbose=verbose,
                  _autoregressive=_autoregressive)]

def online_stream(observations, hsensor, theta, invhsensor=None,
                  kernel=GaussianKernel(1.0), smcprior=None,  nsamples=400,
                  sigmax=1.0, sigmay=1.0, xnoise=None, ynoise=None,
                  svectors=None, verbose=False, _autoregressive=False):
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
		         x and returns a noiseless y, y = h(x).
		theta: (state transition function) transition function, as a
		       function that takes f_t and f_{t+1} and calculates
		       p(f_{t+1}|f_t). Both functions are represented by their
		       mixing parameters.
		invhsensor: inverse of the sensor function, h^-1(y); equivalently
		            solves argmax_x(p(x|y)). If not specified, it will be
		            solved using scipy's solve system.
		kernel: Kernel object describing the domain of the estimation,
		        if it has incomplete parameters, they will be computed
		        from the observations.
		smcprior: state prior for the internal particle filter;
		          if None, use a appropriately sized deterministic zero vector.
		nsamples: number of samples to draw from the MCMC process.
		sigmax: if xnoise is None, additive white gaussian noise will be
		         used with this standard deviation. If both arguments are
		         None, the process will present no noise. 
		sigmay: if ynoise is None, additive white gaussian noise will be
		         used with this standard deviation. If both arguments are
		         None, the process will present no noise. 
		xnoise: state transition known additive noise process W_t, as
		        a function of the state.
		ynoise: sensor known additive noise process W_t, as a function of
		        the reference measurement and the measurement, P(y|y0),
		        where the reference y0 corresponds to the noiseless operation
		        on the known state, y0 = h(x0).
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		verbose: if true, log every time a sample is drawn.
		_autoregressive: internal param. Do not use. Make the proposer
		                 and transition evaluator autoregressive-aware.
	Returns:
		List of tuples, one per time step.
		Each tuple ([a_i], [p_i], s, k) is ocmposed of mixing parameters,
		likelihoods, support vectors and kernel.
		The kernel defines the form of each component, with all
		the required parameters set.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given Kernel space.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
		For each sample, its corresponding likelihood is also saved.
	"""
	
	if invhsensor is None:
		invhsensor = lambda y: scipy.optimize.solve(lambda x: hsensor(x) - y, y)
	
	observations     = iter(observations)
	threshold        = 0.6 * kernel.deviation() 
	firstobservation = next(observations)
	firststate       = invhsensor(firstobservation)
	
	xsize = len(firststate)
	ysize = len(firstobservation)
	
	if svectors is None:
		prevlen  = 0
		svectors = np.array([firststate])
		initial  = np.zeros((len(firststate) if not _autoregressive else 1, 0))
	else:
		prevlen = len(svectors)
		initial = np.zeros((len(firststate) if not _autoregressive else 1,
		                    len(svectors)))
	
	if xnoise is None:
		if sigmax is None:
			xnoise = lambda s: np.zeros(len(s))
		else:
			xnoise = lambda s: np.dot(sigmax, np.random.randn(len(s)))
	
	if ynoise is None:
		if sigmay is None:
			psensor = lambda x0, y: 1.0 if np.abs(hsensor(x0) - y) < 1e-10 else 0.0
		else:
			if np.isscalar(sigmay):
				sigmay = np.array([[sigmay]])
			
			cov        = sigmay.dot(sigmay.T)
			coeff      = 1.0 / np.sqrt((2.0 * np.pi)**ysize * _pdet(cov))
			infomatrix = np.linalg.pinv(cov)
			expcoeff   = -0.5 * infomatrix
			
			def gnoise(x0, y):
				dy = hsensor(x0) - y
				return coeff * np.exp(np.dot(dy, np.dot(expcoeff, dy)))
			
			psensor = gnoise
	else:
		psensor = lambda x, y: ynoise(hsensor(x), y)
	
	if smcprior is None:
		smcprior = lambda: firststate
	
	deviation = kernel.deviation()
	
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
			# to predict this state from the previous one
			mu = state[:len(coverage)] - coverage
			
			alpha = mu + np.dot(deviation, np.random.randn(len(mu)))
			
			return np.append(prefix, alpha[np.newaxis].T, axis=1)
		else:
			return sample + psigmas * np.random.randn(len(svectors))
	
	def thetactx(previous, sample, context):
		if context["augment"]:
			return theta(previous, sample[:, :-1])
		else:
			return theta(previous, sample)
	
	if not _autoregressive:
		def ftransitioner(sample, context):
			svectors = context["svectors"]
			
			return lambda state: (kernel.mixture_eval(sample, svectors, state) +
			                      xnoise(state))
	else:
		def ftransitioner(sample, context):
			svectors  = context["svectors"]
			
			def predict(state):
				predicted = kernel.mixture_eval(sample, svectors, state)
				# the time series is assumed one-dimensional, i.e. the state
				# transition becomes [f(x_t), x_t, x_{t-1}, ...]
				
				pdelays = np.concatenate((predicted, state[:-1]))
				
				return pdelays + xnoise(pdelays)
			
			return predict
	
	if np.isscalar(deviation):
		deviation = deviation * np.eye(initial.shape[0])
	
	propcov      = deviation.dot(deviation.T)
	propcoeff    = 1.0 / (np.sqrt(2 * np.pi)**xsize * _pdet(propcov))
	propinfo     = np.linalg.pinv(propcov)
	propexpcoeff = -0.5 * propinfo
	
	def proppdf(previous, sample, context):
		svectors    = context["svectors"]
		psigmas     = context["psigmas"]
		ilogpsigmas = context["ilogpsigmas"]
		
		if context["augment"]:
			# sample[:-1] will be copied from a random previous state
			# so technically this should be multiplied by
			# dirac(previous), but that doesn't change anything
			# in practice (sample always comes from p(sample|previous)).
			last = sample[:, -1]
			
			presvectors = svectors[:-1]
			
			prevstate = context["prevstate"]
			state     = context["state"]
			coverage  = kernel.mixture_eval(previous, presvectors, prevstate)
			
			# mean = whatever is necessary to raise the transition function
			# to predict this state from the previous one
			mu = state[:len(coverage)] - coverage
			
			delta = last - mu
			
			return propcoeff * np.exp(np.dot(delta, np.dot(propexpcoeff, delta)))
		else:
			delta = previous - sample
			
			logprobs = np.fromiter((ilogsig - 0.5 * 
			                       np.dot(d, d) / sig**2 for
			                       (d, sig, ilogsig) in
			                           zip(delta.T, psigmas, ilogpsigmas)),
			                       dtype=np.float, count=len(psigmas))
			
			return np.exp(np.sum(logprobs))
	
	sampler = RecursivePMCMC(initial          = initial,
	                         prior            = lambda s: 1.0,
	                         proposer         = proposer,
	                         proppdf          = proppdf,
	                         thetatransition  = thetactx,
	                         smcprior         = smcprior,
	                         ftransitioner    = ftransitioner,
	                         hsensor          = psensor,
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
		                   "psigmas":  np.sqrt([psensor(v, prevobservation) for
		                                           v in svectors]),
		                   "prevstate": invhsensor(prevobservation),
		                   "state":     invhsensor(observation)}
		
		sampler.context["ilogpsigmas"] = np.log(1.0 / ((2 * np.pi)**len(svectors) * sampler.context["psigmas"]))
		
		sampler.add_observation(observation)
		
		samples     = np.empty((nsamples,
		                        len(firststate) if not _autoregressive else 1,
		                        len(svectors)))
		likelihoods = np.empty(nsamples)
		
		for k in range(nsamples):
			samples    [k, :, :] = sampler.draw()
			likelihoods[k]       = sampler.likelihood
		
		if verbose:
			print("time", i + 1)
		
		yield (samples, likelihoods, svectors, kernel)
		
		prevobservation = observation
		prevlen         = len(svectors)
		
		svectors = _getsupportvectors_stream(observation, invhsensor,
		                                     svectors, threshold)

def autoregressive(observations, hsensor, theta, invhsensor=None, delays=1,
                   kernel=GaussianKernel(), smcprior=None, nsamples=400,
                   sigmax=1.0, sigmay=1.0, xnoise=None, ynoise=None,
                   svectors=None, verbose=False):
	"""Estimate time-varying state transition considering autoregressive terms.
	
	Given a state-space model
	X_{t+1} = F_t(X_t) + W_t
	Y_t     = h(X_t) + V_t,
	where both the sensor function h() and a observations Y_{1:t} are known
	and update and observation are corrupted with zero-mean noises W_t and V_t,
	estimate the probability distribution of the state transition function f()
	parametrized in the reproducing kernel hilbert space spanned by the
	specified kernels.
	In this model, X_t corresponds to several timesteps,
	X_t = [x_t, x_{t-1}, ...]
	and the transition function considers these delays
	F_t(X_t) = [f_t(x_t), x_t, x_{t-1}, ...].
	
	Args:
		observations: matrix of observations, each row is an observation,
		              as a numpy array.
		hsensor: sensor stochastic model, as a function that takes
		         x and returns a noiseless y, y = h(x).
		invhsensor: inverse of the sensor function, h^-1(y); equivalently
		            solves argmax_x(p(x|y)). If not specified, it will be
		            solved using scipy's solve system.
		theta: (state transition function) transition function, as a
		       function that takes f_t and f_{t+1} and calculates
		       p(f_{t+1}|f_t). Both functions are represented by their
		       mixing parameters.
		delays: number of delays to consider, e.g. delays = 2 means
		        X_t = [x_t, x_{t-1}, x_{t-2}].
		kernel: Kernel object describing the domain of the estimation,
		        if it has incomplete parameters, they will be computed
		        from the observations.
		smcprior: state prior for the internal particle filter;
		          if None, use a appropriately sized deterministic zero vector.
		nsamples: number of samples to draw from the MCMC process.
		sigmax: if xnoise is None, additive white gaussian noise will be
		         used with this standard deviation. If both arguments are
		         None, the process will present no noise. 
		sigmay: if ynoise is None, additive white gaussian noise will be
		         used with this standard deviation. If both arguments are
		         None, the process will present no noise. 
		xnoise: state transition known additive noise process W_t, as
		        a function of the state.
		ynoise: sensor known additive noise process W_t, as a function of
		        the reference measurement and the measurement, P(y|y0),
		        where the reference y0 corresponds to the noiseless operation
		        on the known state, y0 = h(x0).
		svectors: if given, use these points as support vectors; otherwise,
		          compute suitable ones from the observations.
		verbose: if true, log every time a time step has been completed.
	Returns:
		List of tuples, one per time step.
		Each tuple ([a_i], [p_i], s, k) is ocmposed of mixing parameters,
		likelihoods, support vectors and kernel.
		The kernel defines the form of each component, with all
		the required parameters set.
		The list of support vectors s contains the components that span
		a subspace where the estimation occurs. Each s_i is parametrized
		in the given Kernel space.
		The samples a_i represent the probability distribution of the
		state transition function f() where each a_i corresponds to
		the mixing parameters describing a function of the form
		f_i(x) = sum(a_ik * Kernel(s_k, x)).
		For each sample, its corresponding likelihood is also saved.
	"""
	
	if invhsensor is None:
		invhsensor = lambda y: scipy.optimize.solve(lambda x: hsensor(x) - y, y)
	
	ydelays = np.empty((len(observations), delays + 1))
	
	ydelays[:, 0:1] = observations
	zeros = np.zeros(delays)
	
	for i in range(1, delays + 1):
		ydelays[:, i] = np.concatenate((zeros[:i], observations[:-i, 0]))
	
	invhsensor = np.vectorize(invhsensor)
	vhsensor   = lambda x: hsensor(x[0])
	vsmcprior  = lambda: np.concatenate((smcprior(), zeros))
	vxnoise    = None
	vynoise    = None
	
	if xnoise is not None:
		vxnoise = lambda s: np.concatenate(xnoise(s), zeros)
	if sigmax is not None:
		sigmax0      = sigmax
		sigmax       = np.zeros((delays + 1, delays + 1))
		sigmax[0, 0] = sigmax0
		
	
	if ynoise is not None:
		vynoise = lambda s: np.concatenate(ynoise(s), zeros)
	if sigmax is not None:
		sigmay0      = sigmay
		sigmay       = np.zeros((delays + 1, delays + 1))
		sigmay[0, 0] = sigmay0
	
	return online(observations=ydelays, hsensor=vhsensor, theta=theta,
	              invhsensor=invhsensor, kernel=kernel, smcprior=vsmcprior,
	              nsamples=nsamples, sigmax=sigmax, sigmay=sigmay,
	              xnoise=vxnoise, ynoise=vynoise, svectors=svectors,
	              verbose=verbose, _autoregressive=True)

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
	
	threshold = 0.6 * kernel.deviation()
	svectors  = np.zeros((0, len(invhsensor(observations[0]))))
	
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
	
	mindistance2 = float("inf")
	
	for svector in svectors:
		delta = svector - proposal
		mindistance2 = min(mindistance2, np.dot(delta, delta))
	
	if mindistance2 > threshold * threshold:
		svectors = np.concatenate((svectors, [proposal]))
	
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
			delta        = ai - bk
			matrix[i, k] = np.dot(delta, delta)
	
	return matrix

def _pdet(x):
	"""Pseudo-determinant calculation for symmetric matrices."""
	
	eigs, vecs = np.linalg.eigh(x)
	
	return np.prod([eigval for eigval in eigs if eigval > 1e-10])

def test():
	f  = lambda x: 10 * np.sinc(x / 7)
	h  = lambda x: x
	x0 = 0 + np.random.randn() * 10

	sigmax = 2
	sigmay = 2
	size   = 40

	(x, y) = filter(f, h, sigmax, sigmay, x0, size)
	
	(samples, svectors, kernel) = offline(observations = y[np.newaxis].T,
	                                      hsensor      = lambda x, y: 0.5 / sigmay * np.exp(-0.5 * np.dot(x - y, x - y) / sigmay**2),
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
	                                              np.exp(-0.5 * np.dot((x / 2 + 5) - y, (x / 2 + 5) - y)/sigmayt**2),
	                  invhsensor   = lambda y: 2 * (y - 5),
	                  theta        = lambda f1, f2: 0.5 / sigmaf *
	                                                np.exp(-0.5 * np.dot(f1 - f2, f1 - f2) / sigmaf**2),
	                  kernel       = GaussianKernel(),
	                  nsamples     = 400,
	                  snstd        = sigmaxt,
	                  smcprior     = lambda: np.array([0 + np.random.randn() * sigmaxt0]),
	                  verbose      = True)
	# estimate is an array of tuples of the form (samples, svectors, kernel), each one corresponding
	# to a time step and similar to the offline case.