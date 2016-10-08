#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# RecursivePMCMC.py
# Recursive Particle Markov Chain Monte Carlo implementation
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
from .MCMC import MCMC
from .SMC import SMC
import inspect

class RecursivePMCMC(MCMC):
	"Recursive Particle Markov Chain Monte Carlo model."
	
	def __init__(self, initial, prior, proposer, proppdf,
	             thetatransition, smcprior, ftransitioner,
	             hsensor, firstobservation, nsamples):
		"""Construct a new Recursive Particle Markov Chain Monte Carlo model.
		
		Construct a new Recursive Particle Markov Chain Monte Carlo system for
		obtaining samples from a complex distribution where the likelihood
		terms required for the Markov Chain Monte Carlo system can't be
		computed analytically. Instead, this class uses a particle filter
		(aka Sequential Monte Carlo) to solve the subsystem.
		Like a particle filter, the distribution changes each time step
		and consecutive ones are related by a known transition model. 
		
		Args:
			initial: initial sample for the chain.
			prior: prior model for the sampled variable, as a function
			       which takes a sample as argument and returns its probability.
			proposer: proposal function sampler; given the current sample
			          and a context as arguments, propose a new sample.
			proppdf: conditional probability density function (or proportional)
			         for the proposal distribution q(x_t | x_{t-1}) as a
			         function that takes x_{t-1}, x_t and a context and outputs
			         the corresponding probability.
			thetatransition: sample transition stochastic model,
			                 as a function that takes f_t, f_{t+1} and
			                 a context and calculates the likelihood of
			                 the transition.
			smcprior: state prior stochastic model for the particle filter,
			          as a function with no arguments which generates samples
			          from the prior.
			ftransitioner: generate the state transition stochastic model,
			               as a function that takes x_t and returns x_{t+1},
			               from the sample vector and a context,
			               fter(s, ctx) -> f(x_t) -> x_{t+1}.
			hsensor: sensor stochastic model, as a function that takes
			         x_t and y_t and returns a value proportional to p(y_t|x_t).
			firstobservation: initial observation, which doesn't depend on the 
			                  transition function, only on the sensor model.
			nsamples (int): number of samples to draw each batch of the
			                particle filter.
		Notes:
			The difference between thetatransition and ftransitioner is that
			the first models the transition between the sampled variable
			in time, while the second models the transition between states
			in the internal particle filter (also in time).
		"""
		
		self._observation     = firstobservation
		self._prior           = prior
		self._thetatransition = thetatransition
		self._smcprior        = smcprior
		self._ftransitioner   = ftransitioner
		self._hsensor         = hsensor
		self._nsamples        = nsamples
		self._proppdf         = proppdf
		self._context         = None
		
		dummytransition = lambda x: x
		initfilter      = SMC([self._observation], smcprior,
		                      dummytransition,
		                      hsensor, nsamples)
		self._samples     = [initial]
		self._prevsamples = []
		self._filters     = [initfilter]
		self._prevfilters = []
		
		# hastings factor top component
		self._prevq = 1.0
		
		initfilter.get_likelihood()
		
		super().__init__(observations=[self._observation],
		                 initial=(initial, initfilter), proposer=proposer,
		                 likelihood=MCMC._uniform_likelihood, hfactor=None)
	
	@property
	def context(self):
		"""Get the sampling context.
		
		Additional information that may be used in proposals,
		theta transitions and state transitions.
		"""
		
		return self._context
	
	@context.setter
	def context(self, context):
		self._context = context
	
	def add_observation(self, observation):
		"""Start another time step using the given observation.
		
		All the gathered samples from this time step will be used
		as a prior to the next one.
		"""
		
		self._prevsamples = self._samples
		self._prevfilters = self._filters
		self._samples     = []
		self._filters     = []
		self._observation = observation
		self._prevq       = 1.0
		
		# generate a new sample and immediately accept it;
		# samples should only be compared inside the same time step
		self._sample = self._propose(self._sample)
		self._like   = self._get_likelihood(self._sample)
	
	def _propose(self, sample):
		"""Propose a new sample given the current one.
		
		Returns:
			New sample.
		"""
		
		index  = int(np.random.random_sample() * len(self._prevsamples))
		prev   = self._prevsamples[index]
		filter = self._prevfilters[index].clone()
		
		proposal           = self._proposer(prev, self._context)
		filter.ftransition = self._ftransitioner(proposal, self._context)
		
		return (proposal, filter)
	
	def _get_likelihood(self, sampleinfo):
		"""Calculate the likelihood of a sample.
		
		To obtain the likelihood, create a particle filter and run it.
		Additionally multiply the result by the prior on the sample.
		
		Returns:
			Posterior density for the sample (or proportional to it).
		"""
		
		sample = sampleinfo[0]
		filter = sampleinfo[1]
		
		like        = filter.add_observation(self._observation)
		ptransition = np.sum(self._thetatransition(prev, sample, self._context)
		                         for prev in self._prevsamples)
		
		return ptransition * like
	
	def _hastingsfactor(self, sample, previous):
		"Hastings factor, corresponding to q(prev | sample) / q(sample | prev)."
		
		# In this class, the proposal is independent from the previous sample,
		# i.e.  q(x' | x) = q(x') = p (x' | x_{t-1}^r),
		# where x_{t-1}^r is a randomly selected previous particle, r ~ uniform.
		# Therefore,
		# q(x' | x) = sum(p(x'|x_{t-1}^r, r = i) p(r = i))
		#           = 1/N * sum(p(x' | x_{t-1}^i));
		# the constant cancels on the division.
		
		sample = sample[0]
		
		# for efficiency reasons, previous is disregarded and it is assumed
		# to be the last drawn sample
		
		# top    = np.sum(self._proppdf(prev, previous, self._context)
		#                     for prev in self._prevsamples)
		
		top    = self._prevq
		bottom = np.sum(self._proppdf(prev, sample, self._context)
		                    for prev in self._prevsamples)
		
		self._prevq = bottom
		
		return top / bottom
	
	def draw(self):
		"""Get a sample from the state posterior distribution.
		
		Propose a new sample from the proposal distribution and
		either accept or reject it stochastically following the
		Metropolis-Hastings algorithm: accept it with probability
		min {1, p(x') q(x|x') / p(x) q(x'|x)},
		where p() corresponds to the distribution in question and
		q() to the proposal distribution.
		Also, log the sample for future reference (it will become
		the prior for the next time step).
		
		Returns:
			Sample from the state posterior distribution.
		"""
		
		(sample, filter) = super().draw()
		
		self._samples.append(sample)
		self._filters.append(filter)
		
		return sample
