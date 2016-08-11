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

class RecursivePMCMC(MCMC):
	"Recursive Particle Markov Chain Monte Carlo model."
	
	def __init__(self, initial, prior, proposer, thetatransition,
	             smcprior, ftransitioner, hsensor, nsamples,
	             hfactor = None):
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
			          as argument, propose a new sample.
			thetatransition: sample transition stochastic model,
			                 as a function that takes f_t and f_{t+1} and
			                 calculates the likelihood of the transition.
			smcprior: state prior stochastic model for the particle filter,
			          as a function with no arguments which generates samples
			          from the prior.
			ftransitioner: generate the state transition stochastic model,
			               as a function that takes x_t and returns x_{t+1},
			               from the sample vector, fter(s) -> f(x_t) -> x_{t+1}.
			hsensor: sensor stochastic model, as a function that takes
			         x_t and y_t and returns a value proportional to p(y_t|x_t).
			nsamples (int): number of samples to draw each batch of the
			                particle filter.
			hfactor: hastings factor which measures the asymmetry 
			         of the proposal distribution following the formula
			         q(x | x') / q (x' | x).
		Notes:
			The difference between thetatransition and ftransitioner is that
			the first models the transition between the sampled variable
			in time, while the second models the transition between states
			in the internal particle filter (in the same time).
		"""
		
		if hastingsfactor is None:
			hastingsfactor = PMCMC._unity_hastings
		
		self._prior           = prior
		self._thetatransition = thetatransition
		self._smcprior        = smcprior
		self._ftransitioner   = ftransitioner
		self._hsensor         = hsensor
		
		super().__init__([], initial, prior, proposer, likelihood,
		                 nsamples, hastingsfactor)
		
		dummy             = self._ftransition(initial)
		self._samples     = []
		self._prevsamples = []
		self._filters     = []
		self._prevfilters = [SMC([], self._smcprior, dummy,
		                         self._hsensor, self._nsamples)]
	
	@property
	def proposer(self):
		return self._proposer
	
	@proposer.setter
	def ftransition(self, proposer):
		self._proposer = proposer
	
	@property
	def thetatransition(self):
		return self._thetatransition
	
	@thetatransition.setter
	def ftransition(self, thetatransition):
		self._thetatransition = thetatransition
	
	@property
	def ftransitioner(self):
		return self._ftransitioner
	
	@ftransitioner.setter
	def ftransition(self, ftransitioner):
		self._ftransitioner = ftransitioner
	
	def _add_observation(self, observation):
		"""Start another time step using the given observation.
		
		All the gathered samples from this time step will be used
		as a prior to the next one.
		"""
		
		self._prevsamples = self._samples
		self._prevfilters = self._filters
		self._samples     = []
		self._filters     = []
		self._observation = observation
	
	def _propose(self, sample):
		"""Propose a new sample given the current one.
		
		Returns:
			New sample.
		"""
		
		index  = int(np.random_sample() * len(self._prevsamples))
		prev   = self._prevsamples[index]
		filter = self._prevfilter[index].clone()
		
		proposal           = self._proposer(prev)
		filter.ftransition = self._ftransitioner(proposal)
		
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
		ptransition = np.sum(thetatransition(prev, sample)
		                         for prev in self._prevsamples)
		
		return ptransition * like
	
	def _hastingsfactor(self, sample, previous):
		"Hastings factor, corresponding to q(prev | sample) / q(sample | prev)."
		
		# TODO correct this, since q(x | x') = q(x) and this depends on all
		# the preivous samples, it is a large sum (hopefully it shouldn't matter too much
		# in the sense that something reasonable should be output as is,
		# even if very very skewed)
		
		return self._hfactor(sample[0], previous[0])
	
	def _draw(self):
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
