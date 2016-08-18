#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PMCMC.py
# Particle Markov Chain Monte Carlo implementation
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

from .MCMC import MCMC
from .SMC import SMC

class PMCMC(MCMC):
	"Particle Markov Chain Monte Carlo model."
	
	def __init__(self, observations, initial, prior, proposer,
	             smcprior, ftransitioner, hsensor, nsamples,
	             hfactor=None):
		"""Construct a new Particle Markov Chain Monte Carlo model.
		
		Construct a new Particle Markov Chain Monte Carlo system for
		obtaining samples from a complex distribution where the likelihood
		terms required for the Markov Chain Monte Carlo system can't be
		computed analytically. Instead, this class uses a particle filter
		(aka Sequential Monte Carlo) to solve the subsystem.
		
		Args:
			observations: output observations, as a numpy matrix.
			initial: initial sample for the chain.
			prior: prior model for the sampled variable, as a function
			       which takes a sample as argument and returns its probability.
			proposer: proposal function sampler; given the current sample
			          as argument, propose a new sample.
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
		"""
		
		self._prior         = prior
		self._smcprior      = smcprior
		self._ftransitioner = ftransitioner
		self._hsensor       = hsensor
		self._nsamples      = nsamples
		
		super().__init__(observations=observations, initial=initial,
		                 proposer=proposer, likelihood=MCMC._uniform_likelihood,
		                 hfactor=hfactor)
	
	def _get_likelihood(self, sample):
		"""Calculate the likelihood of a sample.
		
		To obtain the likelihood, create a particle filter and run it.
		Additionally multiply the result by the prior on the sample.
		
		Returns:
			Posterior density for the sample (or proportional to it).
		"""
		
		pfilter = SMC(self._observations, self._smcprior,
		              self._ftransitioner(sample), self._hsensor, self._nsamples)
		
		return self._prior(sample) * pfilter.get_likelihood()
