#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MCMC.py
# Markov Chain Monte Carlo implementation
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

class MCMC(object):
	"Markov Chain Monte Carlo model."

	def __init__(self, observations, initial, proposer,
	             likelihood, hfactor=None):
		"""Construct a new Markov Chain Monte Carlo model.

		Construct a new Markov Chain Monte Carlo system for
		obtaining samples from a complex distribution just by
		evaluating the density (or a function proportional to it).

		Args:
			observations: output observations, as a numpy matrix.
			initial: initial sample for the chain.
			proposer: proposal function sampler; given the current sample
			          as argument, propose a new sample.
			likelihood: function that takes a sample and is proportional to
			            the posterior distribution to be sampled.
			hfactor: hastings factor which measures the asymmetry
			         of the proposal distribution following the formula
			         q(x | x') / q (x' | x).
		"""

		if hfactor is None:
			hfactor = MCMC._unity_hastings

		self._observations = observations
		self._sample       = initial
		self._proposer     = proposer
		self._likelihood   = likelihood
		self._hfactor      = hfactor
		self._accepted     = 1
		self._total        = 1
		self._slike        = self._get_likelihood(self._sample)

	@property
	def ratio(self):
		"Accepted rate (i.e. proportion of accepted samples)."

		return self._accepted / self._total

	@property
	def likelihood(self):
		"Likelihood of the last sample."

		return self._slike

	@staticmethod
	def _uniform_likelihood(sample):
		"Uniform likelihood function."

		return 1.0

	@staticmethod
	def _unity_hastings(newsample, prevsample):
		"""Symmetric hastings factor.

		Hastings factor for any symmetric proposal distribution,
		e.g. Gaussian, Exponential. Corresponds to
		q(x | x') / q(x' | x)
		and always equals 1.0.

		Returns:
			Hastings factor for the proposal distribution.
		"""

		return 1.0

	def _propose(self, sample):
		"""Propose a new sample given the current one.

		Returns:
			New sample.
		"""

		return self._proposer(sample)

	def _get_likelihood(self, sample):
		"""Calculate the likelihood of a sample.

		Returns:
			Posterior density for the sample (or proportional to it).
		"""

		return self._likelihood(sample)

	def _hastingsfactor(self, sample, previous):
		"Hastings factor, corresponding to q(prev | sample) / q(sample | prev)."

		return self._hfactor(sample, previous)

	def draw(self):
		"""Get a sample from the state posterior distribution.

		Propose a new sample from the proposal distribution and
		either accept or reject it stochastically following the
		Metropolis-Hastings algorithm: accept it with probability
		min {1, p(x') q(x|x') / p(x) q(x'|x)},
		where p() corresponds to the distribution in question and
		q() to the proposal distribution.

		Returns:
			Sample from the state posterior distribution.
		"""

		accepted = False

		if self._slike <= 0:
			print(self._sample)

		proposal = self._propose(self._sample)
		like     = self._get_likelihood(proposal)

		if self._slike > 0:
			ratio = like / self._slike * self._hastingsfactor(proposal, self._sample)
		else:
			ratio = 1

		rand = np.random.random_sample()

		if rand < min(1.0, ratio):
			accepted = True

		if accepted:
			self._sample    = proposal
			self._slike     = like
			self._accepted += 1

		self._total += 1

		return self._sample
