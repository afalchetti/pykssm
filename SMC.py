#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# SMC.py
# Sequential Monte Carlo implementation (aka particle filter)
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

class SMC(object):
	"Sequential Monte Carlo model."
	
	def __init__(self, observations, sprior, ftransition,
	             hsensor, nsamples):
		"""Construct a new Sequential Monte Carlo model.
		
		Construct a new Sequential Monte Carlo system for
		solving a state-space model of the form
		X_{t+1} = f(X_t) + W_t
		Y_t     = h(X_t) + V_t.
		
		Args:
			observations: output observations, as a numpy matrix.
			sprior: state prior stochastic model, as a function with no
			        arguments which generates samples from the prior.
			ftransition: state transition stochastic model, as a function
			             that takes x_t and returns x_{t+1}.
			hsensor: sensor stochastic model, as a function that takes
			         x_t and y_t and returns a value proportional to p(y_t|x_t).
			nsamples (int): number of samples to draw each batch.
		"""
		
		if nsamples < 1:
			raise ValueException("At least one particle is needed.")
		
		self._ftransition = ftransition
		self._hsensor     = hsensor
		self._timelength  = 
		self._nsamples    = nsamples
		self._usedsamples = 0
		self._samples     = []
		self._like        = -1.0
	
	def get_likelihood(self):
		"Get the likelihood of the observations given the model."
		if self._like < 0.0:
			(self._samples, self._like) = self._getparticles()
			self._usedsamples           = 0
		
		return self._like
	
	def draw(self):
		"""Get a sample from the state posterior distribution.
		
		Generate a particle through a particle filter. Since the
		filter requires multiple particles to be generated in
		parallel this function precomputes several particles if
		necessary and then outputs them sequentially with each
		call made to it. The batch size for the generation is
		determined by the nsamples argument given to the SMC
		constructor.
		
		Returns:
			Sample from the state posterior distribution.
		"""
		
		if self._usedsamples >= len(self._samples):
			(self._samples, self._like) = self._getparticles()
			self._usedsamples           = 0
		
		sample             = self._samples[self._usedsamples]
		self._usedsamples += 1
		
		return sample
	
	def _getparticles(self):
		"""Get particles representing the state posterior distribution.
		
		Particles will be generated from the prior function and updated
		through the hidden Markov model by filtering.
		
		Returns:
			Tuple (particles, likelihood).
			Particles from the state posterior distribution p(x_{1:t}|y_{1:t}).
			Likelihood of the observations given the model (i.e. the state
			transition model).
		"""
		
		particles, weights = self._particlesfromprior()
		likelihood         = 1.0
		
		for observation in observations:
			particles = self._predict(particles)
			weights   = self._measure(particles, weights, observation)
			
			# NOTE it is important that the predicted weights be normalized
			#      so the sum after the measurement, sum(w * p(y_t|x_t)),
			#      approximates int_x(p(y_t|x_t) * p(x_t|y_{1:t-1})) dx_t
			likelihood *= np.sum(weights)
			
			particles, weights = self._resample(particles, weights)
		
		return (particles, likelihood)
	
	def _particlesfromprior(self):
		"""Get particles from the prior state model.
		
		Get particles as an array of whatever the prior function outputs
		and their corresponding (uniform) weights.
		The only requirement for the prior function output is that it
		can be processed by the state transition and the sensor functions.
		
		Returns:
			Particles from the state prior distribution.
		"""
		
		invlen    = 1.0 / self._nsamples
		particles = []
		weights   = []
		
		for i in range(self._nsamples):
			particles.append(self._prior())
			weights.append(invlen)
		
		return (particles, weights)
	
	def _resample(self, particles, weights):
		"""Sample a new set of particles with equal weights.
		
		As the filtering moves along, some particles get stuck representing
		low probability regions, which are uninteresting considering those
		limited resources could be spent modelling other more important areas.
		To solve this situation, this function creates a new set of particles
		with the same probabilistic properties than the original, but discarding
		irrelevant particles and duplicating relevant ones. This is done by
		sampling from the distribution induced by the particles, so likely
		areas will probably get more particles and unlikely areas will not.
		The new particles' weights become uniform.
		
		Notes:
			Naively sampling from the distribution may once in a while give
			very bad results (sampling 1000 bad particles, although improbable,
			is possible), so the stochastic universal sampling algorithm is
			used instead. The unity interval [0, 1) is divided in segments
			according to the normalized weights. Then, from a random first
			value chosen in [0, 1/N), all N new particles are sampled by moving
			in fixed 1/N steps inside the unity interval.
			This sampling removes the possibility of catastrophic events but
			samples are correlated. Nevertheless, it works much better
			in practice.
			
			The arguments may be modified.
		Returns:
			Updated particles.
		"""
		
		invlen     = 1.0 / len(weights)
		newweights = np.normalize(weights)
		rand       = np.random_sample() * invlen
		
		k = 0
		for i in range(len(weights)):
			while rand >= 0 and k < len(weights):
				rand -= weights[k]
			
			newparticles[i] = particles[k - 1]
			newweights  [i] = invlen
			rand           += invlen
		
		return (particles, newweights)
	
	def _predict(self, particles):
		"""Predict step of the filter.
		
		Moves every particle according to the prediction step,
		x_{t+1} = f(x_t) + V_t.
		
		Notes:
			The argument may be modified.
		Returns:
			Updated particles.
		"""
		
		for i in range(len(particles)):
			particles[i] = self._ftransition(particles[i])
		
		return particles
	
	def _measure(self, particles, weights, measurement):
		"""Measurement step of the filter.
		
		Reweights each particle according to the posterior distribution
		induced by the new measurement, i.e. to represent
		p(x_{t+1}|y_{1:t+1}) instead of p(x_{t+1}|y_{1:t}).
		It uses the sensor model,
		y_t = h(x_t) + W_t,
		and assumes the particles have been sampled according to the
		previous prediction step (otherwise the factor would be different).
		
		Notes:
			The arguments may be modified.
		Returns:
			Updated particles.
		"""
		
		for i in range(len(weights)):
			weights[i] *= self._hsensor(particles[i][-1], measurement)
		
		return weights
