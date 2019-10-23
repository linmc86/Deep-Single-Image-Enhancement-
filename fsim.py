#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2016, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2016. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################
"""Objects and methods for computing the quality of reconstructions.
.. moduleauthor:: Daniel J Ching <carterbox@users.noreply.github.com>
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import phasepack as phase

from scipy import ndimage
from scipy import optimize
from scipy import stats
from copy import deepcopy


logger = logging.getLogger(__name__)
__docformat__ = 'restructuredtext en'



def compute_fsim(img0, img1, nlevels=5, nwavelets=16, L=None):
    """FSIM Index with automatic downsampling, Version 1.0
    An implementation of the algorithm for calculating the Feature SIMilarity
    (FSIM) index was ported to Python. This implementation only considers the
    luminance component of images. For multichannel images, convert to
    grayscale first. Dynamic range should be 0-255.
    Parameters
    ----------
    img0 : array
    img1 : array
        Two images for comparison.
    nlevels : scalar
        The number of levels to measure quality.
    nwavelets : scalar
        The number of wavelets to use in the phase congruency calculation.
    Returns
    -------
    metrics : dict
        A dictionary with image quality information organized by scale.
        ``metric[scale] = (mean_quality, quality_map)``
        The valid range for FSIM is (0, 1].
    """
    _full_reference_input_check(img0, img1, 1.2, nlevels, L)
    if nwavelets < 1:
        raise ValueError('There must be at least one wavelet level.')

    Y1 = img0
    Y2 = img1

    scales = np.zeros(nlevels)
    mets = np.zeros(nlevels)
    maps = [None] * nlevels

    for level in range(0, nlevels):
        # sigma = 1.2 is approximately correct because the width of the scharr
        # and min wavelet filter (phase congruency filter) is 3.
        sigma = 1.2 * 2**level

        F = 2  # Downsample (using ndimage.zoom to prevent sampling bias)
        Y1 = ndimage.zoom(Y1, 1/F)
        Y2 = ndimage.zoom(Y2, 1/F)

        # Calculate the phase congruency maps
        [PC1, Orient1, ft1, T1] = phase.phasecongmono(Y1, nscale=nwavelets)
        [PC2, Orient2, ft2, T2] = phase.phasecongmono(Y2, nscale=nwavelets)

        # Calculate the gradient magnitude map using Scharr filters
        dx = np.array([[3., 0., -3.],
                       [10., 0., -10.],
                       [3., 0., -3.]]) / 16
        dy = np.array([[3., 10., 3.],
                       [0., 0., 0.],
                       [-3., -10., -3.]]) / 16

        IxY1 = ndimage.filters.convolve(Y1, dx)
        IyY1 = ndimage.filters.convolve(Y1, dy)
        gradientMap1 = np.sqrt(IxY1**2 + IyY1**2)

        IxY2 = ndimage.filters.convolve(Y2, dx)
        IyY2 = ndimage.filters.convolve(Y2, dy)
        gradientMap2 = np.sqrt(IxY2**2 + IyY2**2)

        # Calculate the FSIM
        T1 = 0.85   # fixed and depends on dynamic range of PC values
        T2 = 160    # fixed and depends on dynamic range of GM values
        PCSimMatrix = (2 * PC1 * PC2 + T1) / (PC1**2 + PC2**2 + T1)
        gradientSimMatrix = ((2 * gradientMap1 * gradientMap2 + T2) /
                             (gradientMap1**2 + gradientMap2**2 + T2))
        PCm = np.maximum(PC1, PC2)
        FSIMmap = gradientSimMatrix * PCSimMatrix
        FSIM = np.sum(FSIMmap * PCm) / np.sum(PCm)

        scales[level] = sigma
        mets[level] = FSIM
        maps[level] = FSIMmap

    return scales, mets, maps


def _full_reference_input_check(img0, img1, sigma, nlevels, L):
    """Checks full reference quality measures for valid inputs."""
    if nlevels <= 0:
        raise ValueError('nlevels must be >= 1.')
    if sigma < 1.2:
        raise ValueError('sigma < 1.2 is effective meaningless.')
    if np.min(img0.shape) / (2**(nlevels - 1)) < sigma * 2:
        raise ValueError("{nlevels} levels makes {shape} smaller than a filter"
                         " size of 2 * {sigma}".format(nlevels=nlevels,
                                                       shape=img0.shape,
                                                       sigma=sigma))
    if L is not None and L < 1:
        raise ValueError("Dynamic range must be >= 1.")
    if img0.shape != img1.shape:
        raise ValueError("original and reconstruction should be the " + "same shape")