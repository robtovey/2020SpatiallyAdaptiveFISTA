'''
Created on 9 Apr 2020

@author: Rob
'''
from numba import jit, prange
from numpy import array, sqrt
_PARALLEL = True
__params = {'nopython':True, 'parallel':False, 'fastmath':True, 'cache':True, 'boundscheck':False}
__pparams = __params.copy(); __pparams['parallel'] = True


@jit(**__params)
def len_intersect(x0, x1, y0, y1): return max(min(x1, y1) - max(x0, y0), 0)


@jit(**__pparams)
def _gradF(dof_map, grid, L, R, DF):
    for i in prange(1, dof_map.shape[0]):
        j = dof_map[i]
        supp0, supp1 = j[1] * 2 ** float(1 - j[0]), (j[1] + 1) * 2 ** float(1 - j[0])
        midpoint = .5 * (supp0 + supp1)

        v = 0
        for k in range(L.size):
            v += R[k] / sqrt(L[k]) * (
                len_intersect(midpoint, supp1, grid[k], grid[k + 1])
                -len_intersect(supp0, midpoint, grid[k], grid[k + 1]))

        DF[i] = 2 ** ((j[0] - 1) / 2) * v


@jit(**__params)
def line_intersect_square(px, py, d, x0, y0, x1, y1):
    '''
    The line is p + t*d for t in R
    The box is bound by the corners [x0,y0] and [x1,y1]

    There are 4 possible intersection points:
        (X0,y0), (X1,y1), (x0,Y0), (x1,Y1)

    We always return the area above (positive y) the line
    If we flip the x axis then we never change the area
    If we flip the y axis then we always change the area of integration
    Using this, we can reduce to the case x0<x1, X0<X1, y0<y1, Y0<Y1

    '''

    # TODO: deal with d[i]=0
    Y0 = py + (x0 - px) / d[0] * d[1]
    Y1 = py + (x1 - px) / d[0] * d[1]
    X0 = px + (y0 - py) / d[1] * d[0]
    X1 = px + (y1 - py) / d[1] * d[0]

    sign = False
    # WLOG X0<X1, Y0<Y1
    if X1 < X0:  # flip in Y-axis
        tmp = X1
        X1 = X0
        X0 = tmp
        Y0, Y1, sign = y1 + y0 - Y0, y1 + y0 - Y1, not sign

    if X1 < x0:  # Line passes above the square
        v = 0
    elif X0 > x1:  # Line passes below the square
        v = (x1 - x0) * (y1 - y0)
    elif X0 >= x0 and X1 <= x1:  # Line passes through top and bottom
        v = (y1 - y0) * (.5 * (X1 + X0) - x0)
    elif Y0 >= y0 and Y1 <= y1:  # Line passes through left and right
        v = (x1 - x0) * (y1 - .5 * (Y1 + Y0))
    elif X0 < x0:  # Line cuts top left corner
        v = .5 * (X1 - x0) * (y1 - Y0)
    else:  # Line cuts bottom right corner
        sign = not sign
        v = .5 * (x1 - X0) * (Y1 - y0)

    if sign:
        v = (x1 - x0) * (y1 - y0) - v

    return v


@jit(**__pparams)
def _FP_tomo(wave, dof_map, g, slope, centre, sino):
    for j in prange(sino.shape[0]):
        # First wavelet is special, just constant
        # Bounding box for support:
        b0x, b0y = dof_map[0, 0] - centre[0], dof_map[0, 1] - centre[1]
        b1x, b1y = b0x + 1, b0y + 1

        # First wavelet effects all pixels in sinogram
        for k in range(g.size - 1):
            p00, p01 = g[k] * slope[j, 1], -g[k] * slope[j, 0]
            p10, p11 = g[k + 1] * slope[j, 1], -g[k + 1] * slope[j, 0]

            sino[j, k] = (
                wave[0, 0] * abs(
                line_intersect_square(p10, p11, slope[j], b0x, b0y, b1x, b1y)
                -line_intersect_square(p00, p01, slope[j], b0x, b0y, b1x, b1y))
            )

        for i in range(dof_map.shape[0]):
            I = i + 1  # index in wave
            h = dof_map[i, 2]  # width of wavelet
            # Bounding box of wavelet:
            b0x, b0y = dof_map[i, 0] - centre[0], dof_map[i, 1] - centre[1]
            b1x, b1y = b0x + h, b0y + h
            midx, midy = b0x + .5 * h, b0y + .5 * h  # midpoints
            r = 2.**(-.5) * h  # diagonal radius of wavelet
            scale = 1 / h  # L2 scaling of wavelets

            # midpoint of wavelet projected onto detector coordinate
            t = midx * slope[j, 1] - midy * slope[j, 0]
            for k in range(g.size - 1):
                if g[k] > t + r:
                    break  # outside of support
                elif g[k + 1] > t - r:
                    # Bounding lines over which to integrate over
                    p00, p01 = g[k] * slope[j, 1], -g[k] * slope[j, 0]
                    p10, p11 = g[k + 1] * slope[j, 1], -g[k + 1] * slope[j, 0]

                    sino[j, k] += (
                        (-wave[I, 0] - wave[I, 1] + wave[I, 2]) * abs(
                        line_intersect_square(p10, p11, slope[j], b0x, b0y, midx, midy)
                        -line_intersect_square(p00, p01, slope[j], b0x, b0y, midx, midy))
                        +(-wave[I, 0] + wave[I, 1] - wave[I, 2]) * abs(
                        line_intersect_square(p10, p11, slope[j], b0x, midy, midx, b1y)
                        -line_intersect_square(p00, p01, slope[j], b0x, midy, midx, b1y))
                        +(+wave[I, 0] - wave[I, 1] - wave[I, 2]) * abs(
                        line_intersect_square(p10, p11, slope[j], midx, b0y, b1x, midy)
                        -line_intersect_square(p00, p01, slope[j], midx, b0y, b1x, midy))
                        +(+wave[I, 0] + wave[I, 1] + wave[I, 2]) * abs(
                        line_intersect_square(p10, p11, slope[j], midx, midy, b1x, b1y)
                        -line_intersect_square(p00, p01, slope[j], midx, midy, b1x, b1y))
                    ) * scale


@jit(**__params)
def _Radon_row(i, p00, p01, p10, p11, slope, b0x, midx, b1x, b0y, midy, b1y):
    if i == 0:
        return (
                # bottom left
                -abs(line_intersect_square(p10, p11, slope, b0x, b0y, midx, midy)
                    -line_intersect_square(p00, p01, slope, b0x, b0y, midx, midy))
                # top left
                -abs(line_intersect_square(p10, p11, slope, b0x, midy, midx, b1y)
                    -line_intersect_square(p00, p01, slope, b0x, midy, midx, b1y))
                # bottom right
                +abs(line_intersect_square(p10, p11, slope, midx, b0y, b1x, midy)
                    -line_intersect_square(p00, p01, slope, midx, b0y, b1x, midy))
                # top right
                +abs(line_intersect_square(p10, p11, slope, midx, midy, b1x, b1y)
                    -line_intersect_square(p00, p01, slope, midx, midy, b1x, b1y))
                )
    elif i == 1:
        return (
                # bottom left
                -abs(line_intersect_square(p10, p11, slope, b0x, b0y, midx, midy)
                    -line_intersect_square(p00, p01, slope, b0x, b0y, midx, midy))
                # top left
                +abs(line_intersect_square(p10, p11, slope, b0x, midy, midx, b1y)
                    -line_intersect_square(p00, p01, slope, b0x, midy, midx, b1y))
                # bottom right
                -abs(line_intersect_square(p10, p11, slope, midx, b0y, b1x, midy)
                    -line_intersect_square(p00, p01, slope, midx, b0y, b1x, midy))
                # top right
                +abs(line_intersect_square(p10, p11, slope, midx, midy, b1x, b1y)
                    -line_intersect_square(p00, p01, slope, midx, midy, b1x, b1y))
                )
    else:
        return (
                # bottom left
                +abs(line_intersect_square(p10, p11, slope, b0x, b0y, midx, midy)
                    -line_intersect_square(p00, p01, slope, b0x, b0y, midx, midy))
                # top left
                -abs(line_intersect_square(p10, p11, slope, b0x, midy, midx, b1y)
                    -line_intersect_square(p00, p01, slope, b0x, midy, midx, b1y))
                # bottom right
                -abs(line_intersect_square(p10, p11, slope, midx, b0y, b1x, midy)
                    -line_intersect_square(p00, p01, slope, midx, b0y, b1x, midy))
                # top right
                +abs(line_intersect_square(p10, p11, slope, midx, midy, b1x, b1y)
                    -line_intersect_square(p00, p01, slope, midx, midy, b1x, b1y)))


@jit(**__params)
def Radon_to_sparse_matrix(data, indcs, ptr, dof_map, g, slope, centre):
    sz = g.size - 1
    indx = 0  # current index of matrix

    # constant pixel
    b0x, b0y, b1x, b1y = 0 - centre[0], 0 - centre[1], 1 - centre[0], 1 - centre[1]
    ptr[0] = indx
    for i0 in range(slope.shape[0]):
        for i1 in range(g.size - 1):
            p00, p01 = g[i1] * slope[i0, 1], -g[i1] * slope[i0, 0]
            p10, p11 = g[i1 + 1] * slope[i0, 1], -g[i1 + 1] * slope[i0, 0]

            data[indx] = abs(line_intersect_square(p10, p11, slope[i0], b0x, b0y, b1x, b1y)
                    -line_intersect_square(p00, p01, slope[i0], b0x, b0y, b1x, b1y))
            indcs[indx] = i0 * sz + i1
            indx += 1
    ptr[1] = indx  # empty row
    ptr[2] = indx  # empty row

    # wavelet pixels
    for j in range(dof_map.shape[0]):
        jj = j + 1  # index of wave
        h = dof_map[j, 2]  # width of wavelet
        # Bounding box of wavelet:
        b0x, b0y = dof_map[j, 0] - centre[0], dof_map[j, 1] - centre[1]
        b1x, b1y = b0x + h, b0y + h
        midx, midy = b0x + .5 * h, b0y + .5 * h  # midpoints
        r = 2.**(-.5) * h  # diagonal radius of wavelet
        scale = 1 / h  # L2 scaling of wavelets

        for i in range(3):  # for each wavelet type
            ptr[3 * jj + i] = indx  # start of new row
            for i0 in range(slope.shape[0]):
                t = midx * slope[i0, 1] - midy * slope[i0, 0]
                for i1 in range(g.size - 1):
                    if g[i1] > t + r:
                        break
                    elif g[i1 + 1] > t - r:
                        p00, p01 = g[i1] * slope[i0, 1], -g[i1] * slope[i0, 0]
                        p10, p11 = g[i1 + 1] * slope[i0, 1], -g[i1 + 1] * slope[i0, 0]

                        data[indx] = scale * _Radon_row(i, p00, p01, p10, p11, slope[i0],
                                                      b0x, midx, b1x, b0y, midy, b1y)
                        indcs[indx] = i0 * sz + i1
                        indx += 1
                        if indx >= data.size:
                            raise MemoryError()

    ptr[3 * dof_map.shape[0] + 3] = indx  # end of data arrays


@jit(**__params)
def Radon_update_sparse_matrix(o_data, o_indcs, o_ptr, refine,
        n_data, n_indcs, n_ptr, dof_map, g, slope, centre):
    sz = g.size - 1
    J = indx = 0  # current wavelet/index of new matrix

    for j in range(refine.shape[0]):  # current wavelet of old matrix

        # always copy old rows to new
        for i in range(3):
            n_ptr[3 * J + i] = indx  # row J of new matrix starts at index indx
            for jj in range(o_ptr[3 * j + i], o_ptr[3 * j + i + 1]):
                n_data[indx] = o_data[jj]
                n_indcs[indx] = o_indcs[jj]
                indx += 1
                if indx >= n_data.size:
                    raise MemoryError()
        J += 1  # move to new wavelet in new matrix

        if refine[j] > 0:  # create new rows
            h = dof_map[J - 1, 2]  # width of new wavelets
            r = 2.**(-.5) * h  # diagonal radius of wavelet
            scale = 1 / h  # L2 scaling of wavelets

            for _ in range(4):  # one pixel splits into 4 consecutive ones
                # Bounding box of wavelet:
                b0x, b0y = dof_map[J - 1, 0] - centre[0], dof_map[J - 1, 1] - centre[1]
                b1x, b1y = b0x + h, b0y + h
                midx, midy = b0x + .5 * h, b0y + .5 * h  # midpoints

                for i in range(3):  # for each wavelet type
                    n_ptr[3 * J + i] = indx  # start of new row
                    for i0 in range(slope.shape[0]):
                        t = midx * slope[i0, 1] - midy * slope[i0, 0]
                        for i1 in range(g.size - 1):
                            if g[i1] > t + r:
                                break
                            elif g[i1 + 1] > t - r:
                                p00, p01 = g[i1] * slope[i0, 1], -g[i1] * slope[i0, 0]
                                p10, p11 = g[i1 + 1] * slope[i0, 1], -g[i1 + 1] * slope[i0, 0]

                                n_data[indx] = scale * _Radon_row(i, p00, p01, p10, p11, slope[i0],
                                                              b0x, midx, b1x, b0y, midy, b1y)
                                n_indcs[indx] = i0 * sz + i1
                                if abs(n_data[indx]) > 1e-14:
                                    indx += 1
                                if indx >= n_data.size:
                                    raise MemoryError()
                J += 1  # move to new wavelet in new matrix

    n_ptr[3 * J] = indx  # end of data arrays


@jit(**__pparams)
def _BP_tomo(sino, dof_map, g, slope, centre, wave):
    # First wavelet is special, just constant
    # Bounding box for support:
    b0x, b0y = dof_map[0, 0] - centre[0], dof_map[0, 1] - centre[1]
    b1x, b1y = b0x + 1, b0y + 1

    # First wavelet effects all pixels in sinogram
    wave[0, 0] = 0
    for j in range(slope.shape[0]):
        for k in range(g.size - 1):
            p00, p01 = g[k] * slope[j, 1], -g[k] * slope[j, 0]
            p10, p11 = g[k + 1] * slope[j, 1], -g[k + 1] * slope[j, 0]

            wave[0, 0] += (
                sino[j, k] * abs(
                line_intersect_square(p10, p11, slope[j], b0x, b0y, b1x, b1y)
                -line_intersect_square(p00, p01, slope[j], b0x, b0y, b1x, b1y))
            )

    for i in prange(dof_map.shape[0]):
        I = i + 1  # index in wave
        h = dof_map[i, 2]  # width of wavelet
        # Bounding box of wavelet:
        b0x, b0y = dof_map[i, 0] - centre[0], dof_map[i, 1] - centre[1]
        b1x, b1y = b0x + h, b0y + h
        midx, midy = b0x + .5 * h, b0y + .5 * h  # midpoints
        r = 2.**(-.5) * h  # diagonal radius of wavelet
        scale = 1 / h  # L2 scaling of wavelets

        v0 = 0; v1 = 0; v2 = 0; v3 = 0
        for j in range(slope.shape[0]):
            # midpoint of wavelet projected onto detector coordinate
            t = midx * slope[j, 1] - midy * slope[j, 0]
            for k in range(g.size - 1):
                if g[k] > t + r:
                    break  # outside of support
                elif g[k + 1] > t - r:
                    # Bounding lines over which to integrate over
                    p00, p01 = g[k] * slope[j, 1], -g[k] * slope[j, 0]
                    p10, p11 = g[k + 1] * slope[j, 1], -g[k + 1] * slope[j, 0]

                    v0 += sino[j, k] * abs(line_intersect_square(p10, p11, slope[j], b0x, b0y, midx, midy)
                        -line_intersect_square(p00, p01, slope[j], b0x, b0y, midx, midy))
                    v1 += sino[j, k] * abs(line_intersect_square(p10, p11, slope[j], b0x, midy, midx, b1y)
                        -line_intersect_square(p00, p01, slope[j], b0x, midy, midx, b1y))
                    v2 += sino[j, k] * abs(line_intersect_square(p10, p11, slope[j], midx, b0y, b1x, midy)
                        -line_intersect_square(p00, p01, slope[j], midx, b0y, b1x, midy))
                    v3 += sino[j, k] * abs(line_intersect_square(p10, p11, slope[j], midx, midy, b1x, b1y)
                        -line_intersect_square(p00, p01, slope[j], midx, midy, b1x, b1y))

        wave[I, 0] = (-v0 - v1 + v2 + v3) * scale
        wave[I, 1] = (-v0 + v1 - v2 + v3) * scale
        wave[I, 2] = (+v0 - v1 - v2 + v3) * scale


@jit(**__params)
def _smallgrad(df, u, a, eps):
    for i in prange(df.size):
        if u[i] > eps:  # u>0
            df[i] += a
        elif u[i] < -eps:  # u<0
            df[i] -= a
        elif df[i] > a:  # u=0, df>a
            df[i] -= a
        elif df[i] < -a:  # u=0, df<-a
            df[i] += a
        else:
            df[i] = 0

###################################################################################################
# # Copyright (C) 2010  Alex Opie  <lx_op@orcon.net.nz>
# #
# # This program is free software; you can redistribute it and/or modify it
# # under the terms of the GNU General Public License as published by
# # the Free Software Foundation; either version 3 of the License, or (at
# # your option) any later version.
# #
# # This program is distributed in the hope that it will be useful, but
# # WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the GNU
# # General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program; see the file COPYING.  If not, see
# # <http://www.gnu.org/licenses/>.


import numpy as np


def phantom (n=256, p_type='Modified Shepp-Logan', ellipses=None):
    """
     phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)

    Create a Shepp-Logan or modified Shepp-Logan phantom.

    A phantom is a known object (either real or purely mathematical)
    that is used for testing image reconstruction algorithms.  The
    Shepp-Logan phantom is a popular mathematical model of a cranial
    slice, made up of a set of ellipses.  This allows rigorous
    testing of computed tomography (CT) algorithms as it can be
    analytically transformed with the radon transform (see the
    function `radon').

    Inputs
    ------
    n : The edge length of the square image to be produced.

    p_type : The type of phantom to produce. Either
      "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
      if `ellipses' is also specified.

    ellipses : Custom set of ellipses to use.  These should be in
      the form
          [[I, a, b, x0, y0, phi],
           [I, a, b, x0, y0, phi],
           ...]
      where each row defines an ellipse.
      I : Additive intensity of the ellipse.
      a : Length of the major axis.
      b : Length of the minor axis.
      x0 : Horizontal offset of the centre of the ellipse.
      y0 : Vertical offset of the centre of the ellipse.
      phi : Counterclockwise rotation of the ellipse in degrees,
            measured as the angle between the horizontal axis and
            the ellipse major axis.
      The image bounding box in the algorithm is [-1, -1], [1, 1],
      so the values of a, b, x0, y0 should all be specified with
      respect to this box.

    Output
    ------
    P : A phantom image.

    Usage example
    -------------
      import matplotlib.pyplot as pl
      P = phantom ()
      pl.imshow (P)

    References
    ----------
    Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
    from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
    Feb. 1974, p. 232.

    Toft, P.; "The Radon Transform - Theory and Implementation",
    Ph.D. thesis, Department of Mathematical Modelling, Technical
    University of Denmark, June 1996.

    """

    if (ellipses is None):
        ellipses = _select_phantom (p_type)
    elif (np.size (ellipses, 1) != 6):
        raise AssertionError ("Wrong number of columns in user phantom")

    # Blank image
    p = np.zeros ((n, n))

    # Create the pixel grid
    ygrid, xgrid = np.mgrid[-1:1:(1j * n), -1:1:(1j * n)]

    for ellip in ellipses:
        I = ellip [0]
        a2 = ellip [1] ** 2
        b2 = ellip [2] ** 2
        x0 = ellip [3]
        y0 = ellip [4]
        phi = ellip [5] * np.pi / 180  # Rotation angle in radians

        # Create the offset x and y values for the grid
        x = xgrid - x0
        y = ygrid - y0

        cos_p = np.cos (phi)
        sin_p = np.sin (phi)

        # Find the pixels within the ellipse
        locs = (((x * cos_p + y * sin_p) ** 2) / a2
              +((y * cos_p - x * sin_p) ** 2) / b2) <= 1

        # Add the ellipse intensity to those pixels
        p [locs] += I

    return p


def _select_phantom (name):
    if (name.lower () == 'shepp-logan'):
        e = _shepp_logan ()
    elif (name.lower () == 'modified shepp-logan'):
        e = _mod_shepp_logan ()
    else:
        raise ValueError ("Unknown phantom type: %s" % name)

    return e


def _shepp_logan ():
    #  Standard head phantom, taken from Shepp & Logan
    return [[   2, .69, .92, 0, 0, 0],
            [-.98, .6624, .8740, 0, -.0184, 0],
            [-.02, .1100, .3100, .22, 0, -18],
            [-.02, .1600, .4100, -.22, 0, 18],
            [ .01, .2100, .2500, 0, .35, 0],
            [ .01, .0460, .0460, 0, .1, 0],
            [ .02, .0460, .0460, 0, -.1, 0],
            [ .01, .0460, .0230, -.08, -.605, 0],
            [ .01, .0230, .0230, 0, -.606, 0],
            [ .01, .0230, .0460, .06, -.605, 0]]


def _mod_shepp_logan ():
    #  Modified version of Shepp & Logan's head phantom,
    #  adjusted to improve contrast.  Taken from Toft.
    return [[   1, .69, .92, 0, 0, 0],
            [-.80, .6624, .8740, 0, -.0184, 0],
            [-.20, .1100, .3100, .22, 0, -18],
            [-.20, .1600, .4100, -.22, 0, 18],
            [ .10, .2100, .2500, 0, .35, 0],
            [ .10, .0460, .0460, 0, .1, 0],
            [ .10, .0460, .0460, 0, -.1, 0],
            [ .10, .0460, .0230, -.08, -.605, 0],
            [ .10, .0230, .0230, 0, -.606, 0],
            [ .10, .0230, .0460, .06, -.605, 0]]
###################################################################################################
