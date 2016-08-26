# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import pandas as pd
import scipy.constants as const
import scipy.interpolate as spi

import backports.lzma as xz


def readMap(filePath, compression=None):
    """ Reads tosca field map from file.

    Parameters
    ----------
    filePath : str
        Path to Tosca field map.
    compression : str (None)
        Which compression is used: None, 'xz'.

    Returns
    -------
    fieldMap : Pandas DataFrame
        Field map in Pandas DataFrame format.
    fieldMapInfo : dict
        Information about the read field map:
        grid dimensions, column names, column units.

    """
    # Need different open function based on compression.
    if not compression:
        jopen = open
    elif compression == 'xz':
        jopen = xz.open
    else:
        errorMsg = 'Compression {} not implemented.\n'.format(compression)
        errorMsg += "  Use: None or 'xz'."
        raise ValueError(errorMsg)

    # Read header of the field map.
    with jopen(filePath, 'r') as fi:
        lenZ, lenY, lenX, _ = (int(it) for it in fi.next().split())

        ls = fi.next().split()
        columns = []
        units = {}
        while ls[0] != '0':
            columns.append(ls[1])
            units[ls[1]] = ls[2]

            ls = fi.next().split()

    fieldMapInfo = {
        'gridDimensions': {'X': lenX, 'Y': lenY, 'Z': lenZ},
        'colNames': columns,
        'colUnits': units
    }

    # Read the field map.
    fieldMap = pd.read_csv(
        filePath, compression=compression,
        delim_whitespace=True, skipinitialspace=True,
        skiprows=len(columns)+2, header=None, names=columns
    )

    # Cross check header and field map.
    for axis, lenAxis in fieldMapInfo['gridDimensions'].items():
        if len(fieldMap[axis].unique()) != lenAxis:
            errorMsg = 'Header info does not match field map.\n'
            errorMsg += '  Length {}: {} vs {}.'.format(
                axis, lenAxis, len(fieldMap[axis].unique()))
            raise RuntimeError(errorMsg)
    if len(fieldMap.index) != lenX * lenY * lenZ:
        errorMsg = 'Header info does not match field map.\n'
        errorMsg += '  Total rows: {} vs {}.'.format(
            lenX * lenY * lenZ, len(fieldMap.index))
        raise RuntimeError(errorMsg)

    # Get grid boundaries.
    fieldMapInfo['gridBoundaries'] = {}
    for axis in fieldMapInfo['gridDimensions'].keys():
        axisMin = min(fieldMap[axis].unique())
        axisMax = max(fieldMap[axis].unique())
        fieldMapInfo['gridBoundaries'][axis] = [axisMin, axisMax]

    return fieldMap, fieldMapInfo


def writeMap(filePath, fieldMap, fieldMapInfo, compression=None):
    """ Writes tosca field map to file.

    Parameters
    ----------
    filePath : str
        Path to Tosca field map.
    fieldMap : Pandas DataFrame
        Field map in Pandas DataFrame format.
    fieldMapInfo : dict
        Information about the read field map.
    compression : str (None)
        Which compression to use: None, 'xz'.

    """
    # Need different open function based on compression.
    if not compression:
        jopen = open
    elif compression == 'xz':
        jopen = xz.open
    else:
        errorMsg = 'Compression {} not implemented.\n'.format(compression)
        errorMsg += "  Use: None or 'xz'."
        raise ValueError(errorMsg)

    with jopen(filePath, 'w') as fo:
        # Write field map header.
        fo.write('{} {} {} 2\n'.format(
            fieldMapInfo['gridDimensions']['Z'],
            fieldMapInfo['gridDimensions']['Y'],
            fieldMapInfo['gridDimensions']['X'],
        ))
        for i, column in enumerate(fieldMapInfo['colNames']):
            fo.write('{} {} {}\n'.format(
                i+1,
                column,
                fieldMapInfo['colUnits'][column]
            ))
        fo.write('0\n')

        # Write field map.
        fieldMap.to_csv(fo, sep=' ', header=False, index=False)


class TransformedFieldInterpolator:
    """ Interpolator over field map in transformed coordinate system.

    The field map is typically provided in its Tosca coordinate system, which
    typically does not match the coordinate system one works in. This class
    provides an interpolator that takes care of this transformation
    automatically.

    The transformation first applies a translation and then a rotation along
    y-axis. Rotation is counter-clockwise in z-x plane or clockwise in x-z
    plane.

    Return value is 0.0 for coordinates outside of the fieldMap.

    Attributes
    ----------
    translation : [float, float, float] ([0.0, 0.0, 0.0])
        Translation from working to Tosca system. In the same units as Tosca
        grid.
    rotationY : float (0.0)
        Rotation along y-axis from working to Tosca system. In radians.
    interpolator : SciPy RegularGridInterpolator
        Interpolates the field map. Don't call directly.

    Parameters
    ----------
    fieldMap : Pandas DataFrame
        Field map in Pandas DataFrame format.
    translation : [float, float, float] ([0.0, 0.0, 0.0])
        Translation from working to Tosca system. In the same units as Tosca
        grid.
    rotationY : float (0.0)
        Rotation along y-axis from working to Tosca system. In radians.
    axis : str ('BY')
        Which component to interpolate. Should be one of components in Tosca
        file.

    """
    def __init__(
        self, fieldMap,
        translation=[0.0, 0.0, 0.0], rotationY=0.0, axis='BY'
    ):
        self.translation = translation
        self.rotationY = rotationY

        # Get the grid ticks.
        xGrid = np.sort(fieldMap['X'].unique())
        yGrid = np.sort(fieldMap['Y'].unique())
        zGrid = np.sort(fieldMap['Z'].unique())
        # Construct a 3D grid of magnetic field.
        magField = fieldMap[axis].values.reshape((
            len(xGrid), len(yGrid), len(zGrid)))

        self.interpolator = spi.RegularGridInterpolator(
            (xGrid, yGrid, zGrid), magField,
            bounds_error=False, fill_value=0.0
        )

    def transform(self, x, y, z):
        """ Transforms coordinates from working to Tosca coordinate system.

        Parameters
        ----------
        x, y, z : float
            Coordinates in the working coordinate system.

        Returns
        -------
        xt, yt, zy : float
            Coordinates in the Tosca coordinate system.

        """
        xt = x - self.translation[0]
        yt = y - self.translation[1]
        zt = z - self.translation[2]

        cos = np.cos(-self.rotationY)
        sin = np.sin(-self.rotationY)

        xt, zt = cos*xt + sin*zt, cos*zt - sin*xt

        return xt, yt, zt

    def invTransform(self, xt, yt, zt):
        """ Transforms coordinates from Tosca to working coordinate system.

        Parameters
        ----------
        xt, yt, zy : float
            Coordinates in the Tosca coordinate system.

        Returns
        -------
        x, y, z : float
            Coordinates in the working coordinate system.

        """
        cos = np.cos(self.rotationY)
        sin = np.sin(self.rotationY)

        x, y, z = cos*xt + sin*zt, yt, cos*zt - sin*xt

        x = x + self.translation[0]
        y = y + self.translation[1]
        z = z + self.translation[2]

        return x, y, z

    def getValue(self, x, y, z):
        """ Gets interpolated value of the magnetic field from interpolator.

        Parameters
        ----------
        x, y, z : float or list-like
            Coordinates in the working coordinate system.

        Returns
        -------
        B : NumPy array of floats
            Magnetic fields corresponding to input coordinates.

        """
        xt, yt, zt = self.transform(x, y, z)

        B = np.array([xt, yt, zt]).T

        return self.interpolator(B)


def raytrace_01(
    fieldInterpolators,
    beamEnergy=11.0, q=-1,
    xInit=0.0, yInit=0.0, zInit=0.0, thetaInit=0.0,
    trackLimit=1300.0, stepSize=0.5
):
    """ A relatively simple charged particle raytracer.

    This version of raytracer takes into account only y component of the
    magnetic field. The particle is only allowed to move in the x-z plane.
    The tracking is done by calculating a deflection angle at each step of the
    track. Particle is assumed to be relativistic.

    Parameters
    ----------
    fieldInterpolators : dictionary of TransformedFieldInterpolator
        Interpolators for y component of the magnetic field. At each step a
        sum of each interpolator is used to calculate deflection angle. Assumed
        to accept coordinates in cm and return field in Gauss.
    beamEnergy : float (11.0)
        Beam energy in GeV.
    q : int (-1)
        Charge of the particle being tracked.
    xInit, yInit, zInit : float (0.0)
        Starting point of the particle. In cm.
    thetaInit : float (0.0)
        Starting angle of the particle. Theta is directed from positive z
        towards positive x. In degrees.
    trackLimit : float (1300.0)
        How far to track the particle. In cm.
    stepSize : float (0.0)
        Step size for tracking. In cm.

    Returns
    -------
    position : list of NumPy array
        Track of the particle. In cm.
    th : NumPy array
        Angle of the particle along track. In radians.
    yFields : dictionary of NumPy array
        Magnetic fields along the track for each field interpolator and for a
        sum. In Gauss.
    l : NumPy array
        Track length along the track. In cm.

    """
    # Initialize tracking.
    nSteps = int(trackLimit / stepSize + 1)
    l = np.zeros(nSteps)

    x, y, z, th = np.zeros((4, nSteps))
    x[0], y[0], z[0], th[0] = xInit, yInit, zInit, thetaInit*const.degree

    yFields = {mag: np.zeros(nSteps) for mag in fieldInterpolators.keys()}
    yFields['all'] = np.zeros(nSteps)
    for mag, interpolator in fieldInterpolators.items():
        yFields[mag][0] = interpolator.getValue(xInit, yInit, zInit)[0]
        yFields['all'][0] += yFields[mag][0]

    # Calculate the 'bend factor'.
    # qc/E : rad * GeV / G*cm
    # need -1 because of coordinate system...
    Gauss2Tesla = 1E-4
    qc_E = -q*const.c/(beamEnergy*const.giga) * Gauss2Tesla*const.centi

    # Track particle.
    for i in xrange(1, nSteps):
        l[i] = l[i-1] + stepSize
        x[i] = x[i-1] + stepSize*np.sin(th[i-1])
        y[i] = y[i-1]
        z[i] = z[i-1] + stepSize*np.cos(th[i-1])

        # Get magnetic fields at current location.
        for mag, interpolator in fieldInterpolators.items():
            yFields[mag][i] = interpolator.getValue(x[i], y[i], z[i])[0]
            yFields['all'][i] += yFields[mag][i]
        # Calculate the bend due to Bdl along the step.
        th[i] = (
            th[i-1] + qc_E * stepSize*(yFields['all'][i-1]+yFields['all'][i])/2
        )

    return [x, y, z], th, yFields, l
