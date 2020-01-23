#!/usr/bin/env python3

import numpy as np
import re
import sys
import logging
import argparse
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.time import Time
from scipy.optimize import curve_fit

from stickel import Stickel

logger = logging.getLogger(__name__)

#---------------------------------------------------------------
class LittleClipError(Exception):
    """Raise when not enough data is clipped"""
    pass

#---------------------------------------------------------------
class LargeClipError(Exception):
    """Raise when too much data is clipped"""
    pass
class NoComponentsError(Exception):
    """Raise when there are no feasible profile components"""
    pass

#---------------------------------------------------------------
def get_from_bestprof(file_loc):
    """
    Get info from a bestprof file

    Parameters:
    -----------
    file_loc: string
        The path to the bestprof file

    Returns:
    --------
    [obsid, pulsar, dm, period, period_uncer, obsstart, obslength, profile, bin_num]: list
        obsid: int
            The observation ID
        pulsar: string
            The name of the pulsar
        dm: float
            The dispersion measure of the pulsar
        period: float
            The period of the pulsar
        period_uncer: float
            The uncertainty in the period measurement
        obsstart: int
            The beginning time of the observation
        obslength: float
            The length of the observation in seconds
        profile: list
            A list of floats containing the profile data
        bin_num: int
            The number of bins in the profile
    """

    with open(file_loc,"r") as bestprof:
        lines = bestprof.readlines()
        # Find the obsid by finding a 10 digit int in the file name
        obsid = re.findall(r'(\d{10})', lines[0])[0]
        try:
            obsid = int(obsid)
        except ValueError:
            obsid = None

        pulsar = str(lines[1].split("_")[-1][:-1])
        if not (pulsar.startswith('J') or pulsar.startswith('B')):
            pulsar = 'J{0}'.format(pulsar)

        dm = lines[14][22:-1]

        period = lines[15][22:-1]
        period, period_uncer = period.split('  +/- ')

        mjdstart = Time(float(lines[3][22:-1]), format='mjd', scale='utc')
        # Convert to gps time
        obsstart = int(mjdstart.gps)

        # Get obs length in seconds by multipling samples by time per sample
        obslength = float(lines[6][22:-1])*float(lines[5][22:-1])

        # Get the pulse profile
        orig_profile = []
        for l in lines[27:]:
            orig_profile.append(float(l.split()[-1]))
        bin_num = len(orig_profile)
        profile = np.zeros(bin_num)

        # Remove min
        min_prof = min(orig_profile)
        for p, _ in enumerate(orig_profile):
            profile[p] = orig_profile[p] - min_prof

    return [obsid, pulsar, dm, period, period_uncer, obsstart, obslength, profile, bin_num]

#---------------------------------------------------------------
def get_from_ascii(file_loc):
    """
    Retrieves the profile from an ascii file

    Parameters:
    -----------
    file_loc: string
        The location of the ascii file

    Returns:
    --------
    [profile, len(profile)]: list
        profile: list
            A list of floats containing the profile data
        len(profile): int
            The number of bins in the profile
    """

    f = open(file_loc)
    lines = iter(f.readlines())
    next(lines) #skip first line
    profile=[]
    for line in lines:
        thisline=line.split()
        profile.append(float(thisline[3]))

    return [profile, len(profile)]

#---------------------------------------------------------------
def get_stokes_from_ascii(file_loc):
    """
    Retrieves the all stokes components from an ascii file

    Parameters:
    -----------
    file_loc: string
        The location of the ascii file

    Returns:
    --------
    [I, Q, U, V, len(profile)]: list
        I: list
            Stokes I
        Q: list
            Stokes Q
        U: list
            Stokes U
        V: list
            Stokes V
        len(profile): int
            The number of bins in the profile
    """
    f = open(file_loc)
    lines = iter(f.readlines())
    next(lines) #skip first line
    I=[]
    Q=[]
    U=[]
    V=[]
    for line in lines:
        thisline=line.split()
        I.append(float(thisline[3]))
        Q.append(float(thisline[4]))
        U.append(float(thisline[5]))
        V.append(float(thisline[6]))

    return [I, Q, U, V, len(I)]

#---------------------------------------------------------------
def sigmaClip(data, alpha=3, tol=0.1, ntrials=10):
    """
    Sigma clipping operation:
    Compute the data's median, m, and its standard deviation, sigma.
    Keep only the data that falls in the range (m-alpha*sigma,m+alpha*sigma) for some value of alpha, and discard everything else.

    This operation is repeated ntrials number of times or until the tolerance level is hit.

    Parameters:
    -----------
    data: list
        A list of floats - the data to clip
    alpha: float
        OPTIONAL - Determines the number of sigmas to use to determine the upper and lower limits. Default=3
    tol: float
        OPTIONAL - The fractional change in the standard deviation that determines when the tolerance is hit. Default=0.1
    ntrils: int
        OPTIONAL - The maximum number of times to apply the operation. Default=10

    Returns:
    --------
    oldstd: float
        The std of the clipped data
    x: list
        The data list that contains only noise, with nans in place of 'real' data
    """
    x = np.copy(data)
    oldstd = np.nanstd(x)
    #When the x[x<lolim] and x[x>hilim] commands encounter a nan it produces a
    #warning. This is expected because it is ignoring flagged data from a
    #previous trial so the warning is supressed.
    old_settings = np.seterr(all='ignore')
    for trial in range(ntrials):
        median = np.nanmedian(x)
        lolim = median - alpha * oldstd
        hilim = median + alpha * oldstd
        x[x<lolim] = np.nan
        x[x>hilim] = np.nan

        newstd = np.nanstd(x)
        tollvl = (oldstd - newstd) / newstd

        if tollvl <= tol:
            logger.debug("Took {0} trials to reach tolerance".format(trial+1))
            np.seterr(**old_settings)
            return oldstd, x

        if trial + 1 == ntrials:
            logger.warn("Reached number of trials without reaching tolerance level")
            np.seterr(**old_settings)
            return oldstd, x

        oldstd = newstd

#---------------------------------------------------------------
def check_clip(clipped_prof, toomuch=0.8, toolittle=0.):
    """
    Determines whether a clipped profile from sigmaClip() has been appropriately clipped by checking the number of nans.
    Raises a LittleClipError or a LargeClipError if too little or toomuch of the data has been clipped respectively.

    Parameters:
    -----------
    clipped_prof: list
        The clipped profile from sigmaClip()
    toomuch: float
        OPTIONAL - The fraction of the clipped profile beyond which is considered overclipped. Default: 0.8
    toolittle: float
        OPTIOANL - The fraction of the clipped profile bleow which is considered underclipped. Default: 0.

    Returns:
    --------
    None
    """
    num_nans = 0
    for i in clipped_prof:
        if np.isnan(i):
            num_nans += 1
    if num_nans == toolittle*len(clipped_prof):
        raise LittleClipError("Not enough data has been clipped. Condsier trying a smaller alpha value when clipping.")
    elif num_nans >= toomuch*len(clipped_prof):
        raise LargeClipError("A large portion of the data has been clipped. Condsier trying a larger alpha value when clipping.")
    return

#---------------------------------------------------------------
def fill_clipped_prof(clipped_prof, search_scope=None, nan_type=0.):
    """
    Intended for use on noisy profiles. Fills nan values that are surrounded by non-nans to avoid discontinuities in the profile

    Parameters:
    -----------
    clipped_prof: list
        The on-pulse profile
    profile: list
        The original profile
    search_scope: int
        The number of bins to search for non-nan values. If None, will search 5% of the total bin number. Default:None.

    Returns:
    --------
    clipped_prof: list
        The clipped profile with nan gaps filled in
    """
    length = len(clipped_prof)
    if search_scope is None:
        #Search 5% ahead for non-nans
        search_scope = round(length*0.05)
    search_scope = np.linspace(1, search_scope, search_scope, dtype=int)

    #loop over all values in clipped profile
    for i, val in enumerate(clipped_prof):
        if val==nan_type and not (i+max(search_scope)) >= length:
            #look 'search_scope' indices ahead for non-nans
            for j in sorted(search_scope, reverse=True):
                #fill in nans
                if clipped_prof[i+j]==nan_type:
                    for k in range(j):
                        clipped_prof[i+k]=nan_type
                    break

    return clipped_prof

#---------------------------------------------------------------
def find_components(profile, min_comp_len=5):
    """
    Given a profile in which the noise is clipped to 0, finds the components that are clumped together.

    Parameters:
    -----------
    profile: list
        A list of floats describing the profile where the noise has been clipped to zero
    min_comp_len: float
        OPTIONAL - Minimum length of a component to be considered real. Measured in bins. Default: 5

    Returns:
    --------
    component_dict: dictionary
        dict["component_x"] contains an array of the component x
    component_idx: dictionary
        dict["component_x"] contains an array of indexes of the original profile corresponding to component x
    """
    component_dict={}
    component_idx={}
    num_components=0
    for i, val in enumerate(profile):
        if val!=0.:
            if profile[i-1]==0 or i==0:
                num_components+=1
                comp_key = "component_{}".format(num_components)
                component_dict[comp_key]=[]
                component_idx[comp_key]=[]
            component_dict[comp_key].append(val)
            component_idx[comp_key].append(i)

    del_comps = []
    for comp_key in component_dict.keys():
        if len(component_dict[comp_key]) < min_comp_len or max(component_dict[comp_key]) < 0.:
            del_comps.append(comp_key)
    for i in del_comps:
        del component_dict[i]
        del component_idx[i]

    if len(component_dict.keys()) == 0:
        raise NoComponentsError("No profile components have been found")

    return component_dict, component_idx

#---------------------------------------------------------------
def find_minima_maxima(profile, ignore_threshold=0, min_comp_len=0):
    """
    Finds all minima and maxima of the input profile. Assumes that the profile has noise zero-clipped.

    profile: list
        The profile with noise zero-clipped
    ignore_threshold: float
        OPTIONAL -  Maxima with values below this number will be ignored. Default: 0
    min_comp_len: float
        OPTIONAL - Minimum length of a component to be considered real. Measured in bins. Default: 0

    Returns:
    --------
    minima: list
        A list of floats corresponding to the bin location of the profile minima
    maxima: list
        A list of floats corresponding to the bin location of the profile maxima
    """
    #If there is more than one component, find each one
    comp_dict, comp_idx = find_components(profile, min_comp_len)

    maxima=[]
    minima=[]
    #loop over each profile component
    for key in comp_dict.keys():
        x = np.linspace(0, len(comp_dict[key])-1, len(comp_dict[key]), dtype=int)
        spline = UnivariateSpline(x, comp_dict[key], s=0.0, k=4)
        comp_roots = spline.derivative().roots()
        # These are the roots, we want to split maxima and minima ^^
        comp_maxima=[]
        comp_minima=[]
        for i, root in enumerate(comp_roots):
            idx = int(root)
            left = comp_dict[key][idx-1]
            if left>comp_dict[key][idx]:
                comp_minima.append(root)
            else:
                comp_maxima.append(root)
        #Turn the root locations into locations on profile, not on component
        for root in comp_minima:
            abs_root = root + comp_idx[key][0]
            minima.append(abs_root)
        for root in comp_maxima:
            abs_root = root + comp_idx[key][0]
            maxima.append(abs_root)

    ignore_idx = []
    for i, mx in enumerate(maxima):
        if max(profile[int(mx-1):int(mx+1)]) < ignore_threshold*max(profile):
            ignore_idx.append(i)
    for i in sorted(ignore_idx, reverse=True):
        del maxima[i]

    return minima, maxima

def find_minima_maxima_gauss(popt, pcov, x_length):
    """
    Finds all roots of a gaussian function

    Parameters:
    -----------
    popt: list
        A list of length 3N where N is the number of gaussians. This list contains the parameters amp, mean, centre respectively
    x_length: int
        The length of the list used to fit the gaussian

    Returns:
    --------
    minima: list
        A list of the minimum points of the fit
    maxima: list
        A list of the maximum points of the fit
    """
    #Create the derivative list and spline it to find roots
    x = np.linspace(0, x_length-1, x_length)
    dy = multi_gauss_ddx(x, *popt)
    spline_dy = UnivariateSpline(x, dy, s=0)
    roots = spline_dy.roots()

    #Find which are max and min
    maxima = []
    minima = []
    for root in roots:
        idx = int(root + 0.5)
        if dy[idx-1] > dy[idx]:
            maxima.append(root)
        else:
            minima.append(root)

    minima_e = find_x_err(minima, popt, pcov)
    maxima_e = find_x_err(maxima, popt, pcov)

    return minima, maxima, minima_e, maxima_e

#---------------------------------------------------------------
def find_x_err(x, popt, pcov):
    """
    Finds the error in the horizontal position of a gaussian fit at the point x.
    Uses the equation sigma_x = sigma_y/d2ydx2 where:
    sigma_x = error in x
    d2ydx2 = second derivative of the gaussian function at point x
    sigma_y = J*C*J_T
    J = Jacobian evalutated at point x
    C = covariance matrix of gaussian fit
    J_T = transposed jacobian

    Parameters:
    -----------
    x: list
        A list of points to evaluate the error at
    popt: list
        The parameters used to describe the gaussian fit
    pcov: numpy.matrix
        The covariance matrix corresponding to popt

    Returns:
    --------
    x_err: list
        The error evaluated at each point, x
    """
    x_err = []
    for i, point in enumerate(x):
        J = jacobian_slope(point, *popt)
        d2dx2 = multi_gauss_d2dx2(point, *popt)
        JC = np.matmul(J, pcov)
        sigma_y = np.sqrt( np.matmul(JC, np.transpose(J)).item(0) )
        x_err.append(sigma_y / abs(d2dx2))
    return x_err

#---------------------------------------------------------------
def find_widths(popt, pcov, fit, std, alpha):
    """
    Attempts to find the W_10, W_50 and equivalent width of a profile by using a spline approach.
    W10 and W50 errors are estimated by using: sigma_x = sigma_y/(dy/dx)
    Weq errors are estimated by finding the average difference in Weq when you add and subtract the std from the on-pulse profile

    Parameters:
    -----------
    profile: list
        The profile to find the widths of
    std: float
        OPTIONAL - The standard deviation of the noise. If unsupplied, will return Nones for uncertainty values. Default: None

    Returns:
    --------
    [W10, W50, Weq, W10_e, W50_e, Weq_e]: list
        W10: float
            The W10 width of the profile measured in number of bins
        W50: float
            The W50 width of the profile measured in number of bins
        Weq: float
            The equivalent width of the profile measured in number of bins
        W10_e: float
            The uncertainty in W10
        W50_e:
            The uncertainty in W50
        Weq_e:
            The uncertainty in Weq
    """
    def error_in_x_pos(pcov, popt, x):
        J = jacobian_slope(x, *popt)
        JC = np.matmul(J, pcov)
        sigma_y = np.sqrt(np.matmul(JC, np.transpose(J)).item(0))
        ddx = multi_gauss_ddx(x, *popt)
        return sigma_y/ddx

    #perform spline operations
    profile = np.array(fit)
    x = np.linspace(0, len(fit)-1, len(fit))
    amp_y = max(profile) - min(profile)
    spline10 = UnivariateSpline(x, fit - np.full(len(x), 0.1*amp_y), s=0)
    spline50 = UnivariateSpline(x, fit - np.full(len(x), 0.5*amp_y), s=0)
    spline_s = UnivariateSpline(x, fit - np.full(len(x), 1/np.exp(1)*amp_y), s=0)

    #find Weq
    _, off_pulse = sigmaClip(profile, alpha=alpha)
    on_pulse=[]
    for i, data in enumerate(off_pulse):
        if np.isnan(data):
            on_pulse.append(profile[i])
    x = np.linspace(0, len(on_pulse)-1, len(on_pulse))
    spline0 = UnivariateSpline(x, on_pulse, s=0)
    integral = spline0.integral(0, len(on_pulse)-1)
    Weq = integral/max(on_pulse)

    #Weq = integral_multi_gauss(*popt)/max(fit)

    #find W10, W50 and Wscat
    W10_roots = spline10.roots()
    W50_roots = spline50.roots()
    Wscat_roots = spline_s.roots()
    W10 = W10_roots[-1] - W10_roots[0]
    W50 = W50_roots[-1] - W50_roots[0]
    Wscat = Wscat_roots[-1] - Wscat_roots[0]

    #W10 root errors
    err_10_1 = error_in_x_pos(pcov, popt, W10_roots[0])
    err_10_2 = error_in_x_pos(pcov, popt, W10_roots[-1])
    W10_e = np.sqrt(err_10_1**2 + err_10_2**2)

    #W50 root errors
    err_50_1 = error_in_x_pos(pcov, popt, W50_roots[0])
    err_50_2 = error_in_x_pos(pcov, popt, W50_roots[-1])
    W50_e = np.sqrt(err_50_1**2 + err_50_2**2)

    #Wscat root errors
    err_scat_1 = error_in_x_pos(pcov, popt, Wscat_roots[0])
    err_scat_2 = error_in_x_pos(pcov, popt, Wscat_roots[-1])
    Wscat_e = np.sqrt(err_scat_1**2 + err_scat_2**2)

    #Weq errors - using covariance formula
    on_pulse_less = (on_pulse - std).clip(min=0)
    on_pulse_more = (on_pulse + std).clip(min=0)

    spline0 = UnivariateSpline(x, on_pulse_less, s=0)
    integral = spline0.integral(0, len(profile)-1)
    dwdint = 1/max(on_pulse)**2
    dwdmax = -integral/max(on_pulse)**2
    int_e = abs(integral/max(on_pulse - std) - integral/max(on_pulse))
    max_e = std
    Weq_e = np.sqrt( dwdint**2 * int_e**2 + dwdmax**2 * max_e**2 + 2*dwdint*dwdmax*int_e*max_e )
    """
    J = jacobian_weq(list(fit).index(max(fit)), max(fit), *popt)
    print("Jacobian: {}".format(J))
    JC = np.matmul(J, pcov)
    Weq_e = np.sqrt(np.matmul(JC, np.transpose(J)).item(0))
    """

    return [W10, W50, Weq, W10_e, Wscat, Wscat_e, W50_e, Weq_e]

#---------------------------------------------------------------
def analyse_pulse_prof(prof_path=None, prof_data=None, period=None, alpha=2):
    """
    Estimates the signal to noise ratio from a pulse profile. Returns are in list form. Will return more for verbose==True setting, explained below.
    NOTE: user must supply EITHER a betprof path OR prof_data and period of the pulse profile.
    Based on code oringally writted by Nick Swainston.

    Parameters:
    -----------
    prof_path: string
        OPTIONAL - The path of the bestprof file
    prof_data: list
        OPTIONAL - A list of floats that contains the pulse profile
    period: float
        OPTIONAL - The pulsar's period in ms
    verbose: boolean
        OPTIONAL - Determines whether to return more detailed information. Detailed below

    Returns:
    --------
    sn: float
        The estimated signal to noise ratio
    u_sn: float
        The estimated signal to noise ratio's its uncertainty
    flags: list
        VERBOSE - a list of flagged data points
    w_equiv_bins: float
        VERBOSE - the equivalent width of the profile measured in bins
    u_w_equiv_bins: float
        VERBOSE - the uncertaintiy in w_equiv_bins
    w_equiv_ms: float
        VERBOSE - the equivalent width of the profile measured in ms
    u_w_equiv_ms: float
        VERBOSE - the uncertainty in w_equiv_ms
    scattered: boolean
        VERBOSE - when true, the profile is highly scattered
    """
    if prof_path is None and (prof_data is None or period is None):
        logger.warning("Insufficient information to attain SN estimate from profile. Returning Nones")
        return None, None

    if prof_data is None:
        _, _, _, period, _, _, _, prof_data, nbins = get_from_bestprof(prof_path)
        nbins = float(nbins)
        period = float(period)
    else:
        nbins = len(prof_data)

    #centre the profile around the max
    shift = -int(np.argmax(prof_data))+int(nbins)//2
    prof_data = np.roll(prof_data, shift)

    #find std and check if profile is scattered
    sigma, flags = sigmaClip(prof_data, tol=0.01, ntrials=100, alpha=alpha)
    check_clip(flags)
    bot_prof_min = (max(prof_data) - min(prof_data)) * .1 + min(prof_data)
    scattered=False
    if (np.nanmin(flags) > bot_prof_min) or ( not np.isnan(flags).any() ):
        logger.warning("The profile is highly scattered. S/N estimate cannot be calculated")
        scattered=True
        #making a new profile with the only bin being the lowest point
        prof_min_i = np.argmin(prof_data)
        flags = []
        for fi, _ in enumerate(prof_data):
            if fi == prof_min_i:
                flags.append(prof_data[fi])
            else:
                flags.append(np.nan)
        flags = np.array(flags)
        prof_data -= min(prof_data)
    else:
        prof_e = 500. #this is an approximation
        non_pulse_bins = 0
        #work out the above parameters
        for i, data in enumerate(prof_data):
            if not np.isnan(flags[i]):
                non_pulse_bins += 1
        sigma_e = sigma / np.sqrt(2 * non_pulse_bins - 2)
        #now calc S/N
        sn = max(prof_data)/sigma
        sn_e = sn * np.sqrt(prof_e/max(prof_data)**2 + (sigma_e/sigma)**2)

    if scattered==False:
        prof_dict = prof_eval_gfit(prof_data)
        W10 = prof_dict["W10"]
        W10_e = prof_dict["W10_e"]
        W50 = prof_dict["W50"]
        W50_e = prof_dict["W50_e"]
        Wscat = prof_dict["Wscat"]
        Wscat_e = prof_dict["Wscat_e"]
        Weq = prof_dict["Weq"]
        Weq_e = prof_dict["Weq_e"]
        maxima = prof_dict["maxima"]
        maxima_e = prof_dict["maxima_e"]
        chisq = prof_dict["redchisq"]
        num_gauss = prof_dict["num_gauss"]
    else:
        #Assuming width is equal to pulsar period because of the scattering
        Weq = nbins
        Weq_e = 0.5
        W10 = W10_e = W50 = W50_e = Wscat = Wscat_e = sn = sn_e = maxima = maxima_e = chisq = num_gauss = None

    prof_dict = {"sn":sn, "sn_e":sn_e, "Weq":Weq, "Weq_e":Weq_e, "W50":W50, "W50_e":W50_e, "W10_e":W10_e,\
                "Wscat":Wscat, "Wscat_e":Wscat_e, "period":period, "bins":nbins, "off_pulse":flags,\
                "scattered":scattered, "maxima":maxima, "maxima_e":maxima_e, "redchisq":chisq, "num_gauss":num_gauss}
    return prof_dict

#---------------------------------------------------------------
def integral_multi_gauss(*params):
    y=0
    for i in range(0, len(params), 3):
        a = params[i]
        b = params[i+1]
        c = params[i+2]
        y = y + a*c*np.sqrt(2*np.pi)
    return y

def multi_gauss(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        a = params[i]
        b = params[i+1]
        c = params[i+2]
        y = y +  a * np.exp( -(((x-b)**2) / (2*c**2)) )
    return y

def multi_gauss_ddx(x, *params):
    #derivative of gaussian
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        a = params[i]
        b = params[i+1]
        c = params[i+2]
        y = y - a/c**2 * (x - b) * np.exp( -(((x-b)**2) / (2*c**2)) )
    return y

def multi_gauss_d2dx2(x, *params):
    #double derivative of gaussian
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        a = params[i]
        b = params[i+1]
        c = params[i+2]
        y = y + (multi_gauss(x, a, b, c) / c**2) * (((x - b)**2)/(c**2) - 1)
    return y

def partial_gauss_dda(x, a, b, c):
        return np.exp((-(b - x)**2)/(2*c**2))
def partial_gauss_ddb(x, a, b, c):
        return a*(x - b) * np.exp((-(b - x)**2)/(2*c**2))/c**2
def partial_gauss_ddc(x, a, b, c):
        return a*(x - b)**2 * np.exp((-(b - x)**2)/(2*c**2))/c**3

def jacobian_weq(x, peak, *params):
    """
    Evaluates the Jacobian matrix of a gaussian integral divided by a peak position for
    the equivalent width calculation.
    W = equivalent width
    f = gaussian integral

    Parameters:
    -----------
    *params: list
        A list containing three parameters per gaussian component in the order: Amp, Mean, Width

    Returns:
    --------
    J: numpy.matrix
        The Jacobian matrix
    """
    def f(a, b, c):
        return a * c * np.sqrt(2 * np.pi)
    def W(peak, a, b, c):
        return f(a, b, c)/peak
    def dfda(a, b, c):
        return c * np.sqrt(2 * np.pi)
    def dfdc(a, b, c):
        return a * np.sqrt(2*np.pi)
    def dwdf(peak, a, b, c):
        return W(peak, a, b, c)/peak
    def dwdpeak(peak, a, b, c):
        return -W(peak, a, b, c)/peak
    def dwda(x, peak, a, b, c):
        return dwdf(peak, a, b, c) * dfda(a, b, c) +\
               dwdpeak(peak, a, b, c) * partial_gauss_dda(x, a, b, c)
    def dwdc(x, peak, a, b, c):
        return dwdf(peak, a, b, c) * dfdc(a, b, c) +\
               dwdpeak(peak, a, b, c) * partial_gauss_ddc(x, a, b, c)

    J = []
    for i in range(0, len(params), 3):
        a = params[i]
        b = params[i+1]
        c = params[i+2]
        mypars = [x, peak, a, b, c]
        J.append(dwda(*mypars))
        J.append(0)
        J.append(dwdc(*mypars))
    J = np.asmatrix(J)
    return J

#---------------------------------------------------------------
def jacobian_slope(x, *params):
    """
    Evaluates the Jacobian matrix of a gaussian slope at a single point, x

    Parameters:
    -----------
    x: float
        The point to evaluate
    *params: list
        A list containing three parameters per gaussian component in the order: Amp, Mean, Width

    Returns:
    --------
    J: numpy.matrix
        The Jacobian matrix
    """
    def dda(a, b, c, x):
        return -multi_gauss(x, a, b, c) * (x - b)/(c**2)/a
    def ddb(a, b, c, x):
        return multi_gauss(x, a, b, c) * (1 - (x - b)**2/(c**2))/c**2
    def ddc(a, b, c, x):
        return multi_gauss(x, a, b, c) * (x - b)/(c**3) * (2 - (x-b)**2/(c**2))
    J = []
    for i in range(0, len(params), 3):
        a = params[i]
        b = params[i+1]
        c = params[i+2]
        mypars = [a, b, c, x]
        J.append(dda(*mypars))
        J.append(ddb(*mypars))
        J.append(ddc(*mypars))
    J = np.asmatrix(J)
    return J

#---------------------------------------------------------------
def plot_fit(plot_name, y, fit, popt, maxima=None, maxima_e=None):

    x = np.linspace(0, len(y)-1, len(y))
    plt.figure(figsize=(30, 18))

    for j in range(0, len(popt), 3):
        z = multi_gauss(x, *popt[j:j+3])
        plt.plot(x, z, "--", label="Gaussian Component {}".format(int((j+3)/3)))
    if maxima:
        for i, mx in enumerate(maxima):
            plt.axvline(x=(mx + maxima_e[i]), ls=":", lw=2, color="gray")
            plt.axvline(x=(mx - maxima_e[i]), ls=":", lw=2, color="gray")

    plt.title(plot_name.split("/")[-1].split(".")[0], fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, len(y))
    plt.xlabel("Bins", fontsize=20)
    plt.ylabel("Intensity", fontsize=20)

    plt.plot(x, y, label="Original Profile", color="black")
    plt.plot(x, fit, label="Gaussian Model", color="red")
    plt.legend(loc="upper right", prop={'size': 16})
    plt.savefig(plot_name)
    plt.close()
    return

#---------------------------------------------------------------
def fit_gaussian(profile, max_N=6, min_comp_len=0, plot_name=None, alpha=2):
    """
    Fits multiple gaussian components to a pulse profile and finds the best number to use for a fit.
    Will always fit at least one gaussian per profile component.
    Profile components are defined by find_components().
    Each gaussian is defined by the following: y = amp * np.exp( -((x - ctr)/wid)**2)

    Parameters:
    -----------
    profile: list
        A list containing the profile data
    max_N: int
        OPTIONAL - The maximum number of gaussain components to attempt to fit. Default: 6
    chi_threshold: float
        OPTIONAL - The script will stop trying new fits when the reduced chi-squared is within this amount to unity. Default: 0
    plot_name: string
        OPTIONAL - If not none, will make a plot of the best fit with this name. Default: None

    Returns:
    --------
    [fit, redchisq, best_bic, popt, pcov]: list
        fit: list
            The data containing the multi-component gaussian fit to the input profile
        redchisq: float
            The reduced chi-sqaured value of the fit
        best_bic: float
            The bayesian information criterion for the fit
        popt: list
            A list of floats where each 3 numbers describes a single gaussain and are 'ctr', 'amp' and 'wid' respectively
        pcov: numpy matrix
            The covariance matrix generated by the curve_fit function
    """
    #chi sqaured evaluation
    def chsq(observed_values, expected_values, err):
        test_statistic=0
        for observed, expected in zip(observed_values, expected_values):
            test_statistic+=((float(observed)-float(expected))/float(err))**2
        return test_statistic

    #Take noise mean and normalize the profile
    _, clipped = sigmaClip(profile, alpha=alpha)

    #Check the clipped profile
    check_clip(clipped)

    y = np.array(profile) - np.nanmean(np.array(clipped))
    max_y = max(y)
    len_y = len(y)
    y = np.array(y)/max_y
    noise_std = np.nanstd(np.array(clipped)/max_y)
    plt.plot(clipped)
    plt.savefig("test_flag.png")
    plt.close()

    #Find profile components
    clipped = fill_clipped_prof(clipped, search_scope=int(len(profile)/100))
    on_pulse=[]
    for i, val in enumerate(clipped):
        if not np.isnan(val):
            on_pulse.append(0)
        else:
            on_pulse.append(y[i])
    comp_dict, comp_idx = find_components(on_pulse, min_comp_len=min_comp_len)

    #Estimate gaussian parameters based on profile components
    comp_centres = []
    comp_max = []
    comp_width = []
    for i in range(max_N//len(comp_idx.keys())+1):
        for key in comp_idx.keys():
            comp_centres.append(np.mean(comp_idx[key]))
            comp_max.append(max(comp_dict[key])*0.5)
            comp_width.append((max(comp_idx[key])-min(comp_idx[key])))
    centre_guess = iter(comp_centres)
    width_guess=iter(comp_width)
    max_guess=iter(comp_max)

    n_comps=len(comp_dict.keys())
    logger.debug("Number of profile components: {0} ({1})".format(n_comps, comp_centres[:n_comps]))

    #Fit from 1 to max_N gaussians to the profile. Evaluate profile fit using bayesian information criterion
    x=np.linspace(0, len(y)-1, len(y))
    bounds_arr=[[],[]]
    guess = []
    fit_dict = {}

    for num in range(n_comps-1, max_N):
        guess += [next(max_guess), next(centre_guess), next(width_guess)]
        bounds_arr[0].append(0)
        bounds_arr[0].append(0)
        bounds_arr[0].append(0)
        bounds_arr[1].append(max(y))
        bounds_arr[1].append(len(y))
        bounds_arr[1].append(len(y))
        bounds_tuple=(tuple(bounds_arr[0]), tuple(bounds_arr[1]))
        popt, pcov = curve_fit(multi_gauss, x, y, bounds=bounds_tuple,  p0=guess, maxfev=100000)
        fit = multi_gauss(x, *popt)
        chisq = chsq(y, fit, noise_std)
        #Bayesian information criterion for gaussian noise
        k = 3*(num+1)
        bic = chisq + k*np.log(len(y))
        fit_dict[str(num+1)]={"popt":[], "pcov":[], "fit":[], "chisq":[], "bic":[]}
        fit_dict[str(num+1)]["popt"] = popt
        fit_dict[str(num+1)]["pcov"] = pcov
        fit_dict[str(num+1)]["fit"] = fit
        fit_dict[str(num+1)]["redchisq"] = chisq/(len(y)-1)
        fit_dict[str(num+1)]["bic"] = bic
        logger.debug("Reduced chi squared for               {0} components: {1}".format(num+1, fit_dict[str(num+1)]["redchisq"]))
        logger.debug("Bayesian Information Criterion for    {0} components: {1}".format(num+1, fit_dict[str(num+1)]["bic"]))

    #Find the best fit according to the BIC
    best_bic = np.inf
    best_fit = None
    for n_components in fit_dict.keys():
        if fit_dict[n_components]["bic"] < best_bic:
            best_bic = fit_dict[n_components]["bic"]
            best_fit = n_components
    logger.info("Fit {0} gaussians for a reduced chi sqaured of {1}".format(best_fit, fit_dict[best_fit]["redchisq"]))
    popt = fit_dict[best_fit]["popt"]
    pcov = fit_dict[best_fit]["pcov"]
    fit = fit_dict[best_fit]["fit"]
    redchisq = fit_dict[best_fit]["redchisq"]

    return [fit, redchisq, best_bic, popt, pcov]

#---------------------------------------------------------------
def prof_eval_gfit(profile, max_N=6, ignore_threshold=None, plot_name=None, min_comp_len=None, alpha=2):
    """
    Fits multiple gaussians to a profile and subsequently finds W10, W50, Weq and maxima

    Parameters:
    -----------
    profile: list
        The pulse profile to evaluate
    plot_name: string
        OPTIONAL - If not none, will make a plot of the best fit with this name. Default: None
    ignore_threshold: float
        OPTIONAL -  Maxima with values below this number will be ignored. If none, will use 3*noise. Default: None
    min_comp_len: float
        OPTIONAL - Minimum length of a component to be considered real. Measured in bins. If none, will use 1% of total profile lengths + 2. Default: None

    Returns:
    --------
    fit_dict: dictionary
        contains the following keys:
        W10: float
            The W10 width of the profile measured in number of bins
        W10_e: float
            The uncertainty in the W10
        W50: float
            The W50 width of the profile measured in number of bins
        W50_e: float
            The uncertainty in the W50
        Weq: float
            The equivalent width of the profile measured in number of bins
        Weq_e: float
            The uncertainty in the equivalent width
        Wscat: float
            The scattering width of the profile measured in number of bins
        Wscat_e: float
            The uncertainty in the scattering width
        maxima: list
            A lost of floats corresponding to the bin location of each maximum point
        maxima_e: list
            A list of floats, each correspinding to the error of the maxima of the same index. Measured in bins
        redchisq: float
            The reduced chi sqared of the fit
        num_gauss: int
            The number of gaussian components used in the best fit
        bic: float
            The Bayesian Information Criterion for the best fit
        gaussian_params: list
            A list of length 3*N there N is num_gauss. Each set of 3 parameters corresponds to the amp, centre and width of a guassian component
        cov_mat: np.matrix
            The covariance matrix from the fit
        alpha: float
            The alpha value used in sigmaClip()
        profile: list
            The input profile
        fit: list
            The best fit made into a list form
    """
    #initialize minimum component length and ignore threshold
    if min_comp_len is None:
        min_comp_len = int(len(profile)/100 + 0.5) + 2
    if min_comp_len < 5:
        min_comp_len = 5

    #Normalize, find the std
    y = np.array(profile)/max(profile)
    noise_std, clipped = sigmaClip(y, alpha=alpha)
    check_clip(clipped, toomuch=0.8, toolittle=0.)

    if ignore_threshold is None:
        ignore_threshold = 3 * noise_std

    y = y - np.nanmean(clipped)
    y = y/max(y)

    #fit gaussians
    fit, chisq, bic, popt, pcov = fit_gaussian(y, max_N=max_N, min_comp_len=min_comp_len, alpha=alpha)
    fit = np.array(fit)
    n_rows, _ = np.shape(pcov)
    num_gauss = n_rows/3

    #Find widths + error
    W10, W50, Weq, W10_e, Wscat, Wscat_e, W50_e, Weq_e = find_widths(popt, pcov, fit, noise_std, alpha)

    #find max, min, error
    for i, val in enumerate(clipped):
        if np.isnan(val):
            clipped[i]=0.
    clipped = fill_clipped_prof(clipped, search_scope=int(len(profile)/100))
    on_pulse=[]
    for i, val in enumerate(clipped):
        if val!=0:
            on_pulse.append(0)
        else:
            on_pulse.append(fit[i])

    _, maxima, _, maxima_e = find_minima_maxima_gauss(popt, pcov, len(fit))

    logger.info("W10:                   {0} +/- {1}".format(W10, W10_e))
    logger.info("W50:                   {0} +/- {1}".format(W50, W50_e))
    logger.info("Wscat:                 {0} +/- {1}".format(Wscat, Wscat_e))
    logger.info("Weq:                   {0} +/- {1}".format(Weq, Weq_e))
    logger.info("Maxima:                {0}".format(maxima))
    logger.info("Maxima error:          {0}".format(maxima_e))

    #plotting
    if plot_name:
        plot_fit(plot_name, y, fit, popt, maxima=maxima, maxima_e=maxima_e)

    fit_dict = {"W10":W10, "W10_e":W10_e, "W50":W50, "W50_e":W50_e, "Wscat":Wscat, "Wscat_e":Wscat_e,\
                "Weq":Weq, "Weq_e":Weq_e, "maxima":maxima, "maxima_e":maxima_e, "redchisq":chisq,\
                "num_gauss":num_gauss, "bic":bic, "gaussian_params":popt, "cov_mat":pcov, "alpha":alpha,\
                "profile":y, "fit":fit}

    return fit_dict

def auto_gfit(profile, max_N=6, plot_name=None, ignore_threshold=None, min_comp_len=None):
    """
    runs the gaussian fit evaluation for a range of values of alpha. This is necessary as there is no way to know
    a priori which alpha to use beforehand. Alpha is the input for sigmaClip() and can be interpreted as the level
    of verbosity in clipping.

    Parameters:
    -----------
    profile: list
        A list containing the pulse profile to evaluate
    max_N: int
        OPTIONAL - The maximum number of gaussian components to use when fitting
    plot_name: string
        OPTIONAL - If not none, will make a plot of the best fit with this name. Default: None
    ignore_threshold: float
        OPTIONAL -  Maxima with values below this number will be ignored. If none, will use 3*noise. Default: None
    min_comp_len: float
        OPTIONAL - Minimum length of a component to be considered real. Measured in bins. If none, will use 1% of total profile lengths + 2. Default: None

    Returns:
    --------
    fit_dict: dictionary
        contains the following keys:
        W10: float
            The W10 width of the profile measured in number of bins
        W10_e: float
            The uncertainty in the W10
        W50: float
            The W50 width of the profile measured in number of bins
        W50_e: float
            The uncertainty in the W50
        Weq: float
            The equivalent width of the profile measured in number of bins
        Weq_e: float
            The uncertainty in the equivalent width
        Wscat: float
            The scattering width of the profile measured in number of bins
        Wscat_e: float
            The uncertainty in the scattering width
        maxima: list
            A lost of floats corresponding to the bin location of each maximum point
        maxima_e: list
            A list of floats, each correspinding to the error of the maxima of the same index. Measured in bins
        redchisq: float
            The reduced chi sqared of the fit
        num_gauss: int
            The number of gaussian components used in the best fit
        bic: float
            The Bayesian Information Criterion for the best fit
        gaussian_params: list
            A list of length 3*N there N is num_gauss. Each set of 3 parameters corresponds to the amp, centre and width of a guassian component
        cov_mat: np.matrix
            The covariance matrix from the fit
        alpha: float
            The alpha value used in sigmaClip()
        profile: list
            The input profile
        fit: list
            The best fit made into a list form
    """
    alphas = np.linspace(1, 5, 9)
    attempts_dict = {}

    loglvl = logger.level
    logger.setLevel(logging.WARNING) #squelch logging for the loop

    #loop over the gaussian evaluation fucntion, excepting in-built errors
    for alpha in alphas:
        try:
            prof_dict = prof_eval_gfit(profile, max_N=6, ignore_threshold=ignore_threshold, min_comp_len=min_comp_len, alpha=alpha)
            attempts_dict[alpha] = prof_dict
        except(LittleClipError, LargeClipError, NoComponentsError) as e:
            pass
    logger.setLevel(loglvl)

    #Evaluate the best profile based on reduced chi-squared.
    chi_diff = []
    alphas = []
    for alpha_key in attempts_dict.keys():
        chi_diff.append(abs(1 - attempts_dict[alpha_key]["redchisq"]))
        alphas.append(alpha_key)
    best_chi = min(chi_diff)
    best_alpha = alphas[chi_diff.index(best_chi)]
    fit_dict = attempts_dict[best_alpha]

    logger.info("### Best fit results ###")
    logger.info("Best model found with BIC of {0} and reduced Chi of {1} using an alpha value of {2}"\
                .format(fit_dict["bic"], fit_dict["redchisq"], best_alpha))
    logger.info("W10:                   {0} +/- {1}".format(fit_dict["W10"], fit_dict["W10_e"]))
    logger.info("W50:                   {0} +/- {1}".format(fit_dict["W50"], fit_dict["W50_e"]))
    logger.info("Wscat:                 {0} +/- {1}".format(fit_dict["Wscat"], fit_dict["Wscat_e"]))
    logger.info("Weq:                   {0} +/- {1}".format(fit_dict["Weq"], fit_dict["Weq_e"]))
    logger.info("Maxima:                {0}".format(fit_dict["maxima"]))
    logger.info("Maxima error:          {0}".format(fit_dict["maxima_e"]))

    if plot_name:
        plot_fit(plot_name, fit_dict["profile"], fit_dict["fit"], fit_dict["gaussian_params"], maxima=fit_dict["maxima_e"], maxima_e=fit_dict["maxima"])

    return fit_dict

#---------------------------------------------------------------
if __name__ == '__main__':

    loglevels = dict(DEBUG=logging.DEBUG,\
                    INFO=logging.INFO,\
                    WARNING=logging.WARNING,\
                    ERROR=logging.ERROR)

    parser = argparse.ArgumentParser(description="""A utility file for calculating a variety of pulse profile properties""",\
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    inputs = parser.add_argument_group("Inputs")
    inputs.add_argument("--bestprof", type=str, help="The pathname of the file containing the pulse profile. Use if in .pfd.bestprof format")
    inputs.add_argument("--ascii", type=str, help="The pathname of the file containing the pulse profile. Use if in ascii text format")
    inputs.add_argument("--ignore_threshold", type=float, default=0.02, help="Maxima with values below this fraction of the profile maximum will be ignored.")
    inputs.add_argument("--min_comp_len", type=int, default=None,\
                        help="Minimum length of a component to be considered real. Measured in bins. If none, will use 1 percent of total profile length")
    inputs.add_argument("--alpha", type=float, default=2, help="Used by the clipping function to determine the noise level. A lower value indicates\
                        a higher verbosity level in the noise clipping function.")
    inputs.add_argument("--auto", action="store_true", help="Used to automatically find the best alpha value to clip this profile")

    g_inputs = parser.add_argument_group("Gaussian Inputs")
    g_inputs.add_argument("--max_N", type=int, default=6, help="The maximum number of gaussian components to attempt to fit")

    other_inputs = parser.add_argument_group("Other Inputs")
    other_inputs.add_argument("--plot_name", type=str, help="The name of the output plot file. If none, will not plot anything")
    other_inputs.add_argument("-L", "--loglvl", type=str, default="INFO", help="Logger verbostity level")
    args = parser.parse_args()

    logger.setLevel(loglevels[args.loglvl])
    ch = logging.StreamHandler()
    ch.setLevel(loglevels[args.loglvl])
    formatter = logging.Formatter('%(asctime)s  %(filename)s  %(name)s  %(lineno)-4d  %(levelname)-9s :: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.bestprof:
        profile = get_from_bestprof(args.bestprof)[-2]
    elif args.ascii:
        profile = get_from_ascii(args.ascii)[0]
    else:
        logger.error("Please supply either an ascii or bestprof profile")
        sys.exit(1)

    if args.auto:
        auto_gfit(profile, max_N=args.max_N, ignore_threshold=args.ignore_threshold,\
                        plot_name=args.plot_name, min_comp_len=args.min_comp_len)
    else :
        prof_eval_gfit(profile, max_N=args.max_N, ignore_threshold=args.ignore_threshold,\
                        plot_name=args.plot_name, min_comp_len=args.min_comp_len, alpha=args.alpha)