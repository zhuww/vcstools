#! /usr/bin/env python3

#other
import logging
import argparse
import os
import sys
import numpy as np
import psrqpy

#matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Astropy
from astropy.table import Table

#scipy
from scipy.optimize import curve_fit

#vcstools and mwa_search
from mwa_pb import primarybeammap_tant as pbtant
import find_pulsar_in_obs as fpio
import mwa_metadb_utils
import submit_to_database
import process_vcs

logger = logging.getLogger(__name__)

try:
    ATNF_LOC = os.environ['PSRCAT_FILE']
except KeyError:
    logger.warning("ATNF database could not be found on disk.")
    ATNF_LOC = None

#---------------------------------------------------------------
def plot_flux_estimation(nu_atnf, S_atnf, S_atnf_e, my_nu, my_S, my_S_e, a, K, covar_mat, pulsar, obsid):
    """
    Parameters:
    """
    nu_range = list(nu_atnf)
    nu_range.append(my_nu)
    S_range = list(S_atnf)
    S_range.append(my_S)

    nu_cont = np.logspace(np.log10(min(nu_range)), np.log10(max(nu_range)), num=500)
    S_cont, S_cont_e = flux_from_plaw(nu_cont, K, a, covar_mat)

    nu_range=[]
    S_range=[]
    for element in nu_cont:
        nu_range.append(element)
    for element in S_cont:
        S_range.append(element)

    plt.fill_between(nu_cont, S_cont - S_cont_e, S_cont + S_cont_e, facecolor='gray') # Model errors
    plt.plot(nu_cont, S_cont, 'k--', label="model") # Modelled line
    plt.errorbar(nu_atnf, S_atnf, yerr=S_atnf_e, fmt='o', label="ATNF data points") # Original data points
    plt.errorbar(my_nu, my_S, yerr=my_S_e, fmt='o', label="Extrapolated data points") # Extrapolated data point
    plt.axis([0.5*min(nu_range), 2.0*max(nu_range), 0.5*min(S_range), 2.0*max(S_range)])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Flux (Jy)')
    plt.legend()
    plt.savefig("flux_density_{0}_{1}.png".format(pulsar, obsid))

#---------------------------------------------------------------
def analyse_pulse_prof(prof_path=None, prof_data=None, period=None, verbose=False):
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
        _, _, _, period, _, _, _, prof_data, nbins = submit_to_database.get_from_bestprof(prof_path)
        nbins = float(nbins)
        period = float(period)
    else:
        nbins = len(prof_data)

    #centre the profile around the max
    shift = -int(np.argmax(prof_data))+int(nbins)//2
    prof_data = np.roll(prof_data, shift)

    #find sigma and check if profile is scattered
    sigma, flags = submit_to_database.sigmaClip(prof_data, alpha=3., tol=0.01, ntrials=100)
    bot_prof_min = (max(prof_data) - min(prof_data)) * .1 + min(prof_data)
    scattered=False
    if (np.nanmin(flags) > bot_prof_min) or ( not np.isnan(flags).any() ):
        logger.warning("The profile is highly scattered. S/N estimate cannot be calculated")
        if verbose == True:
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
            #Assuming width is equal to pulsar period because of the scattering
            w_equiv_ms = period
            u_w_equiv_ms = period/nbins
        sn = None
        u_sn = None
    else:
        u_prof = 500. #this is an approximation
        pulse_width_bins = 0
        non_pulse_bins = 0
        p_total = 0.
        u_p = 0.
        #work out the above parameters
        for i, data in enumerate(prof_data):
            if np.isnan(flags[i]):
                pulse_width_bins += 1
                p_total += data
                u_p = np.sqrt(u_p**2 + u_prof**2)
            else:
                non_pulse_bins += 1
        u_simga = sigma / np.sqrt(2 * non_pulse_bins - 2)

        #now calc S/N
        sn = max(prof_data)/sigma
        u_sn = sn * np.sqrt(u_prof/max(prof_data)**2 + (u_simga/sigma)**2)

    if verbose==False:
        return [sn, u_sn]

    elif scattered==False:
        off_pulse_mean = np.nanmean(flags)
        prof_data -= off_pulse_mean
        flags -= off_pulse_mean

        prof_max = max(prof_data)
        w_equiv_bins = p_total / prof_max
        w_equiv_ms = w_equiv_bins / nbins * period # in ms
        u_w_equiv_bins = np.sqrt(p_total /prof_max)**2 +\
                                   (p_total * u_prof / (prof_max)**2)**2
        u_w_equiv_ms = u_w_equiv_bins / nbins * period # in ms

    else:
        w_equiv_ms = period
        u_w_equiv_ms = period/nbins
        w_equiv_bins = w_equiv_ms/period*nbins
        u_w_equiv_bins = (u_w_equiv_ms/w_equiv_ms)*w_equiv_bins

    return [sn, u_sn, flags, w_equiv_bins, u_w_equiv_bins, w_equiv_ms, u_w_equiv_ms, scattered]

#---------------------------------------------------------------
def pulsar_beam_coverage(obsid, pulsar, beg=None, end=None, ondisk=False):
    """
    Finds the normalised time that a pulsar is in the beam for a given obsid
    If pulsar is not in beam, returns None, None

    Parameters:
    -----------
    obsid: int
        The observation ID
    pulsar: string
        The pulsar's J name
    beg: int
        OPTIONAL - The beginning of the observing time in gps time
    end: int
        OPTIONAL - The end of the observing time in gps time
    ondisk: boolean
        Whether to use files that are on-disk for beginning and end times. Default=False

    Returns:
    --------
    enter_files: float
        A float between 0 and 1 that describes the normalised time that the pulsar enters the beam
    exit_files: float
         a float between 0 and 1 that describes the normalised time that the pulsar exits the beam
    """
    #Find the beginning and end of obs
    obs_beg, obs_end = files_beg, files_end = mwa_metadb_utils.obs_max_min(obsid)
    obs_dur = obs_end-obs_beg + 1

    #Logic loop:
    if ondisk==True:
        #find the beginning and end time of the observation FILES you have on disk
        files_beg, files_end = process_vcs.find_combined_beg_end(obsid)
        files_duration = files_end - files_beg + 1
    elif beg is None and end is None:
        logger.warning("ondisk==False so beg and end can not be None. Returning Nones")
        return None, None
    else:
        #uses manually input beginning and end times to find beam coverage
        files_beg = beg
        files_end = end
        files_duration = files_end - files_beg + 1

    #find the enter and exit times of pulsar normalized with the observing time
    names_ra_dec = fpio.grab_source_alog(pulsar_list=[pulsar])
    beam_source_data, _ = fpio.find_sources_in_obs([obsid], names_ra_dec)
    enter_obs_norm = beam_source_data[obsid][0][1]
    exit_obs_norm = beam_source_data[obsid][0][2]

    #times the source enters and exits beam
    time_enter = obs_beg + obs_dur*enter_obs_norm
    time_exit = obs_beg + obs_dur*exit_obs_norm

    #normalised time the source enters/exits the beam in the files
    enter_files = (time_enter-files_beg)/files_duration
    exit_files = (time_exit-files_beg)/files_duration

    if enter_files<0.:
        enter_files=0.
    if exit_files>1.:
        exit_files=1.
    if enter_files>1. or exit_files<0.:
        logger.warning("source {0} is not in the beam for the files on disk".format(pulsar))
        enter_files = None
        exit_files = None

    return enter_files, exit_files

#---------------------------------------------------------------
def least_squares_fit_plaw(x_data, y_data, y_err):
    """
    Used primarily by est_pulsar_flux() to to attain a power law function. Intended for use with pulsar flux densities

    Parameters:
    -----------
    x_data: list
        A list of frequencies in Hz
    y_data: list
        A list of fluxes in Jy correspodning to the input x_data frequencies
    y_err: list
        A list containing the corresponding error values for y_data in Hz

    Returns:
    --------
    a: float
        The fit spectral index
    c: float
        The fit y-intercept
    covar_mat: np.matrix
        The covariance matrix of the fit. Contains the information required to for uncertainty calculations

    """
    #convert everything to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    y_err = np.array(y_err)

    #Set up matrices in log space
    Y = np.log(y_data)
    X = np.vstack((np.ones(len(x_data)), np.log(x_data))).T

    #Set up errors. We will use the avg. length of the errors in logspace
    Y_err = 0.5 * np.log((y_data + y_err)/(y_data - y_err))

    #Convert the errors to weights
    W = np.diag(1/Y_err**2)

    #for reference: https://en.wikipedia.org/wiki/Weighted_least_squares
    # B =(X'*W*X)' * X'WY
    XW     = np.matmul(X.T, W)
    XWX    = np.matmul(XW, X)
    XWY    = np.matmul(XW, Y)
    XWXinv = np.linalg.pinv(XWX)
    b      = np.matmul(XWXinv, XWY)

    c = b[0]
    a = b[1]
    #convert c from log tospace to linear space
    K = np.exp(c)

    #The covariance matrix
    covar_mat = XWXinv

    logger.debug("a: {0}".format(a))
    logger.debug("K: {0}".format(K))
    logger.debug("Covariance Matrix: {0}".format(covar_mat))

    return a, K, covar_mat

def flux_from_plaw(freq, K, a, covar_mat):
    """
    Calculates the flux and error from a power law fit by extrapolating to the desired frequency.
    The power law is of the form S = c * nu**a

    Parameters:
    -----------
    freq: float
        The frequency for which we want to calculate a flux for (nu)
    K: float
        The value of K from the power law function
    a: float
        The value of a (spectral index) from the power law function
    covar_matrix: numpy matrix
        The covariance matrix from our power law fit. The main diagonal elements are sigma_c^2, sigma_a^2 respectively

    Returns:
    --------
    flux: float
        The calculated flux
    flux_err: float
        The uncertainty of the calculated flux
    """

    def plaw_func(nu, K, a):
        #Power law function
        return K*nu**a

    flux = plaw_func(freq, K, a)

    #Calculate the error. For reference: https://en.wikipedia.org/wiki/Propagation_of_uncertainty
    log_freq = np.log(freq)

    dc2 = covar_mat[0, 0]
    da2 = covar_mat[1, 1]
    dac = covar_mat[0, 1]

    flux_err_log = np.sqrt(dc2 + da2*log_freq**2 + 2*dac*log_freq)

    #convert the error to linear space. We will use the 'average' logspace error
    z = np.exp(2*flux_err_log)
    flux_err = flux*(z-1)/(z+1)

    return flux, flux_err

#---------------------------------------------------------------
def est_pulsar_flux(pulsar, obsid, plot_flux=False, metadata=None, query=None):
    """
    Estimates a pulsar's flux from archival data by assuming a power law relation between flux and frequency. Frist tries to attain a apectral index from the ATNF database. If this fails, try to work out a spectrla index. If this fails, uses an index of -1.4 with uncertainty of 1.

    Parameters:
    -----------
    pulsar: string
        The puslar's name. e.g. 'J2241-5236'
    obsid: int
        The observation ID
    plot_flux: boolean
        OPTIONAL - Whether or not to make a plot of the flux estimation. Default = False
    metadata: list
        OPTIONAL - The metadata call for this obsid
    query: object
        OPTIONAL - The return from psrqpy.QueryATNF for this pulsar

    Returns:
    -------
    flux: float
        The estimated flux in Jy
    flux_err: float
        The estimated flux's uncertainty in Jy
    """

    if metadata is None:
        logger.debug("obtaining mean freq from obs metadata")
        metadata = mwa_metadb_utils.get_common_obs_metadata(obsid)
    f_mean = metadata[5]*1e6

    if query is None:
        query = psrqpy.QueryATNF(psrs=[pulsar], loadfromdb=ATNF_LOC).pandas

    flux_queries = ["S40", "S50", "S60", "S80", "S100", "S150", "S200",\
                    "S300", "S400", "S600", "S700", "S800", "S900",\
                    "S1400", "S1600", "S2000", "S3000", "S4000", "S5000",\
                    "S6000", "S8000"]
    freq_all=[]
    flux_all=[]
    flux_err_all=[]
    #Get all available data from dataframe and check for missing values
    for flux_query in flux_queries:
        flux = query[flux_query][0]
        if not np.isnan(flux):
            #sometimes error values don't exist, causing a key error in pandas
            try:
                flux_err = query[flux_query+"_ERR"][0]
                if flux_err == 0.0:
                    logger.warning("{0} flux error for query: {1}, is zero. Assuming 20% uncertainty"\
                            .format(pulsar, flux_query))
                    flux_err = flux*0.2
            except KeyError:
                logger.warning("{0} flux error value {1}, not available. assuming 20% uncertainty"\
                            .format(pulsar, flux_query))
                flux_err = flux*0.2

            if np.isnan(flux_err):
                logger.warning("{0} flux error value for {1} not available. assuming 20% uncertainty"\
                            .format(pulsar, flux_query))
                flux_err = flux*0.2

            freq_all.append(int(flux_query.split()[0][1:])*1e6) #convert to Hz
            flux_all.append(flux*1e-3) #convert to Jy
            flux_err_all.append(flux_err*1e-3) #convert to Jy

    #Also get spectral index if it exists
    spind = query["SPINDX"][0]
    spind_err = query["SPINDX_ERR"][0]

    logger.debug("Freqs: {0}".format(freq_all))
    logger.debug("Fluxes: {0}".format(flux_all))
    logger.debug("Flux Errors: {0}".format(flux_err_all))
    logger.info("{0} there are {1} flux values available on the ATNF database"\
                .format(pulsar, len(flux_all)))
    #Attempt to estimate flux
    if len(flux_all) > 1:
        logger.info("{0} calculating power law from archive data".format(pulsar))
        for i, _ in enumerate(flux_all):
            flux_all[i] = flux_all[i]
            flux_err_all[i] = flux_err_all[i]

        #Find params from least squares fit
        spind, K, covar_mat = least_squares_fit_plaw(freq_all, flux_all, flux_err_all)
        logger.info("{0} derived spectral index: {1} +/- {2}".format(pulsar, spind, covar_mat.item(3)))

        #Get flux estimation
        flux_est, flux_est_err = flux_from_plaw(f_mean, K, spind, covar_mat)
        #Plot estimation if in debug mode
        if plot_flux==True:
            plot_flux_estimation(freq_all, flux_all, flux_err_all, f_mean, flux_est, flux_est_err, spind, K,\
                                covar_mat, pulsar, obsid)

    #Do something different if there is only one flux value in archive
    elif len(flux_all) == 1:
        logger.warning("Only a single flux value available on the archive")

        if not np.isnan(spind) and np.isnan(spind_err):
            logger.warning("{} spectral index error not available. Assuming 20% error".format(pulsar))
            spind_err = spind*0.2
        if np.isnan(spind):
            logger.warning("{} insufficient archival data to estimate spectral index. Using alpha=-1.4 +/- 1.0 as per Bates2013".format(pulsar))
            spind = -1.4
            spind_err = 1.

        #formula for flux:
        #S_1 = nu_1^a * nu_2 ^-a * S_2
        nu_1 = f_mean #in Hz
        nu_2 = freq_all[0] #in Hz
        s_2 = flux_all[0] #in Jy
        s_2_err = flux_err_all[0]
        a = spind
        a_err = spind_err

        logger.debug("nu1 {0}".format(nu_1))
        logger.debug("nu2 {0}".format(nu_2))
        logger.debug("s2 {0}".format(s_2))
        logger.info("{0} calculating flux using spectral index: {1} and error: {2}"\
                    .format(pulsar, spind, spind_err))

        flux_est = nu_1**a * nu_2**(-a) * s_2
        #variance formula error est
        s_2_var = nu_1**a * nu_2**(-a)
        a_var = s_2 * nu_1**a * nu_2**(-a) * (np.log(nu_1)-np.log(nu_2))
        a_var = a_var**2 * a_err**2
        s_2_var = s_2_var**2 * s_2_err**2
        flux_est_err = np.sqrt(s_2_var + a_var)

    elif len(flux_all) < 1:
        logger.warning("{} no flux values on archive. Cannot estimate flux. Will return Nones".format(pulsar))
        return None, None


    logger.info("{0} flux estimate at {1} MHz: {2} +/- {3} Jy"\
                .format(pulsar, f_mean/1e6, flux_est, flux_est_err))

    return flux_est, flux_est_err

#---------------------------------------------------------------
def find_pulsar_w50(pulsar, query=None):
    """
    Attempts to find a pulsar's W50 from the ATNF catalogue. If unavailable, will estimate

    Paramters:
    ---------
    pulsar: string
        The J-name of the pulsar. e.g. 'J2241-5236'
    query: object
        OPTIONAL - The query for 'pulsar' returned from psrqpy.QueryATNF
    Returns:
    --------
    w50: float
        The pulsar's W50
    w50_err: float
        The W50's uncertainty
    """
    if query is None:
        #returns W_50 and error for a pulsar from the ATNF archive IN SECONDS
        logger.debug("Accessing ATNF database")
        query = psrqpy.QueryATNF(psrs=[pulsar], loadfromdb=ATNF_LOC).pandas

    W_50 = query["W50"][0]
    W_50_err = query["W50_ERR"][0]
    if np.isnan(W_50):
        W_50=np.nan
        W_50_err=np.nan
    else:
        #convert to seconds
        W_50 = W_50/1000.

    if np.isnan(W_50_err) and not np.isnan(W_50):
        logger.warning("{} W_50 error not on archive. returning standard 5% error".format(pulsar))
        W_50_err = W_50*0.05
    else:
        #convert to seconds
        W_50_err = W_50_err/1000.

    if np.isnan(W_50):
        logger.warning("{} applying estimated W_50. Uncertainty will be inflated".format(pulsar))
        #Rankin1993 - W = x*P^0.5 where x=4.8+/-0.5 degrees of rotation at 1GHz
        #We will nflate this error due to differing frequencies and pulsar behaviour. W_50_err=1. degrees
        coeff = 4.8
        coeff_err = 2.
        period = float(query["P0"][0])

        #This estimation is worse for msps, add extra uncetainty if period < 50ms
        if period <0.05:
            coeff_err = 4.

        #calculate
        W_50 = coeff*period**0.5
        W_50_err = coeff_err*period**0.5 #error from variance calculation

        #W_50 is now in degrees of rotation phase. Convert to seconds
        W_50 = (W_50/360.)*period
        W_50_err = (W_50_err/360.)*period

    if np.isnan(W_50):
        W_50=None
    if np.isnan(W_50_err):
        W_50_err=None
    return W_50, W_50_err

#---------------------------------------------------------------
def find_times(obsid, pulsar, beg=None, end=None):
    """
    Find the total integration time of a pulsar in the primary beam of an obsid

    Parameters:
    ----------
    obsid: int
        The observation ID
    pulsar: string
        The J name of the pulsar
    beg: int
        OPTIONAL - The beginning of the observing time
    end: int
        OPTINAL - The end of the observing time

    Returns:
    -------
    beg: int
        The time when the pulsar enters the beam in gps
    end: int
        The time when the pulsar exits thebeam in gps
    t_int: int
        The total time that the pulsar is in the beam
    """
    t_int=None
    if beg is None or end is None:
        logger.info("Using duration for entire observation")
        beg, end = mwa_metadb_utils.obs_max_min(obsid)
        t_int = end - beg + 1
        enter_norm, exit_norm = pulsar_beam_coverage(obsid, pulsar, beg=beg, end=end)
        beg = beg + enter_norm * t_int
        end = beg + exit_norm * t_int

    if t_int is None:
        enter_norm, exit_norm = pulsar_beam_coverage(obsid, pulsar, beg=beg, end=end)
        if beg is not None and end is not None:
            if beg<obsid or end<obsid or beg>(obsid+10000) or end>(obsid+10000):
                logger.warning("Beginning/end times supplied are outside the obsid")
                logger.warning("Have you entered the correct times and obsid?")
            dur = end-beg
        else: #use entire obs duration
            beg, end = mwa_metadb_utils.obs_max_min(obsid)
            dur = end - beg + 1
        if enter_norm is None or exit_norm is None or dur is None:
            t_int=0
        else:
            t_int = dur*(exit_norm-enter_norm)

    return beg, end, t_int

#---------------------------------------------------------------
def find_t_sys_gain(pulsar, obsid, beg=None, end=None, t_int=None, p_ra=None, p_dec=None,\
                    obs_metadata=None, query=None, trcvr="/group/mwaops/PULSAR/MWA_Trcvr_tile_56.csv"):

    """
    Finds the system temperature and gain for an observation.
    A function snippet originally written by Nick Swainston - adapted for general VCS use.

    Parameters:
    -----------
    pulsar: str
        the J name of the pulsar. e.g. J2241-5236
    obsid: int
        The observation ID. e.g. 1226406800
    beg: int
        The beginning of the observing time
    t_int: float
        The total time that the target is in the beam
    p_ra: str
        OPTIONAL - the target's right ascension
    p_dec: str
        OPTIONAL - the target's declination
    obs_metadata: list
        OPTIONAL - the array generated from mwa_metadb_utils.get_common_obs_metadata(obsid)
    query: object
        OPTIONAL - The return of the psrqpy function for this pulsar
    trcvr: str
        The location of the MWA receiver temp csv file. Default = '/group/mwaops/PULSAR/MWA_Trcvr_tile_56.csv'

    Returns:
    --------
    t_sys: float
        The system temperature
    t_sys_err: float
        The system temperature's uncertainty
    gain: float
        The system gain
    gain_err: float
        The gain's uncertainty
    """

    #get ra and dec if not supplied
    if p_ra is None or p_dec is None and query is None:
        logger.debug("Obtaining pulsar RA and Dec from ATNF")
        query = psrqpy.QueryATNF(psrs=[pulsar], loadfromdb=ATNF_LOC).pandas
        p_ra = query["RAJ"][0]
        p_dec = query["DECJ"][0]
    elif p_ra is None and p_dec is None and query is not None:
        p_ra = query["RAJ"][0]
        p_dec= query["DECJ"][0]

    #get metadata if not supplied
    if obs_metadata is None:
        logger.debug("Obtaining obs metadata")
        obs_metadata = mwa_metadb_utils.get_common_obs_metadata(obsid)

    obsid, obs_ra, obs_dec, _, delays, centrefreq, channels = obs_metadata

    #get beg if not supplied
    if beg is None or t_int is None:
        logger.debug("Calculating beginning time for pulsar coverage")
        beg, _, t_int = find_times(obsid, pulsar, beg=beg, end=end)

    #Find 'start_time' for fpio - it's usually about 7 seconds
    #obs_start, _ = mwa_metadb_utils.obs_max_min(obsid)
    start_time = beg-int(obsid)

    #Get important info
    trec_table = Table.read(trcvr,format="csv")
    ntiles = 128 #TODO actually we excluded some tiles during beamforming, so we'll need to account for that here

    beam_power = fpio.get_beam_power_over_time([obsid, obs_ra, obs_dec, t_int, delays,\
                                                centrefreq, channels],\
                                                np.array([[pulsar, p_ra, p_dec]]),\
                                                dt=100, start_time=start_time)
    beam_power = np.mean(beam_power)

    # Usa a primary beam function to convolve the sky temperature with the primary beam
    # (prints suppressed)
    sys.stdout = open(os.devnull, 'w')
    _, _, Tsky_XX, _, _, _, Tsky_YY, _ = pbtant.make_primarybeammap(int(obsid), delays, centrefreq*1e6, 'analytic', plottype='None')
    sys.stdout = sys.__stdout__


    #TODO can be inaccurate for coherent but is too difficult to simulate
    t_sky = (Tsky_XX + Tsky_YY) / 2.
    # Get T_sys by adding Trec and Tsky (other temperatures are assumed to be negligible
    t_sys_table = t_sky + submit_to_database.get_Trec(trec_table, centrefreq)
    t_sys = np.mean(t_sys_table)
    t_sys_err = t_sys*0.02 #TODO: figure out what t_sys error is

    logger.debug("pul_ra: {} pul_dec: {}".format(p_ra, p_dec))
    _, _, zas = mwa_metadb_utils.mwa_alt_az_za(obsid, ra=p_ra, dec=p_dec)
    theta = np.radians(zas)
    gain = submit_to_database.from_power_to_gain(beam_power, centrefreq*1e6, ntiles, coh=True)
    logger.debug("beam_power: {} theta: {} pi: {}".format(beam_power, theta, np.pi))
    gain_err = gain * ((1. - beam_power)*0.12 + 2.*(theta/(0.5*np.pi))**2. + 0.1)

    # Removed the below error catch because couldn't find an obs that breaks it
    #sometimes gain_err is a numpy array and sometimes it isnt so i have to to this...
    #try:
    #    gain_err.shape
    #    gain_err = gain_err[0]
    #except:
    #    pass

    return t_sys, t_sys_err, gain, gain_err

#---------------------------------------------------------------
def est_pulsar_sn(pulsar, obsid,\
                 beg=None, end=None, p_ra=None, p_dec=None, obs_metadata=None, plot_flux=False,\
                 query=None, o_enter=None, o_exit=None, trcvr="/group/mwaops/PULSAR/MWA_Trcvr_tile_56.csv"):

    """
    Estimates the signal to noise ratio for a pulsar in a given observation using the radiometer equation
    S/N = (s_mean * gain * sqrt(n_p * t_int * df * (period - W_50)/W_50)) / t_sys
    Note that W_50 should be W_equiv but we can't figure that out so we're estimating

    Parameters:
    ----------
    pulsar: string
        Name of the pulsar e.g. J2241-5236
    obsid: int
        Observation ID e.g. 1226406800
    beg: int
        OPTIONAL - beginning of the observing time
    end: int
        OPTIONAL - end of the observing time
    p_ra: str
        OPTIONAL - the target's right ascension
    p_dec: str
        OPTIONAL - the target's declination
    obs_metadata: list
        OPTIONAL - the array generated from mwa_metadb_utils.get_common_obs_metadata(obsid)
    plot_flux: boolean
        OPTIONAL - whether or not to produce a plot of the flux estimation. Default = False
    o_enter: float
        OPTIONAL - The normalised o_enter time of the pulsar's coverage in the beam (between 0 and 1)
    o_exit: float
        OPTIONAL - The normalised o_exit time of the pulsar's covreage in the beam (between 0 and 1)

    Returns:
    --------
    sn: float
        The expected signal to noise ratio for the given inputs
    sn_err: float
        The uncertainty in the signal to noise ratio
    """
    #We will attain uncertainties for s_mean, gain, t_sys and W_50.
    # other uncertainties are considered negligible

    if query is None:
        query = psrqpy.QueryATNF(psrs=pulsar, loadfromdb=ATNF_LOC).pandas

    if p_ra is None or p_dec is None:
        #Get some basic pulsar and obs info info
        p_ra = query["RAJ"][0]
        p_dec = query["DECJ"][0]

    #get metadata if not supplied
    if obs_metadata is None:
        logger.debug("Obtaining obs metadata")
        obs_metadata = mwa_metadb_utils.get_common_obs_metadata(obsid)

    n_p = 2 #constant
    df = 30.72e6 #(24*1.28e6)

    #estimate flux
    s_mean, s_mean_err = est_pulsar_flux(pulsar, obsid, plot_flux=plot_flux,\
                         metadata=obs_metadata, query=query)

    #fluxes may be Nones. If so, return None
    if s_mean is None and s_mean_err is None:
        return None, None

    #find integration time
    if o_enter is not None and o_exit is not None:
        t_int = o_exit-o_enter
        if beg is not None and end is not None:
            t_int = t_int*(end-beg)
        else:
            t_int = t_int*obs_metadata[3] #duration
    else:
        beg, end, t_int = find_times(obsid, pulsar, beg=beg, end=end)
    if t_int<=0.:
        logger.warning("{} not in beam for obs files or specificed beginning and end times"\
                    .format(pulsar))
        return 0., 0.

    #find system temp and gain
    t_sys, t_sys_err, gain, gain_err = find_t_sys_gain(pulsar, obsid,\
                                beg=beg, end=end, p_ra=p_ra, p_dec=p_dec, query=query,\
                                obs_metadata=obs_metadata, trcvr=trcvr)

    #Find W_50
    W_50, W_50_err = find_pulsar_w50(pulsar, query=query)

    #calculate SN
    period = float(query["P0"][0])
    SN = ((s_mean * gain)/t_sys) * np.sqrt(n_p * t_int * df * (period - W_50)/W_50)

    #Calculate SN uncertainty using variance formula. Assuming error from period, df and t_int is zero
    dc_expr = np.sqrt((period-W_50)/W_50)
    var_s_mean = (gain * np.sqrt(n_p * t_int * df)) * dc_expr / t_sys
    var_gain = s_mean * np.sqrt(n_p * t_int * df) * dc_expr / t_sys
    var_W_50 = s_mean * gain * np.sqrt(n_p * t_int * df)/t_sys * (period/(-2.*W_50**2.)) * dc_expr**-1
    var_t_sys = -s_mean * gain * np.sqrt(n_p * t_int * df) * dc_expr / t_sys**2.
    var_s_mean      = var_s_mean**2.    * s_mean_err**2.
    var_gain        = var_gain**2.      * gain_err**2.
    var_W_50        = var_W_50**2.      * W_50_err**2.
    var_t_sys       = var_t_sys**2.     * t_sys_err**2.

    logger.debug("variance estimates for s_mean: {0}, gain: {1}, W_50: {2}, t_sys: {3}"\
                .format(var_s_mean, var_gain, var_W_50, var_t_sys))
    SN_err = np.sqrt(var_s_mean + var_gain + var_W_50 + var_t_sys)

    logger.debug("S_mean: {0} +/- {1}".format(s_mean, s_mean_err))
    logger.debug("Gain: {0} +/- {1}".format(gain, gain_err))
    logger.debug("t_int: {0}".format(t_int))
    logger.debug("df: {0}".format(df))
    logger.debug("period: {0}".format(period))
    logger.debug("W_50: {0} +/- {1}".format(W_50, W_50_err))
    logger.debug("t_sys: {0} +/- {1}".format(t_sys, t_sys_err))
    logger.info("Pulsar S/N: {0} +/- {1}".format(SN, SN_err))

    return SN, SN_err

#---------------------------------------------------------------
if __name__ == "__main__":

    loglevels = dict(DEBUG=logging.DEBUG,\
                    INFO=logging.INFO,\
                    WARNING=logging.WARNING,\
                    ERROR=logging.ERROR)

    parser = argparse.ArgumentParser(description="""A utility file for estimating the S/N of a pulsar in an obsid""")
    parser.add_argument("-o", "--obsid", type=int, help="The Observation ID (e.g. 1221399680)")
    parser.add_argument("-p", "--pulsar", type=str, help="The pulsar's name (e.g. J2241-5236)")
    parser.add_argument("-L", "--loglvl", type=str, default="INFO", help="Logger verbostity level. Running in DEBUG mode will output a plot of the power law fit where applicable. Default: INFO")
    parser.add_argument("-b", "--beg", type=int, default=None, help="The beginning time of observation. If None, will use beginning given by a metadata call")
    parser.add_argument("-e", "--end", type=int, default=None, help="The end time of observation. If None, will use the end given by a metadata call")
    parser.add_argument("--pointing", type=str, default=None, help="The pointing of the target in the format '12:34:56_98:76:54'. If None, will obtain from a call to ATNF")
    args = parser.parse_args()

    logger.setLevel(loglevels[args.loglvl])
    ch = logging.StreamHandler()
    ch.setLevel(loglevels[args.loglvl])
    formatter = logging.Formatter('%(asctime)s  %(filename)s  %(name)s  %(lineno)-4d  %(levelname)-9s :: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False

    if args.obsid==None or args.pulsar==None:
        logger.error("Obsid and Pulsar name must be supplied. Exiting...")
        sys.exit(1)

    query = psrqpy.QueryATNF(psrs=[args.pulsar], loadfromdb=ATNF_LOC).pandas
    #Decide what to use as ra and dec
    if args.pointing is None:
        raj = None
        decj = None
    else:
        raj = args.pointing.split("_")[0]
        decj = args.pointing.split("_")[1]

    if args.loglvl=="DEBUG":
        plot=True
    else:
        plot=False

    SN, SN_err = est_pulsar_sn(args.pulsar, args.obsid,\
             beg=args.beg, end=args.end, p_ra=raj, p_dec=decj, plot_flux=plot, query=query)
