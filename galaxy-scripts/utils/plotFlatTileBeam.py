#!/usr/bin/env python

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.coordinates import SkyCoord,EarthLocation,AltAz
from astropy.time import Time
import astropy.units as u
import argparse


def plot_beam_pattern(obsid, obsfreq, obstime, ra, dec, cutoff=0.1):
	
	# extra imports from MWA_Tools to access database and beam models
	from mwapy.pb import primary_beam as pb
	from mwapy.pb import mwa_db_query as dbq

	_az = np.linspace(0, 360, 3600)
	_za = np.linspace(0, 90, 900)
	az, za = np.meshgrid(_az, _za)
	
	## TARGET TRACKING ## 
	times = Time(obstime, format='gps')
	target = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
	location = EarthLocation(lat=-26.7033 * u.deg,
                             lon=116.671 * u.deg,
                             height=377.827 * u.m)	
	
	altaz = target.transform_to(AltAz(obstime=times, location=location))
	targetAZ = altaz.az.deg
	targetZA = 90 - altaz.alt.deg
	colours = cm.viridis(np.linspace(1, 0, len(targetAZ)))

	## MWA BEAM CALCULATIONS ##
	xdelays, ydelays = dbq.get_delays_obsid(obsid)[0] # x-pol and y-pol delays for obsid
	ptAZ, _, ptZA = dbq.get_beam_pointing(obsid) # Az and ZA in degrees for obsid	

	print "obs pointing: ({0}, {1})".format(ptAZ,ptZA)

	xp,yp = pb.MWA_Tile_full_EE(np.radians(za), np.radians(az), obsfreq*1e6, delays=[xdelays,ydelays], zenithnorm=True, interp=True)
	pattern = np.sqrt(xp**2+yp**2) # sum to get "total intensity"
	pmax = pattern.max()
	hpp = 0.5 * pmax # half-power point
	print "tile pattern maximum: {0:.3f}".format(pmax)
	print "tile pattern half-max: {0:.3f}".format(hpp)
	pattern[np.where(pattern < cutoff)] = 0 # ignore everything below cutoff 

	# figure out the fwhm
	fwhm_idx = np.where((pattern > 0.498*pmax) & (pattern < 0.502*pmax))
	fwhm_az_idx = fwhm_idx[1]
	fwhm_za_idx = fwhm_idx[0]

	# collapse beam pattern along axes
	pattern_ZAcol = pattern.mean(axis=0)
	pattern_AZcol = pattern.mean(axis=1)

	# figure out beam pattern value at target tracking points
	track_lines = []
	print "beam power at target track points:"
	for ta,tz in zip(targetAZ, targetZA):
		xp, yp = pb.MWA_Tile_full_EE(np.radians([[tz]]), np.radians([[ta]]), obsfreq*1e6, delays=[xdelays,ydelays], zenithnorm=True, interp=False)
		bp = (xp + yp) / 2
		track_lines.append(bp[0])
		print "({0:.2f},{1:.2f}) = {2:.3f}".format(ta, tz, bp[0][0])

	
	## PLOTTING ##
	fig = plt.figure(figsize=(10,8))
	gs = gridspec.GridSpec(4,4)
	axP = plt.subplot(gs[1:,0:3])
	axAZ = plt.subplot(gs[0,:3])
	axZA = plt.subplot(gs[1:,3])
	axtxt = plt.subplot(gs[0,3])

	# info text in right-hand corner axis
	axtxt.axis('off')
    infostr = """Obs ID: {0}
    Frequency: {1:.2f}MHz
    Beam Pmax: {2:.3f}
    Beam half-Pmax: {3:.3f}
    """.format(obsid, obsfreq, pmax,hpp)
	axtxt.text(0.01, 0.5, infostr, verticalalignment='center')

	# plot the actual beam patter over sky
	p = axP.contourf(az, za, pattern, 100, cmap=plt.get_cmap('gist_yarg'), vmax=pmax) # plot beam contours
	axP.scatter(_az[fwhm_az_idx], _za[fwhm_za_idx], marker='.', s=1, color='r') # plot the fwhm border
	axP.plot(ptAZ, ptZA, marker="+", ms=8, ls="", color='C0') # plot the tile beam pointing
	for ta,tz,c in zip(targetAZ, targetZA, colours):
		axP.plot(ta, tz, marker="x", s=8, ls="", color=c) # plot the target track through the beam
	axP.set_xlim(0, 360)
	axP.set_ylim(0, 90)
	axP.set_xticks(np.arange(0, 361, 60))
	axP.set_xlabel("Azimuth (deg)")
	axP.set_ylabel("Zenith angle (deg)")

	# setup and configure colourbar
	cbar_ax = fig.add_axes([0.122, -0.01, 0.58, 0.03])
	cbar = plt.colorbar(p, cax=cbar_ax, orientation='horizontal', label="Zenith normalised power")
	cbar.ax.plot(hpp/pmax, [0.5], 'r.')
	for l,c in zip(track_lines,colours):
		cbar.ax.plot([l/pmax,l/pmax], [0,1], color=c, lw=1.5)
	
	# plot collapsed beam patterns:
	axAZ.plot(_az, pattern_ZAcol, color='k') # collapsed along ZA
	for ta,c in zip(targetAZ, colours): # draw tracking points
		axAZ.axvline(ta, color=c)
	axAZ.set_xlim(0, 360)
	axAZ.set_xticks(np.arange(0, 361, 60))
	axAZ.set_xticklabels([])
	axAZ.set_yscale('log')

	axZA.plot(pattern_AZcol, _za, color='k') # collapsed along AZ
	for tz,c in zip(targetZA, colours): # draw tracking points
		axZA.axhline(tz, color=c)
	axZA.set_ylim(0, 90)	
	axZA.set_yticklabels([])
	axZA.set_xscale('log')

	plt.savefig("{0}_{1:.2f}MHz_flattile.png".format(obsid, obsfreq), bbox_inches='tight')



parser = argparse.ArgumentParser()
parser.add_argument("-o", "--obsid", type=int, help="Observation ID")
parser.add_argument("-f", "--freq", type=float, help="Observing frequency (MHz)")
parser.add_argument("-t", "--times", type=int, nargs='+', help="GPS seconds to evaluate target positions. For multiple values, provide a space-separated list.")
parser.add_argument("-c", "--cutoff", type=float, help="Cut-off value for beam pattern [default: 0.1]", default=0.1)
parser.add_argument("--ra", type=str, help="RAJ2000 of target")
parser.add_argument("--dec", type=str, help="DECJ2000 of target")

args = parser.parse_args()

# do the things
plot_beam_pattern(args.obsid, args.freq, args.times, args.ra, args.dec, args.cutoff)