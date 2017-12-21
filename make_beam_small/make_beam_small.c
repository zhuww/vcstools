#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <complex.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include "mwac_utils.h"
#include "slalib.h"
#include "slamac.h"
#include "ascii_header.h"
#include "mwa_header.h"
#include <omp.h>
#include <glob.h>
#include <fcntl.h>
#include <assert.h>
#include "beam_common.h"
#include "beam_psrfits.h"
#include "make_beam_small.h"

// Are GPU available

#ifdef HAVE_CUDA
#include "gpu_utils.h"
#include <cuda_runtime.h>
#else
#define Complex float _Complex
#endif

//
// write out psrfits directly
#include "psrfits.h"
#include "antenna_mapping.h"
#include "beamer_version.h"

#define MAX_COMMAND_LENGTH 1024

int main(int argc, char **argv) {

    // A place to hold the beamformer settings
    struct make_beam_opts opts;

    // These are used to calculate how the input data are ordered
    const int npfb = 4;
    const int nrec = 16;
    const int ninc = 4;

    /* Set default beamformer settings */

    // Variables for required options
    opts.obsid       = NULL; // The observation ID
    opts.begin       = 0;    // GPS time -- when to start beamforming
    opts.end         = 0;    // GPS time -- when to stop beamforming
    opts.time_utc    = NULL; // utc time string "yyyy-mm-ddThh:mm:ss"
    opts.dec_ddmmss  = NULL; // "dd:mm:ss"
    opts.ra_hhmmss   = NULL; // "hh:mm:ss"
    opts.datadir     = NULL; // The path to where the recombined data live
    opts.metafits    = NULL; // filename of the metafits file
    opts.rec_channel = NULL; // 0 - 255 receiver 1.28MHz channel
    opts.frequency   = 0;    // = rec_channel expressed in Hz

    // Variables for MWA/VCS configuration
    opts.nstation      = 128;    // The number of antennas
    opts.nchan         = 128;    // The number of fine channels (per coarse channel)
    opts.chan_width    = 10000;  // The bandwidth of an individual fine chanel (Hz)
    opts.sample_rate   = 10000;  // The VCS sample rate (Hz)
    opts.use_ant_flags = 0;      // Use flags in metafits file?

    // Output options
    opts.out_incoh     = 0;  // Default = incoherent output turned OFF
    opts.out_coh       = 1;  // Default = coherent   output turned ON

    // Variables for calibration settings
    opts.cal.filename          = NULL;
    opts.cal.bandpass_filename = NULL;
    opts.cal.chan_width        = 40000;
    opts.cal.nchan             = 0;
    opts.cal.cal_type          = NO_CALIBRATION;
    opts.cal.offr_chan_num     = 0;

    // Parse command line arguments
    make_beam_parse_cmdline( argc, argv, &opts );

    // Create "shorthand" variables for options that are used frequently
    int nstation       = opts.nstation;
    int nchan          = opts.nchan;
    const int npol     = 2;      // (X,Y)
    const int outpol   = 4;      // (I,Q,U,V)

    // Start counting time from here (i.e. after parsing the command line)
    double begintime = omp_get_wtime();
    printf("[%f]  Starting make_beam\n", omp_get_wtime()-begintime);

    // Calculate the number of files
    int nfiles = opts.end - opts.begin + 1;
    if (nfiles <= 0) {
        fprintf(stderr, "Cannot beamform on %d files (between %lu and %lu)\n", nfiles, opts.begin, opts.end);
        exit(EXIT_FAILURE);
    }

    // Allocate memory for the file name list
    char **filenames = NULL;
    filenames = (char **)malloc( nfiles*sizeof(char *) );

    // Allocate memory and write filenames
    int second;
    unsigned long int timestamp;
    for (second = 0; second < nfiles; second++) {
        timestamp = second + opts.begin;
        filenames[second] = (char *)malloc( MAX_COMMAND_LENGTH*sizeof(char) );
        sprintf( filenames[second], "%s/%s_%ld_ch%s.dat", opts.datadir, opts.obsid, timestamp, opts.rec_channel );
    }

    // Allocate memory for complex weights matrices
    int ant, p, ch; // Loop variables

    complex double  ***complex_weights_array = NULL; // [ant][ch][pol]
    
    complex_weights_array = (complex double ***)malloc( nstation * sizeof(complex double **) );
    for (ant = 0; ant < nstation; ant++) {
        complex_weights_array[ant] = (complex double **)malloc( nchan * sizeof(complex double *) );
        for (ch = 0; ch < nchan; ch++) {
            complex_weights_array[ant][ch] = (complex double *)malloc( npol * sizeof(complex double) );
        }
    }

    // Allocate memory for (inverse) Jones matrices
    complex double ****invJi = NULL; // [ant][ch][pol][pol]

    invJi = (complex double ****)malloc( nstation * sizeof(complex double ***) );
    for (ant = 0; ant < nstation; ant++) {
        invJi[ant] =(complex double ***)malloc( nchan * sizeof(complex double **) );
        for (ch = 0; ch < nchan; ch++) {
            invJi[ant][ch] = (complex double **)malloc( npol * sizeof(complex double *) );
            for (p = 0; p < npol; p++) {
                invJi[ant][ch][p] = (complex double *)malloc( npol * sizeof(complex double) );
            }
        }
    }

    // Structures for holding psrfits header information
    struct psrfits pf;
    struct psrfits pf_incoh;

    // Read in info from metafits file
    printf("[%f]  Reading in metafits file information from %s\n", omp_get_wtime()-begintime, opts.metafits);
    struct metafits_info mi;
    get_metafits_info( opts.metafits, &mi, opts.chan_width );

    // If using bandpass calibration solutions, calculate number of expected bandpass channels
    if (opts.cal.cal_type == RTS_BANDPASS)
        opts.cal.nchan = (nchan * opts.chan_width) / opts.cal.chan_width;

    int i;
    if (!opts.use_ant_flags)
        for (i = 0; i < nstation*npol; i++)
            mi.weights_array[i] = 1.0;

    double wgt_sum = 0;
    for (i = 0; i < nstation*npol; i++)
        wgt_sum += mi.weights_array[i];
    double invw = 1.0/wgt_sum;

    // Run get_delays to populate the delay_vals struct
    printf("[%f]  Setting up output header information\n", omp_get_wtime()-begintime);
    struct delays delay_vals;
    get_delays(
            opts.dec_ddmmss,    // dec as a string "dd:mm:ss"
            opts.ra_hhmmss,     // ra  as a string "hh:mm:ss"
            opts.frequency,     // middle of the first frequency channel in Hz
            &opts.cal,          // struct holding info about calibration
            opts.sample_rate,   // = 10000 samples per sec
            opts.time_utc,      // utc time string
            0.0,           // seconds offset from time_utc at which to calculate delays
            &delay_vals,   // Populate psrfits header info
            &mi,           // Struct containing info from metafits file
            NULL,          // complex weights array (ignore this time)
            NULL           // invJi array           (ignore this time)
    );

    // now we need to create a fits file and populate its header
    populate_psrfits_header( &pf, opts.metafits, opts.obsid, opts.time_utc, opts.sample_rate,
            opts.frequency, nchan, opts.chan_width, outpol, opts.rec_channel, &delay_vals );

    // If incoherent sum requested, populate incoherent psrfits header
    if (opts.out_incoh)
    {
        int incoh_npol = 1;
        populate_psrfits_header( &pf_incoh, opts.metafits, opts.obsid, opts.time_utc, opts.sample_rate,
                opts.frequency, nchan, opts.chan_width, incoh_npol, opts.rec_channel, &delay_vals );

        // The above sets the RA and Dec to be the beamforming pointing, but it ought
        // to have the tile pointing, not the beam pointing
        pf_incoh.hdr.ra2000  = mi.tile_pointing_ra;
        pf_incoh.hdr.dec2000 = mi.tile_pointing_dec;
        pf_incoh.hdr.summed_polns = 1;

        // Also, we need to give it a different base name for the output files
        sprintf(pf_incoh.basefilename, "%s_%s_ch%03d_incoh",
                pf_incoh.hdr.project_id, pf_incoh.hdr.source, atoi(opts.rec_channel));
    }

    // Create array for holding the raw data
    int bytes_per_file = opts.sample_rate * nstation * npol * nchan;
    uint8_t *data = (uint8_t *)malloc( bytes_per_file * sizeof(uint8_t) );
    assert(data);

    int8_t *out_buffer_8_psrfits = (int8_t *)malloc( outpol*nchan*pf.hdr.nsblk * sizeof(int8_t) );
    float  *data_buffer_psrfits  =  (float *)malloc( nchan*outpol*pf.hdr.nsblk * sizeof(float) );
    int8_t *out_buffer_8_incoh   = NULL;
    float  *data_buffer_incoh    = NULL;
    if (opts.out_incoh)
    {
        out_buffer_8_incoh = (int8_t *)malloc( nchan*pf_incoh.hdr.nsblk * sizeof(int8_t) );
        data_buffer_incoh = (float *)malloc( nchan*pf_incoh.hdr.nsblk * sizeof(float) );
    }

    int offset_in_psrfits;
    int offset_in_incoh;

    int file_no = 0;
    int sample;

    printf("[%f]  **BEGINNING BEAMFORMING**\n", omp_get_wtime()-begintime);
    for (file_no = 0; file_no < nfiles; file_no++) {

        // Read in data from next file
        printf("[%f]  Reading in data from %s [%d/%d]\n", omp_get_wtime()-begintime,
                filenames[file_no], file_no+1, nfiles);
        read_data( filenames[file_no], data, bytes_per_file  );

        // Get the next second's worth of phases / jones matrices, if needed
        printf("[%f]  Calculating delays\n", omp_get_wtime()-begintime);
        get_delays(
                opts.dec_ddmmss,    // dec as a string "dd:mm:ss"
                opts.ra_hhmmss,     // ra  as a string "hh:mm:ss"
                opts.frequency,     // middle of the first frequency channel in Hz
                &opts.cal,          // struct holding info about calibration
                opts.sample_rate,   // = 10000 samples per sec
                opts.time_utc,      // utc time string
                (double)file_no, // seconds offset from time_utc at which to calculate delays
                NULL,          // Don't update delay_vals
                &mi,           // Struct containing info from metafits file
                complex_weights_array,  // complex weights array (answer will be output here)
                invJi );       // invJi array           (answer will be output here)

        printf("[%f]  Calculating beam\n", omp_get_wtime()-begintime);

        offset_in_psrfits  = 0;
        offset_in_incoh    = 0;

        for (i = 0; i < nchan*outpol*pf.hdr.nsblk; i++)
            data_buffer_psrfits[i] = 0.0;

        if (opts.out_incoh)
            for (i = 0; i < nchan*pf_incoh.hdr.nsblk; i++)
                data_buffer_incoh[i] = 0.0;

#pragma omp parallel for
        for (sample = 0; sample < (int)opts.sample_rate; sample++ ) {

            int ch, ant, pol, opol, opol1, opol2;
            uint8_t uD, uDr, uDi;
            int sDr, sDi;
            int pfb, rec, inc;
            int data_idx;

            complex float beam[nchan][nstation][npol];
            float         incoh_beam[nchan][nstation][npol];
            complex float detected_beam[nchan][npol];
            float         detected_incoh_beam[nchan];
            float         spectrum[nchan*outpol];
            complex float noise_floor[nchan][npol][npol];
            complex float e_true[npol], e_dash[npol];


            // Initialise noise floor to zero
            for (ch    = 0; ch    < nchan;  ch++   )
            for (opol1 = 0; opol1 < npol; opol1++)
            for (opol2 = 0; opol2 < npol; opol2++)
                noise_floor[ch][opol1][opol2] = 0.0;

            // Initialise detected beam to zero
            for (ch  = 0; ch  < nchan   ; ch++ )
            for (pol = 0; pol < npol    ; pol++)
                detected_beam[ch][pol] = 0.0 + 0.0*I;

            // Initialise incoherent beam arrays to zero, if necessary
            if (opts.out_incoh)
                for (ch  = 0; ch  < nchan   ; ch++ )
                    detected_incoh_beam[ch] = 0.0;

            // Calculate the beam, noise floor
            for (ant = 0; ant < nstation; ant++) {

                // Get the index for the data that corresponds to this
                //   sample, channel, antenna, polarisation
                // Justification for the rather bizarre mapping is found in
                // the docs.
                // (rec depends on polarisation, so is calculating in the inner loop)
                pfb = ant / 32;
                inc = (ant / 8) % 4;

                for (ch = 0; ch < nchan; ch++ ) {

                    // Calculate quantities that depend only on "input" polarisation
                    for (pol = 0; pol < npol    ; pol++) {

                        rec = (2*ant+pol) % 16;

                        data_idx = sample * (ninc*nrec*npfb*nchan) +
                                   ch     * (ninc*nrec*npfb)       +
                                   pfb    * (ninc*nrec)            +
                                   rec    * (ninc)                 +
                                   inc;

                        uD  = data[data_idx];
                        uDr = uD & 0xf;        // Real part = least significant nibble
                        uDi = (uD >> 4) & 0xf; // Imag part = most  significant nibble

                        // Convert from unsigned to signed
                        sDr = (uDr >= 0x8 ? (signed int)uDr - 0x10 : (signed int) uDr);
                        sDi = (uDi >= 0x8 ? (signed int)uDi - 0x10 : (signed int) uDi);

                        // Form a single complex number
                        e_dash[pol]  = (float)sDr + (float)sDi * I;

                        // Detect the incoherent beam, if requested
                        if (opts.out_incoh)
                            incoh_beam[ch][ant][pol] = creal(e_dash[pol] * conj(e_dash[pol]));

                        // Apply complex weights
                        e_dash[pol] *= complex_weights_array[ant][ch][pol];

                    }

                    // Calculate quantities that depend on output polarisation
                    // (i.e. apply inv(jones))
                    for (pol = 0; pol < npol; pol++) {

                        e_true[pol] = 0.0 + 0.0*I;

                        for (opol = 0; opol < npol; opol++)
                            e_true[pol] += invJi[ant][ch][pol][opol] * e_dash[opol];

                        for (opol = 0; opol < npol; opol++)
                            noise_floor[ch][pol][opol] += e_true[pol] * conj(e_true[opol]);

                        beam[ch][ant][pol] = e_true[pol];
                    }
                }
            }

            // Detect the beam = sum over antennas
            for (ant = 0; ant < nstation; ant++)
            for (pol = 0; pol < npol    ; pol++)
            for (ch  = 0; ch  < nchan   ; ch++ )
            {
                detected_beam[ch][pol] += beam[ch][ant][pol];

                // ...and the incoherent beam, if requested
                if (opts.out_incoh)
                    detected_incoh_beam[ch] += incoh_beam[ch][ant][pol];
            }

            // Calculate the Stokes parameters
            double beam00, beam11;
            double noise0, noise1, noise3;
            complex double beam01;
            unsigned int stokesIidx, stokesQidx, stokesUidx, stokesVidx;

            for (ch = 0; ch < nchan; ch++) {

                beam00 = (double)(detected_beam[ch][0] * conj(detected_beam[ch][0]));
                beam11 = (double)(detected_beam[ch][1] * conj(detected_beam[ch][1]));
                beam01 = detected_beam[ch][0] * conj(detected_beam[ch][1]);

                noise0 = noise_floor[ch][0][0];
                noise1 = noise_floor[ch][0][1];
                noise3 = noise_floor[ch][1][1];

                stokesIidx = 0*nchan + ch;
                stokesQidx = 1*nchan + ch;
                stokesUidx = 2*nchan + ch;
                stokesVidx = 3*nchan + ch;

                // Looking at the dspsr loader the expected order is <ntime><npol><nchan>
                // so for a single timestep we do not have to interleave - I could just stack these
                spectrum[stokesIidx] = (beam00 + beam11 - noise0 - noise3) * invw;
                spectrum[stokesQidx] = (beam00 - beam11 - noise0 + noise3) * invw;
                spectrum[stokesUidx] = 2.0 * (creal(beam01) - noise1)*invw;
                spectrum[stokesVidx] = -2.0 * cimag((beam01 - noise1)*invw);
            }

            offset_in_psrfits  = sizeof(float)*nchan*outpol * sample;
            offset_in_incoh    = sizeof(float)*nchan * sample;

            memcpy((void *)((char *)data_buffer_psrfits + offset_in_psrfits), spectrum, sizeof(float)*nchan*outpol);

            if (opts.out_incoh)
                memcpy((void *)((char *)data_buffer_incoh + offset_in_incoh), detected_incoh_beam, sizeof(float)*nchan);

        }

        // We've arrived at the end of a second's worth of data...

        printf("[%f]  Flattening bandpass\n", omp_get_wtime()-begintime);
        flatten_bandpass(pf.hdr.nsblk, nchan, outpol,
                data_buffer_psrfits, pf.sub.dat_scales,
                pf.sub.dat_offsets, 32, 0, 1, 1, 1, 0);

        float2int8_trunc(data_buffer_psrfits, pf.hdr.nsblk*nchan*outpol,
                -126.0, 127.0, out_buffer_8_psrfits);

        int8_to_uint8(pf.hdr.nsblk*nchan*outpol, 128,
                (char *) out_buffer_8_psrfits);

        memcpy(pf.sub.data, out_buffer_8_psrfits, pf.sub.bytes_per_subint);

        if (opts.out_incoh)
        {
            flatten_bandpass(pf_incoh.hdr.nsblk, nchan, 1,
                    data_buffer_incoh, pf_incoh.sub.dat_scales,
                    pf_incoh.sub.dat_offsets, 32, 0, 1, 1, 1, 0);

            float2int8_trunc(data_buffer_incoh, pf_incoh.hdr.nsblk*nchan,
                    -126.0, 127.0, out_buffer_8_incoh);

            int8_to_uint8(pf_incoh.hdr.nsblk*nchan, 128,
                    (char *) out_buffer_8_incoh);

            memcpy(pf_incoh.sub.data, out_buffer_8_incoh, pf_incoh.sub.bytes_per_subint);
        }

        printf("[%f]  Writing data to file\n", omp_get_wtime()-begintime);
        if (psrfits_write_subint(&pf) != 0) {
            fprintf(stderr, "Write subint failed file exists?\n");
            break; // Exit from while loop
        }

        pf.sub.offs = roundf(pf.tot_rows * pf.sub.tsubint) + 0.5*pf.sub.tsubint;
        pf.sub.lst += pf.sub.tsubint;

        if (opts.out_incoh)
        {
            if (psrfits_write_subint(&pf_incoh) != 0) {
                fprintf(stderr, "Write incoherent subint failed file exists?\n");
                break; // Exit from while loop
            }

            pf_incoh.sub.offs = roundf(pf_incoh.tot_rows * pf_incoh.sub.tsubint) +
                                0.5*pf_incoh.sub.tsubint;
            pf_incoh.sub.lst += pf_incoh.sub.tsubint;
        }

    }

    printf("[%f]  **FINISHED BEAMFORMING**\n", omp_get_wtime()-begintime);
    printf("[%f]  Starting clean-up\n", omp_get_wtime()-begintime);

    if (pf.status == 0) {
        correct_psrfits_stt( &pf );

        flatten_bandpass(pf.hdr.nsblk, nchan, outpol, data_buffer_psrfits,
                pf.sub.dat_scales, pf.sub.dat_offsets, 32, 0, 0, 0, 0, 1);
    }

    if (pf_incoh.status == 0) {
        correct_psrfits_stt( &pf_incoh );

        flatten_bandpass(pf_incoh.hdr.nsblk, nchan, 1, data_buffer_incoh,
                pf_incoh.sub.dat_scales, pf_incoh.sub.dat_offsets, 32, 0, 0, 0, 0, 1);
    }

    // Free up memory for filenames
    if (opts.datadir) {
        int second;
        for (second = 0; second < nfiles; second++)
            free( filenames[second] );
        free( filenames );
    }

    destroy_metafits_info( &mi );
    free( out_buffer_8_psrfits );
    free( out_buffer_8_incoh );
    free( data_buffer_psrfits  );
    free( data_buffer_incoh  );
    free( data );

    return 0;
}


void usage() {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: make_beam_small [OPTIONS]\n");

    fprintf(stderr, "\n");
    fprintf(stderr, "REQUIRED OPTIONS\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\t-o, --obsid=GPSTIME       ");
    fprintf(stderr, "Observation ID (GPS seconds).\n");
    fprintf(stderr, "\t-b, --begin=GPSTIME       ");
    fprintf(stderr, "Begin time of observation, in GPS seconds\n");
    fprintf(stderr, "\t-e, --end=GPSTIME         ");
    fprintf(stderr, "End time of observation, in GPS seconds\n");
    fprintf(stderr, "\t-z, --utc-time=UTCTIME    ");
    fprintf(stderr, "The UTC time that corresponds to the GPS time given by the -b\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "option. UTCTIME must have the format: yyyy-mm-ddThh:mm:ss\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\t-D, --dec=dd:mm:ss.s      ");
    fprintf(stderr, "Declination of pointing direction\n");
    fprintf(stderr, "\t-R, --ra=hh:mm:ss.s       ");
    fprintf(stderr, "Right ascension of pointing direction\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\t-d, --data-location=PATH  ");
    fprintf(stderr, "PATH is the directory containing the recombined data\n");
    fprintf(stderr, "\t-m, --metafits-file=FILE  ");
    fprintf(stderr, "FILE is the metafits file pertaining to the OBSID given by the\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr,  "-o option\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\t-f, --coarse-chan=N       ");
    fprintf(stderr, "Absolute coarse channel number (0-255)\n");

    fprintf(stderr, "\n");
    fprintf(stderr, "OUTPUT OPTIONS\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\t-i, --incoh               ");
    fprintf(stderr, "Turn on incoherent beam output (off by default)\n");
    fprintf(stderr, "\t--no-psrfits              ");
    fprintf(stderr, "Turn off PSRFITS output (on by default) (NOT YET IMPLEMENTED)\n");

    fprintf(stderr, "\n");
    fprintf(stderr, "MWA/VCS CONFIGURATION OPTIONS\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\t-a, --antennas=N          ");
    fprintf(stderr, "The number of antennas in the array. For MWA Phase 2, N=128.     ");
    fprintf(stderr, "[default: 128]\n");
    fprintf(stderr, "\t-n, --num-fine-chans=N    ");
    fprintf(stderr, "The number of fine channels per coarse channel.                  ");
    fprintf(stderr, "[default: 128]\n");
    fprintf(stderr, "\t-w, --fine-chan-width=N   ");
    fprintf(stderr, "The bandwidth of an individual fine channel (Hz).                ");
    fprintf(stderr, "[default: 10000]\n");
    fprintf(stderr, "\t-r, --sample-rate=N       ");
    fprintf(stderr, "The VCS sample rate, in Hz. (The sample rate given in the meta-  ");
    fprintf(stderr, "[default: 10000]\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "metafits file matches the correlator settings at the time of\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "the observation, which is not necessarily the same as that of\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "the VCS. Hence the necessity of this option.)\n");
    fprintf(stderr, "\t-F, --use-ant-flags       ");
    fprintf(stderr, "Only include those antennas in the beamformer that have not      ");
    fprintf(stderr, "[default: off]\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "been flagged in the metafits file given by the -m option.\n");

    fprintf(stderr, "\n");
    fprintf(stderr, "CALIBRATION OPTIONS (RTS)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\t-J, --dijones-file=PATH   ");
    fprintf(stderr, "The direction-independent Jones matrix file that is output from\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "the RTS. Using this option instructs the beamformer to use the\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "RTS-generated calibration solution. Either -J or -O must be\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "supplied. If both are supplied the one that comes last will\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "override the former.\n");
    fprintf(stderr, "\t-B, --bandpass-file=PATH  ");
    fprintf(stderr, "The bandpass file that is output from the RTS. If this option\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "is given, the RTS calibration solution will be applied to each\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "fine channel. If -J is supplied but -B is not, then the coarse\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "channel solution will be applied to ALL fine channels\n");
    fprintf(stderr, "\t-W, --rts-chan-width      ");
    fprintf(stderr, "RTS calibration channel bandwidth (Hz)                           ");
    fprintf(stderr, "[default: 40000]\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "CALIBRATION OPTIONS (OFFRINGA)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\t-O, --offringa-file=PATH  ");
    fprintf(stderr, "The calibration solution file that is output from the tools\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "made by Andre Offringa. Using this option instructs the beam-\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "former to use the Offringa-style calibration solution. Either\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "-J or -O must be supplied. If both are supplied the one that\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "comes last will override the former.\n");
    fprintf(stderr, "\t-C, --offringa-chan=N     ");
    fprintf(stderr, "The zero-offset position of the coarse channel solution in the   ");
    fprintf(stderr, "[default: 0]\n");
    fprintf(stderr, "\t                          ");
    fprintf(stderr, "calibration file given by the -O option.\n");

    fprintf(stderr, "\n");
    fprintf(stderr, "OTHER OPTIONS\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\t-h, --help                ");
    fprintf(stderr, "Print this help and exit\n");
    fprintf(stderr, "\t-V, --version             ");
    fprintf(stderr, "Print version number and exit\n");
    fprintf(stderr, "\n");
}



void make_beam_parse_cmdline(
        int argc, char **argv, struct make_beam_opts *opts )
{
    if (argc > 1) {

        int c;
        while (1) {

            static struct option long_options[] = {
                {"obsid",           required_argument, 0, 'o'},
                {"begin",           required_argument, 0, 'b'},
                {"end",             required_argument, 0, 'e'},
                {"incoh",           no_argument,       0, 'i'},
                {"utc-time",        required_argument, 0, 'z'},
                {"dec",             required_argument, 0, 'D'},
                {"ra",              required_argument, 0, 'R'},
                {"data-location",   required_argument, 0, 'd'},
                {"metafits-file",   required_argument, 0, 'm'},
                {"coarse-chan",     required_argument, 0, 'f'},
                {"antennas",        required_argument, 0, 'a'},
                {"num-fine-chans",  required_argument, 0, 'n'},
                {"fine-chan-width", required_argument, 0, 'w'},
                {"sample-rate",     required_argument, 0, 'r'},
                {"use-ant-flags",   no_argument,       0, 'F'},
                {"dijones-file",    required_argument, 0, 'J'},
                {"bandpass-file",   required_argument, 0, 'B'},
                {"rts-chan-width",  required_argument, 0, 'W'},
                {"offringa-file",   required_argument, 0, 'O'},
                {"offringa-chan",   required_argument, 0, 'C'},
                {"help",            required_argument, 0, 'h'},
                {"version",         required_argument, 0, 'V'}
            };

            int option_index = 0;
            c = getopt_long( argc, argv,
                             "a:b:B:C:d:D:e:f:FhiJ:m:n:o:O:r:R:Vw:W:z:",
                             long_options, &option_index);
            if (c == -1)
                break;

            switch(c) {

                case 'a':
                    opts->nstation = atoi(optarg);
                    break;
                case 'b':
                    opts->begin = atol(optarg);
                    break;
                case 'B':
                    opts->cal.bandpass_filename = strdup(optarg);
                    opts->cal.cal_type = RTS_BANDPASS;
                    break;
                case 'C':
                    opts->cal.offr_chan_num = atoi(optarg);
                    break;
                case 'd':
                    opts->datadir = strdup(optarg);
                    break;
                case 'D':
                    opts->dec_ddmmss = strdup(optarg);
                    break;
                case 'e':
                    opts->end = atol(optarg);
                    break;
                case 'f':
                    opts->rec_channel = strdup(optarg);
                    // The base frequency of the coarse channel in Hz
                    opts->frequency = atoi(optarg) * 1.28e6 - 640e3;
                    break;
                case 'F':
                    opts->use_ant_flags = 1;
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'i':
                    opts->out_incoh = 1;
                    break;
                case 'J':
                    opts->cal.filename = strdup(optarg);
                    if (opts->cal.cal_type != RTS_BANDPASS)
                        opts->cal.cal_type = RTS;
                    break;
                case 'm':
                    opts->metafits = strdup(optarg);
                    break;
                case 'n':
                    opts->nchan = atoi(optarg);
                    break;
                case 'o':
                    opts->obsid = strdup(optarg);
                    break;
                case 'O':
                    opts->cal.filename = strdup(optarg);
                    opts->cal.cal_type = OFFRINGA;
                    break;
                case 'r':
                    opts->sample_rate = atoi(optarg);
                    break;
                case 'R':
                    opts->ra_hhmmss = strdup(optarg);
                    break;
                case 'V':
                    printf("%s\n", MAKE_BEAM_VERSION);
                    exit(0);
                    break;
                case 'w':
                    opts->chan_width = atoi(optarg);
                    break;
                case 'W':
                    opts->cal.chan_width = atoi(optarg);
                    break;
                case 'z':
                    opts->time_utc = strdup(optarg);
                    break;
                default:
                    fprintf(stderr, "error: make_beam_parse_cmdline: "
                                    "unrecognised option '%s'\n", optarg);
                    usage();
                    exit(EXIT_FAILURE);
            }
        }
    }
    else {
        usage();
        exit(EXIT_FAILURE);
    }

    // Check that all the required options were supplied
    assert( opts->obsid        != NULL );
    assert( opts->begin        != 0    );
    assert( opts->end          != 0    );
    assert( opts->time_utc     != NULL );
    assert( opts->dec_ddmmss   != NULL );
    assert( opts->ra_hhmmss    != NULL );
    assert( opts->datadir      != NULL );
    assert( opts->metafits     != NULL );
    assert( opts->rec_channel  != NULL );
    assert( opts->cal.cal_type != NO_CALIBRATION );
    assert( opts->out_incoh || opts->out_coh );

}
