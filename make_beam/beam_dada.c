/********************************************************
 *                                                      *
 * Licensed under the Academic Free License version 3.0 *
 *                                                      *
 ********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "psrfits.h"
#include "beam_common.h"
#include "beam_dada.h"
#include "mycomplex.h"

#ifndef HAVE_CUDA
#include <omp.h>
#endif


void dada_write_header( FILE *df, dada_header *dhdr )
/* Formats and writes a DADA header to a specified file stream
 */
{
    int nchars = 0;
    nchars += fprintf( df, "HDR_SIZE %d\n",            dhdr->hdr_size );
    nchars += fprintf( df, "POPULATED %d\n",           dhdr->populated );
    nchars += fprintf( df, "OBS_ID %s\n",              dhdr->obsid );
    nchars += fprintf( df, "SUBOBS_ID %s\n",           dhdr->subobs_id );
    nchars += fprintf( df, "COMMAND %s\n",             dhdr->command );
    nchars += fprintf( df, "UTC_START %s\n",           dhdr->utc_start);
    nchars += fprintf( df, "OBS_OFFSET %d\n",          dhdr->obs_offset );
    nchars += fprintf( df, "NBIT %d\n",                dhdr->nbit );
    nchars += fprintf( df, "NPOL %d\n",                dhdr->npol );
    nchars += fprintf( df, "NTIMESAMPLES %d\n",        dhdr->ntimesamples );
    nchars += fprintf( df, "NINPUTS %d\n",             dhdr->ninputs );
    nchars += fprintf( df, "NINPUTS_XGPU %d\n",        dhdr->ninputs_xgpu );
    nchars += fprintf( df, "METADATA_BEAMS %d\n",      dhdr->metadata_beams );
    nchars += fprintf( df, "APPLY_PATH_WEIGHTS %d\n",  dhdr->apply_path_weights );
    nchars += fprintf( df, "APPLY_PATH_DELAYS %d\n",   dhdr->apply_path_delays );
    nchars += fprintf( df, "INT_TIME_MSEC %d\n",       dhdr->inttime_msec );
    nchars += fprintf( df, "FSCRUNCH_FACTOR %d\n",     dhdr->fscrunch_factor )
    nchars += fprintf( df, "TRANSFER_SIZE %d\n",       dhdr->transfer_size );
    nchars += fprintf( df, "PROJ_ID %s\n",             dhdr->proj_id );
    nchars += fprintf( df, "EXPOSURE_SECS %d\n",       dhdr->exptime_sec );
    nchars += fprintf( df, "COARSE_CHANNEL %d\n",      dhdr->coarse_channel );
    nchars += fprintf( df, "CORR_COARSE_CHANNEL %d\n", dhdr->corr_coarse_channel );
    nchars += fprintf( df, "SECS_PER_SUBOBS %d\n",     dhdr->secs_per_subobs );
    nchars += fprintf( df, "UNIXTIME %d\n",            dhdr->start_uxtime );
    nchars += fprintf( df, "UNIXTIME_MSEC %d\n",       dhdr->uxtime_msec );
    nchars += fprintf( df, "FINE_CHAN_WIDTH_HZ %d\n",  dhdr->fine_chan_width_hz );
    nchars += fprintf( df, "NFINE_CHAN %d\n",          dhdr->nfine_chan );
    nchars += fprintf( df, "BANDWIDTH_HZ %d\n",        dhdr->bandwidth_hz );
    nchars += fprintf( df, "SAMPLE_RATE %d\n",         dhdr->sample_rate );
    nchars += fprintf( df, "MC_IP %d.%d.%d.%d",        dhdr->mc_ip[0],
                                                       dhdr->mc_ip[1],
                                                       dhdr->mc_ip[2],
                                                       dhdr->mc_ip[3] );
    nchars += fprintf( df, "MC_PORT %d\n",             dhdr->mc_port );
    nchars += fprintf( df, "MC_SRC_IP %d.%d.%d.%d\n",  dhdr->mc_src_ip[0],
                                                       dhdr->mc_src_ip[1],
                                                       dhdr->mc_src_ip[2],
                                                       dhdr->mc_src_ip[3] );
    nchars += fprintf( df, "FILE_SIZE %d\n",           dhdr->file_size );
    nchars += fprintf( df, "FILE_NUMBER %d\n",         dhdr->file_number );

    // Check that header isn't too big
    if (nchars > dhdr->hdr_size)
    {
        fprintf( stderr, "error: dada_write_header: " );
        fprintf( stderr, "DADA header bigger than stated header size (hdr_size)\n" );
        exit(EXIT_FAILURE);
    }

    // Pad the heeader with null bytes
    int i;
    for (i = 0; i < dhdr->hdr_size - nchars; i++)
        fprintf( df, "%c", '\0' );
}


void populate_dada_header(
        dada_header     *dhdr,
        char            *metafits,
        char            *obsid,
        char            *time_utc,
        int              sample_rate,
        int              nchan,
        long int         chan_width,
        char            *rec_channel )
/* Fills in the dada_header struct with values appropriate for this bemforming job
 */
{
    int i; // Generic counter
    int nsecs = 200; // Number of seconds in each dada file

    // Get project info from metafits
    fitsfile *fptr = NULL;
    int status     = 0;

    fits_open_file(&fptr, metafits, READONLY, &status);
    fits_read_key(fptr, TSTRING, "PROJECT", dhdr->proj_id, NULL, &status);
    fits_close_file(fptr, &status);

    // Fill in header values
    dhdr->hdr_size            = 4096;
    dhdr->populated           = 1;
    snprintf( dhdr->obsid,     17, "%s", obsid );
    snprintf( dhdr->subobs_id, 17, "%s", obsid );
    snprintf( dhdr->command,   17, "CAPTURE" );
    snprintf( dhdr->utc_start, 65, "%s", time_utc );
    dhdr->obs_offset          = 0;
    dhdr->nbit                = 32;
    dhdr->npol                = 2;
    dhdr->ntimesamples        = nsecs * sample_rate;
    dhdr->ninputs             = 1;
    dhdr->ninputs_xgpu        = 1;
    dhdr->metadata_beams      = 2;
    dhdr->apply_path_weights  = 1;
    dhdr->apply_path_delays   = 2;
    dhdr->int_time_msec       = 1000;
    dhdr->fscrunch_factor     = 1;
    dhdr->transfer_size       = 8*dhdr->ntimesamples;
    dhdr->exposure_secs       = 8;
    dhdr->coarse_channel      = atoi(rec_channel);
    dhdr->corr_coarse_channel = 1;
    dhdr->secs_per_subobs     = nsecs;
    dhdr->unixtime            = 0; // This should be the unix time equivalent of time_utc
    dhdr->unixtime_msec       = 0; // Always start at the beginning of a second
    dhdr->fine_chan_width_hz  = chan_width;
    dhdr->nfine_chan          = nchan;
    dhdr->bandwidth_hz        = nchan * chan_width;
    dhdr->sample_rate         = sample_rate;
    for (i = 0; i < 4; i++)  dhdr->cm_ip[i] = 0;
    dhdr->mc_port             = 0;
    for (i = 0; i < 4; i++)  dhdr->mc_src_ip[i] = 0;
    dhdr->file_size           = dhdr->ntimesamples*dhdr->nfine_chan + dhdr->hdr_size;
    dhdr->file_number         = 0; // This should be updated as needed

    dhdr->df       = NULL;
}


void dada_write_data( FILE *df, ComplexDouble ***detected_beam, int file_no,
                      int nsamples, int nchan, int npol )
/* Write out 1 second's worth of the post-beamformed, pre-PFB-inverted
 * voltages to file.
 *
 * Like cu_invert_pfb_ord() in ipfb.cu, it is expected that detected_beam
 * actually contains 2 seconds' worth of data, the file_no is used to
 * calculate where in detected_beam to start.
 */
{
    // Calculate the starting index for the first dimension of detected_beam.
    int start_s = (file_no % 2) * nsamples;

    // Write out the contents of detected_beam to file df
    int s, ch, pol;
    float re_im[2];
    for (s = start_s; s < nsamples; s++)
    {
        for (ch = 0; ch < nchan; ch++)
        {
            for (pol = 0; pol < npol; pol++)
            {
                re_im[0] = CRealf( detected_beam[s][ch][pol] );
                re_im[1] = CImagf( detected_beam[s][ch][pol] );
                fwrite( re_im, sizeof(float), 2, df );
            }
        }
    }
}


void dada_write_second( dada_header *dhdr, ComplexDouble ***detected_beam,
                        int file_no, int nsamples, int nchan, int npol )
/* Write out a second's worth of data in the DADA format.
 * This function takes care of file I/O, deciding when to start a new file
 * with a new header, etc., so it should be considered a wrapper function
 * for dada_write_header() and dada_write_data().
 */
{
    // Open the file for writing
    if (file_no % dhdr->secs_per_subobs == 0) /* if at the beginning of a (200 second) block */
    {
        // Start a new file
        dhdr->file_number = file_no / dhdr->secs_per_subobs;
        sprintf( dhdr->filename, "%s_ch%03d_%04d.dada", dhdr->obsid,
                dhdr->coarse_channel, dhdr->file_number );
        dhdr->df = fopen( dhdr->filename, "w" );

        // Write the header
        dada_write_header( dhdr->df, dhdr );
    }
    else /* append this second to the same file as the previous second */
    {
        dhdr->df = fopen( dhdr->filename, "a" );
    }

    // Write data
    dada_write_data( dhdr->df, detected_beam, file_no, nsamples, nchan, npol );

    // Close file
    fclose( dhdr->df );
}
