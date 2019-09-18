/********************************************************
 *                                                      *
 * Licensed under the Academic Free License version 3.0 *
 *                                                      *
 ********************************************************/

#ifndef BEAM_DADA_H
#define BEAM_DADA_H

#include <fftw3.h>
#include "beam_common.h"
//#include "dadaio.h"
#include "filter.h"
#include "mycomplex.h"

#define  DADA_HEADER_SIZE  4096

/* convenience type - this just collects all the dada info together */
typedef struct dada_header_t {

    // Values that get put in the header
    int hdr_size;
    int populated;
    char obsid[17];
    char subobs_id[17];
    char command[17];
    char utc_start[65];
    int obs_offset;
    int nbit;
    int npol;
    int ntimesamples;
    int ninputs;
    int ninputs_xgpu;
    int metadata_beams;
    int apply_path_weights;
    int apply_path_delays;
    int int_time_msec;
    int fscrunch_factor;
    int transfer_size;
    char proj_id[17];
    int exposure_secs;
    int coarse_channel;
    int corr_coarse_channel;
    int secs_per_subobs;
    int unixtime;
    int unixtime_msec;
    int fine_chan_width_hz;
    int nfine_chan;
    int bandwidth_hz;
    int sample_rate;
    int cm_ip[4];
    int mc_port;
    int mc_src_ip[4];
    size_t file_size;
    int file_number;

    // Values that are used for housekeeping
    FILE *df;
    char filename[257];

} dada_header;

void dada_write_header( FILE *df, dada_header *dhdr );
void dada_write_data( FILE *df, ComplexDouble ***detected_beam, int file_no,
                      int nsamples, int nchan, int npol );
void dada_write_second( dada_header *dhdr, ComplexDouble ***detected_beam,
                        int file_no, int nsamples, int nchan, int npol );

void populate_dada_header(
        dada_header     *dhdr,
        char            *metafits,
        char            *obsid,
        char            *time_utc,
        int              sample_rate,
        int              nchan, 
        long int         chan_width,
        char            *rec_channel );


#endif
