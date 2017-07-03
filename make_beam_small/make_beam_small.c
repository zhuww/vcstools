#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <complex.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include "mwac_utils.h"
#include "slalib.h"
#include "slamac.h"
#include "ascii_header.h"
#include "mwa_header.h"
#include <omp.h>
//#include <mpi.h>
#include <glob.h>
#include <fcntl.h>
#include <assert.h>
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

void usage() {
    fprintf(stderr, "make_beam -n <nchan> [128] -a <nant> \ntakes input from stdin and dumps to stdout|psrfits\n");
    fprintf(stderr, "-a <number of antennas>\n");
    fprintf(stderr, "-b Begin time [must be supplied]\n");
    fprintf(stderr, "-C <position of channel solution in Offringa calibration file\n");
    fprintf(stderr, "-d <data directory root> -- where the recombined data is\n");
    fprintf(stderr, "-D dd:mm:ss -- the declination to get passed to get_delays\n");
    fprintf(stderr, "-e End time [must be supplied]\n");
    fprintf(stderr, "-f <channel number>\n");
    fprintf(stderr, "-J <DI Jones file from the RTS> Jones matrix input\n");
    fprintf(stderr, "-m <metafits file> for this obsID\n");
    fprintf(stderr, "-n <number of channels>\n");
    fprintf(stderr, "-o obs id\n");
    fprintf(stderr, "-O <Offringa-style calibration solution file>\n");
    fprintf(stderr, "-r <sample rate in Hz>\n");
    fprintf(stderr, "-R hh:mm:ss -- the right ascension to get passed to get_delays\n");
    fprintf(stderr, "-S <bit mask> -- bit number 0 = swap pol, 1 == swap R and I, 2 conjugate sky\n");
    fprintf(stderr, "-V print version number and exit\n");
    fprintf(stderr, "-w use weights from metafits file [0]\n");
    fprintf(stderr, "-z <utc time string> yyyy-mm-ddThh:mm:ss\n");
    fprintf(stderr, "options: -t [1 or 2] sample size : 1 == 8 bit (INT); 2 == 32 bit (FLOAT)\n");

}

void populate_psrfits_header(
        struct psrfits *pf,
        char           *metafits,
        char           *obsid,
        char           *time_utc,
        unsigned int    sample_rate,
        long int        frequency,
        int             nchan,
        long int        chan_width,
        int             outpol,
        int             summed_polns,
        char           *rec_channel,
        struct delays  *delay_vals ) {

    fitsfile *fptr = NULL;
    int status      = 0;

    fits_open_file(&fptr, metafits, READONLY, &status);
    fits_read_key(fptr, TSTRING, "PROJECT", pf->hdr.project_id, NULL, &status);
    fits_close_file(fptr, &status);

    // Now set values for our hdrinfo structure
    strcpy(pf->hdr.obs_mode,  "SEARCH");
    strcpy(pf->hdr.observer,  "MWA User");
    strcpy(pf->hdr.telescope, "MWA");
    strncpy(pf->hdr.source, obsid, 23);
    pf->hdr.scanlen = 1.0; // in sec

    strcpy(pf->hdr.frontend, "MWA-RECVR");
    snprintf(pf->hdr.backend, 24*sizeof(char), "GD-%s-MB-%s-U-%s",
            GET_DELAYS_VERSION, MAKE_BEAM_VERSION, UTILS_VERSION);

    // Now let us finally get the time right
    strcpy(pf->hdr.date_obs,   time_utc);
    strcpy(pf->hdr.poln_type,  "LIN");
    strcpy(pf->hdr.track_mode, "TRACK");
    strcpy(pf->hdr.cal_mode,   "OFF");
    strcpy(pf->hdr.feed_mode,  "FA");

    pf->hdr.dt   = 1.0/sample_rate;                            // (sec)
    pf->hdr.fctr = (frequency + (nchan/2.0)*chan_width)/1.0e6; // (MHz)
    pf->hdr.BW   = (nchan*chan_width)/1.0e6;                   // (MHz)

    // npols + nbits and whether pols are added
    pf->filenum       = 0;       // This is the crucial one to set to initialize things
    pf->rows_per_file = 200;     // I assume this is a max subint issue

    pf->hdr.npol         = outpol;
    pf->hdr.summed_polns = summed_polns;
    pf->hdr.nchan        = nchan;
    pf->hdr.onlyI        = 0;

    pf->hdr.scan_number   = 1;
    pf->hdr.rcvr_polns    = 2;
    pf->hdr.summed_polns  = 0;
    pf->hdr.offset_subint = 0;

    pf->hdr.df         = chan_width/1.0e6; // (MHz)
    pf->hdr.orig_nchan = pf->hdr.nchan;
    pf->hdr.orig_df    = pf->hdr.df;
    pf->hdr.nbits      = 8;
    pf->hdr.nsblk      = sample_rate;  // block is always 1 second of data

    pf->hdr.ds_freq_fact = 1;
    pf->hdr.ds_time_fact = 1;

    // some things that we are unlikely to change
    pf->hdr.fd_hand  = 0;
    pf->hdr.fd_sang  = 0.0;
    pf->hdr.fd_xyph  = 0.0;
    pf->hdr.be_phase = 0.0;
    pf->hdr.chan_dm  = 0.0;

    // Now set values for our subint structure
    pf->tot_rows     = 0;
    pf->sub.tsubint  = roundf(pf->hdr.nsblk * pf->hdr.dt);
    pf->sub.offs     = roundf(pf->tot_rows * pf->sub.tsubint) + 0.5*pf->sub.tsubint;

    pf->sub.feed_ang = 0.0;
    pf->sub.pos_ang  = 0.0;
    pf->sub.par_ang  = 0.0;

    // Specify psrfits data type
    pf->sub.FITS_typecode = TBYTE;

    pf->sub.bytes_per_subint = (pf->hdr.nbits * pf->hdr.nchan *
                                pf->hdr.npol  * pf->hdr.nsblk) / 8;

    // Create and initialize the subint arrays
    pf->sub.dat_freqs   = (float *)malloc(sizeof(float) * pf->hdr.nchan);
    pf->sub.dat_weights = (float *)malloc(sizeof(float) * pf->hdr.nchan);

    double dtmp = pf->hdr.fctr - 0.5 * pf->hdr.BW + 0.5 * pf->hdr.df;
    int i;
    for (i = 0 ; i < pf->hdr.nchan ; i++) {
        pf->sub.dat_freqs[i] = dtmp + i * pf->hdr.df;
        pf->sub.dat_weights[i] = 1.0;
    }

    // the following is definitely needed for 8 bit numbers
    pf->sub.dat_offsets = (float *)malloc(sizeof(float) * pf->hdr.nchan * pf->hdr.npol);
    pf->sub.dat_scales  = (float *)malloc(sizeof(float) * pf->hdr.nchan * pf->hdr.npol);
    for (i = 0 ; i < pf->hdr.nchan * pf->hdr.npol ; i++) {
        pf->sub.dat_offsets[i] = 0.0;
        pf->sub.dat_scales[i]  = 1.0;
    }

    pf->sub.data    = (unsigned char *)malloc(pf->sub.bytes_per_subint);
    pf->sub.rawdata = pf->sub.data;

    int ch = atoi(rec_channel);
    sprintf(pf->basefilename, "%s_%s_ch%03d",
            pf->hdr.project_id, pf->hdr.source, ch);

    // Update values that depend on get_delays()
    if (delay_vals != NULL) {

        pf->hdr.ra2000  = delay_vals->mean_ra  * DR2D;
        pf->hdr.dec2000 = delay_vals->mean_dec * DR2D;

        dec2hms(pf->hdr.ra_str,  pf->hdr.ra2000/15.0, 0);
        dec2hms(pf->hdr.dec_str, pf->hdr.dec2000,     1);

        pf->hdr.azimuth    = delay_vals->az*DR2D;
        pf->hdr.zenith_ang = 90.0 - (delay_vals->el*DR2D);

        pf->hdr.beam_FWHM = 0.25;
        pf->hdr.start_lst = delay_vals->lmst * 60.0 * 60.0;        // Local Apparent Sidereal Time in seconds
        pf->hdr.start_sec = roundf(delay_vals->fracmjd*86400.0);   // this will always be a whole second
        pf->hdr.start_day = delay_vals->intmjd;
        pf->hdr.MJD_epoch  = delay_vals->intmjd + delay_vals->fracmjd;

        // Now set values for our subint structure
        pf->sub.lst      = pf->hdr.start_lst;
        pf->sub.ra       = pf->hdr.ra2000;
        pf->sub.dec      = pf->hdr.dec2000;
        slaEqgal(pf->hdr.ra2000*DD2R, pf->hdr.dec2000*DD2R,
                 &pf->sub.glon, &pf->sub.glat);
        pf->sub.glon    *= DR2D;
        pf->sub.glat    *= DR2D;
        pf->sub.tel_az   = pf->hdr.azimuth;
        pf->sub.tel_zen  = pf->hdr.zenith_ang;
    }
}


void get_metafits_info( char *metafits, struct metafits_info *mi ) {
/* Read in the relevant information from the metafits file.
 * This function allocates dynamic memory. Destroy it with
 *   destroy_metafits_info(...)
 */

    fitsfile *fptr = NULL;
    int status     = 0;
    int anynull    = 0;

    // Open the metafits file
    fits_open_file(&fptr, metafits, READONLY, &status);
    if (fptr == NULL) {
        fprintf( stderr, "Failed to open metafits file \"%s\"\n", metafits );
        exit(EXIT_FAILURE);
    }

    // Read in the tile pointing information (and channel width)
    fits_read_key(fptr, TDOUBLE, "RA",       &(mi->tile_pointing_ra),  NULL, &status);
    fits_read_key(fptr, TDOUBLE, "DEC",      &(mi->tile_pointing_dec), NULL, &status);
    fits_read_key(fptr, TDOUBLE, "AZIMUTH",  &(mi->tile_pointing_az),  NULL, &status);
    fits_read_key(fptr, TDOUBLE, "ALTITUDE", &(mi->tile_pointing_el),  NULL, &status);
    fits_read_key(fptr, TINT,    "FINECHAN", &(mi->chan_width),        NULL, &status);
    mi->chan_width *= 1000; // Convert from kHz to Hz

    if (status != 0) {
        fprintf(stderr, "Fits status set: failed to read az/alt, ");
        fprintf(stderr, "ra/dec fits keys from the header\n");
        exit(EXIT_FAILURE);
    }

    // Move to the binary table
    fits_movnam_hdu(fptr, BINARY_TBL, "TILEDATA", 0, &status);
    if (status != 0) {
        fprintf(stderr, "Error: Failed to move to TILEDATA HDU\n");
        exit(EXIT_FAILURE);
    }

    // Read in the number of inputs (= nstation * npol)
    fits_read_key(fptr, TINT, "NAXIS2", &(mi->ninput), NULL, &status);
    if (status != 0) {
        fprintf(stderr, "Error: Failed to read size of binary table in TILEDATA\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory
    mi->N_array         =     (float *)malloc( mi->ninput*sizeof(float)    );
    mi->E_array         =     (float *)malloc( mi->ninput*sizeof(float)    );
    mi->H_array         =     (float *)malloc( mi->ninput*sizeof(float)    );
    mi->cable_array     =     (float *)malloc( mi->ninput*sizeof(float)    );
    mi->flag_array      =       (int *)malloc( mi->ninput*sizeof(int)      );
    mi->weights_array   =    (double *)malloc( mi->ninput*sizeof(double)   );
    mi->antenna_num     = (short int *)malloc( mi->ninput*sizeof(short int));
    mi->tilenames       =     (char **)malloc( mi->ninput*sizeof(char *)   );
    int i;
    for (i = 0; i < (int)(mi->ninput); i++) {
        mi->tilenames[i] = (char *)malloc( 32*sizeof(char) );
    }
    char *testval = (char *) malloc(1024);

    /* Read the columns */
    int colnum;

    // Cable lengths
    for (i=0; i < (int)(mi->ninput); i++) {

        fits_get_colnum(fptr, 1, "Length", &colnum, &status);
        if(fits_read_col_str(fptr, colnum, i+1, 1, 1, "0.0", &testval, &anynull, &status)) {
            fprintf(stderr, "Error: Failed to cable column  in metafile\n");
            exit(EXIT_FAILURE);
        }

        sscanf(testval, "EL_%f", &(mi->cable_array[i]));
    }

    // Tile names
    fits_get_colnum(fptr, 1, "TileName", &colnum, &status);
    if (status != 0) {
        status = 0;
        fits_get_colnum(fptr, 1, "Tile", &colnum, &status);
    }
    if (status != 0) {
        fprintf(stderr, "Could not find either column \"TileName\" or \"Tile\" in metafits file\n");
        exit(EXIT_FAILURE);
    }

    fits_read_col(fptr, TSTRING, colnum, 1, 1, mi->ninput, NULL, mi->tilenames, &anynull, &status);
    if (status != 0){
        fprintf(stderr, "Error: Failed to read Tile(Name) in metafile\n");
        exit(EXIT_FAILURE);
    }

    // North coordinate
    fits_get_colnum(fptr, 1, "North", &colnum, &status);
    fits_read_col_flt(fptr, colnum, 1, 1, mi->ninput, 0.0, mi->N_array, &anynull, &status);
    if (status != 0){
        fprintf(stderr, "Error: Failed to read  N coord in metafile\n");
        exit(EXIT_FAILURE);
    }

    // East coordinate
    fits_get_colnum(fptr, 1, "East", &colnum, &status);
    fits_read_col_flt(fptr, colnum, 1, 1, mi->ninput, 0.0, mi->E_array, &anynull, &status);
    if (status != 0){
        fprintf(stderr, "Error: Failed to read E coord in metafile\n");
        exit(EXIT_FAILURE);
    }

    // Height coordinate
    fits_get_colnum(fptr, 1, "Height", &colnum, &status);
    fits_read_col_flt(fptr, colnum, 1, 1, mi->ninput, 0.0, mi->H_array, &anynull, &status);

    if (status != 0){
        fprintf(stderr, "Error: Failed to read H coord in metafile\n");
        exit(EXIT_FAILURE);
    }

    // Antenna number
    fits_get_colnum(fptr, 1, "Antenna", &colnum, &status);
    fits_read_col_sht(fptr, colnum, 1, 1, mi->ninput, 0.0, mi->antenna_num, &anynull, &status);

    if (status != 0){
        fprintf(stderr, "Error: Failed to read field \"Antenna\" in metafile\n");
        exit(EXIT_FAILURE);
    }

    // Flags & weights
    float wgt_sum = 0.0;
    fits_get_colnum(fptr, 1, "Flag", &colnum, &status);
    fits_read_col_int(fptr, colnum, 1, 1, mi->ninput, 0.0, mi->flag_array, &anynull, &status);
    if (status != 0){
        fprintf(stderr, "Error: Failed to read flags column in metafile\n");
        exit(EXIT_FAILURE);
    }

    // Invert value (flag off = full weight; flag on = zero weight)
    for (i = 0; i < mi->ninput; i++) {
        mi->weights_array[i] = 1.0 - (double)mi->flag_array[i];
        wgt_sum += mi->weights_array[i];
        // This differs from Ord's orig code, which sums squares. However,
        // all values should be = 1, so end result should be the same
    }

    // Exit with error if there are no weights
    if (wgt_sum == 0.0) {
        fprintf(stderr, "Zero weight sum on read\n");
        exit(EXIT_FAILURE);
    }

    // Clean up
    free( testval );
    fits_close_file(fptr, &status);
}


void destroy_metafits_info( struct metafits_info *mi ) {
/* Frees the memory allocated in the metafits_info struct
 */
    free( mi->N_array       );
    free( mi->E_array       );
    free( mi->H_array       );
    free( mi->cable_array   );
    free( mi->flag_array    );
    free( mi->weights_array );
    free( mi->antenna_num   );
    int i;
    for (i = 0; i < mi->ninput; i++)
        free( mi->tilenames[i] );
    free( mi->tilenames     );
}


void int8_to_uint8(int n, int shift, char * to_convert) {
    int j;
    int scratch;
    int8_t with_sign;

    for (j = 0; j < n; j++) {
        with_sign = (int8_t) *to_convert;
        scratch = with_sign + shift;
        *to_convert = (uint8_t) scratch;
        to_convert++;
    }
}
void float2int8_trunc(float *f, int n, float min, float max, int8_t *i) /*includefile*/
{
    int j;
    for (j = 0; j < n; j++) {
        f[j] = (f[j] > max) ? (max) : f[j];
        f[j] = (f[j] < min) ? (min) : f[j];
        i[j] = (int8_t) rint(f[j]);

    }
}
void to_offset_binary(int8_t *i, int n){
    int j;
    for (j = 0; j < n; j++) {
        i[j] = i[j] ^ 0x80;
    }
}
complex float get_std_dev_complex(complex float *input, int nsamples) {
    // assume zero mean
    float rtotal = 0;
    float itotal = 0;
    float isigma = 0;
    float rsigma = 0;
    int i;

    for (i=0;i<nsamples;i++){
         rtotal = rtotal+(crealf(input[i])*crealf(input[i]));
         itotal = itotal+(cimagf(input[i])*cimagf(input[i]));

     }
    rsigma = sqrtf((1.0/(nsamples-1))*rtotal);
    isigma = sqrtf((1.0/(nsamples-1))*itotal);

    return rsigma+I*isigma;
}
void set_level_occupancy(complex float *input, int nsamples, float *new_gain) {

    float percentage = 0.0;
    float limit = 0.00001;
    float step = 0.001;
    int i = 0;
    float gain = *new_gain;

    float percentage_clipped = 100;
    while (percentage_clipped > 0 && percentage_clipped > limit) {
        int count = 0;
        int clipped = 0;
        for (i=0;i<nsamples;i++) {
            if (gain*creal(input[i]) >= 0 && gain*creal(input[i]) < 64) {
                count++;
            }
            if (fabs(gain*creal(input[i])) > 127) {
                clipped++;
            }
        }
        percentage_clipped = ((float) clipped/nsamples) * 100;
        if (percentage_clipped < limit) {
            gain = gain + step;
        }
        else {
            gain = gain - step;
        }
        percentage = ((float)count/nsamples)*100.0;
        fprintf(stdout, "Gain set to %f (linear)\n", gain);
        fprintf(stdout, "percentage of samples in the first 64 (+ve) levels - %f percent \n", percentage);
        fprintf(stdout, "percentage clipped %f percent\n", percentage_clipped);
    }
    *new_gain = gain;
}

void get_mean_complex(complex float *input, int nsamples, float *rmean, float *imean, complex float *cmean) {

    int i=0;
    float rtotal = 0;
    float itotal = 0 ;
    complex float ctotal = 0 + I*0.0;
    for (i=0;i<nsamples;i++){
        rtotal = rtotal+crealf(input[i]);
        itotal = itotal+cimagf(input[i]);
        ctotal = ctotal + input[i];
    }
    *rmean=rtotal/nsamples;
    *imean=itotal/nsamples;
    *cmean=ctotal/nsamples;

}
void normalise_complex(complex float *input, int nsamples, float scale) {

    int i=0;

    for (i=0;i<nsamples;i++){
        input[i]=input[i]*scale;
    }

}

void flatten_bandpass(int nstep, int nchan, int npol, void *data, float *scales, float *offsets, int new_var, int iscomplex, int normalise, int update, int clear, int shutdown) {
    // putpose is to generate a mean value for each channel/polaridation

    int i=0, j=0;
    int p=0;
    float *data_ptr = NULL;

    static float **band;

    static float **chan_min;

    static float **chan_max;


    static int setup = 0;

    if (setup == 0) {
        band = (float **) calloc (npol, sizeof(float *));
        chan_min = (float **) calloc (npol, sizeof(float *));
        chan_max = (float **) calloc (npol, sizeof(float *));
        for (i=0;i<npol;i++) {
            band[i] = (float *) calloc(nchan, sizeof(float));
            chan_min[i] = (float *) calloc(nchan, sizeof(float));
            chan_max[i] = (float *) calloc(nchan, sizeof(float));
        }
        setup = 1;
    }

    if (update) {
        for (p = 0;p<npol;p++) {
            for (j=0;j<nchan;j++){

                band[p][j] = 0.0;
            }
        }

        if (iscomplex == 0) {
            data_ptr = (float *) data;

            for (i=0;i<nstep;i++) {
                for (p = 0;p<npol;p++) {
                    for (j=0;j<nchan;j++){


                        if (i==0) {
                            chan_min[p][j] = *data_ptr;
                            chan_max[p][j] = *data_ptr;
                        }
                        band[p][j] += fabsf(*data_ptr);
                        if (*data_ptr < chan_min[p][j]) {
                            chan_min[p][j] = *data_ptr;
                        }
                        else if (*data_ptr > chan_max[p][j]) {
                            chan_max[p][j] = *data_ptr;
                        }
                        data_ptr++;
                    }
                }

            }
        }
        else {
            complex float  *data_ptr = (complex float *) data;
            for (i=0;i<nstep;i++) {
                for (p = 0;p<npol;p++) {
                    for (j=0;j<nchan;j++){

                        band[p][j] += cabsf(*data_ptr);
                        data_ptr++;
                    }
                }

            }

        }

    }
    // set the offsets and scales - even if we are not updating ....

    float *out=scales;
    float *off = offsets;
    for (p = 0;p<npol;p++) {
        for (j=0;j<nchan;j++){

            // current mean
            *out = ((band[p][j]/nstep))/new_var; // removed a divide by 32 here ....
            //fprintf(stderr, "Channel %d pol %d mean: %f normaliser %f (max-min) %f\n", j, p, (band[p][j]/nstep), *out, (chan_max[p][j] - chan_min[p][j]));
            out++;
            *off = 0.0;

            off++;

        }
    }
    // apply them to the data

    if (normalise) {

        data_ptr = (float *) data;

        for (i=0;i<nstep;i++) {
            float *normaliser = scales;
            float *off  = offsets;
            for (p = 0;p<npol;p++) {
                for (j=0;j<nchan;j++){

                    *data_ptr = ((*data_ptr) - (*off))/(*normaliser); // 0 mean normalised to 1
                    //fprintf(stderr, "%f %f %f\n", *data_ptr, *off, *normaliser);
                    off++;
                    data_ptr++;
                    normaliser++;
                }
            }

        }
    }

    // clear the weights if required

    if (clear) {

        float *out=scales;
        float *off = offsets;
        for (p = 0;p<npol;p++) {
            for (j=0;j<nchan;j++){

                // reset
                *out = 1.0;



                out++;
                *off = 0.0;

                off++;

            }
        }
    }

    // free the memory
    if (shutdown) {
        for (i=0;i<npol;i++) {
            free(band[i]);
            free(chan_min[i]);
            free(chan_max[i]);
        }


        free(band);
        free(chan_min);
        free(chan_max);
        setup = 0;
    }
}



int read_pfb_call(char *in_name, char *heap) {


    int retval = 1;

    int fd_in = open(in_name, O_RDONLY);

    if (fd_in < 0) {
        fprintf(stderr, "Failed to open %s:%s\n", in_name, strerror(errno));
        exit(EXIT_FAILURE);
    }

    if ((default_read_pfb_call(fd_in, 0, heap)) < 0){
        fprintf(stderr, "Error in default_read_pfb\n");
        retval = -1;
    }

    close(fd_in);
    return retval;

}


/*****************
 * MAIN FUNCTION *
 *****************/

int main(int argc, char **argv) {

    double begintime = omp_get_wtime();
    printf("[%f]  Starting make_beam\n", omp_get_wtime()-begintime);

    int c  = 0;
    int ch = 0;

    unsigned long int begin = 0;
    unsigned long int end   = 0;

    int weights = 0;
    char *rec_channel = NULL; // 0 - 255 receiver 1.28MHz channel

    char *obsid = NULL;
    char *datadirroot = NULL;
    char **filenames = NULL;
    int nfiles = 0;

    unsigned int sample_rate = 10000;

    int nchan = 128;

    nfrequency = nchan;
    nstation = 128;
    npol = 2;
    int outpol = 4;
    int summed_polns = 0;

    char *dec_ddmmss    = NULL; // "dd:mm:ss"
    char *ra_hhmmss     = NULL; // "hh:mm:ss"
    char *metafits      = NULL; // filename of the metafits file
    char *time_utc      = NULL; // utc time string "yyyy-mm-ddThh:mm:ss"

    long int frequency  = 0;

    struct calibration cal;

    if (argc > 1) {

        while ((c = getopt(argc, argv, "a:b:C:d:D:e:f:hJ:m:n:o:O:r:R:VwW:z:")) != -1) {
            switch(c) {

                case 'a':
                    nstation = atoi(optarg);
                    break;
                case 'b':
                    begin = atol(optarg);
                    break;
                case 'C':
                    cal.offr_chan_num = atoi(optarg);
                    break;
                case 'd':
                    datadirroot = strdup(optarg);
                    break;
                case 'D':
                    dec_ddmmss = strdup(optarg);
                    break;
                case 'e':
                    end = atol(optarg);
                    break;
                case 'f':
                    rec_channel = strdup(optarg);
                    frequency = atoi(optarg) * 1.28e6 - 640e3; // The base frequency in Hz
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'J':
                    cal.filename = strdup(optarg);
                    cal.cal_type = RTS;
                    break;
                case 'm':
                    metafits = strdup(optarg);
                    break;
                case 'n':
                    nchan = atoi(optarg);
                    break;
                case 'o':
                    obsid = strdup(optarg);
                    break;
                case 'O':
                    cal.filename = strdup(optarg);
                    cal.cal_type = OFFRINGA;
                    break;
                case 'r':
                    sample_rate = atoi(optarg);
                    break;
                case 'R':
                    ra_hhmmss = strdup(optarg);
                    break;
                case 'V':
                    printf("%s\n", MAKE_BEAM_VERSION);
                    exit(0);
                    break;
                case 'w':
                    weights = 1;
                    break;
                case 'z':
                    time_utc = strdup(optarg);
                    break;
                default:
                    usage();
                    exit(EXIT_FAILURE);
            }
        }
    }

    if (datadirroot) {

        // Generate list of files to work on

        // Calculate the number of files
        nfiles = end - begin + 1;
        if (nfiles <= 0) {
            fprintf(stderr, "Cannot beamform on %d files (between %lu and %lu)\n", nfiles, begin, end);
            exit(EXIT_FAILURE);
        }

        // Allocate memory for the file name list
        filenames = (char **)malloc( nfiles*sizeof(char *) );

        // Allocate memory and write filenames
        int second;
        unsigned long int timestamp;
        for (second = 0; second < nfiles; second++) {
            timestamp = second + begin;
            filenames[second] = (char *)malloc( MAX_COMMAND_LENGTH*sizeof(char) );
            sprintf( filenames[second], "%s/%s_%ld_ch%s.dat", datadirroot, obsid, timestamp, rec_channel );
        }

    }

    complex double **complex_weights_array = NULL;
    complex double **invJi = NULL;

    // Allocate memory for complex weights and jones matrices
    int i;
    complex_weights_array = (complex double **)malloc( nstation * npol * sizeof(complex double *) );
    for (i = 0; i < nstation*npol; i++)
        complex_weights_array[i] = (complex double *)malloc( nchan * sizeof(complex double) );

    invJi = (complex double **)malloc( nstation * sizeof(complex double *) );
    for (i = 0; i < nstation; i++)
        invJi[i] =(complex double *)malloc( npol * npol * sizeof(complex double) );

    // these are only used if we are prepending the fitsheader
    struct psrfits pf;


    // Read in info from metafits file
    printf("[%f]  Reading in metafits file information from %s\n", omp_get_wtime()-begintime, metafits);
    struct metafits_info mi;
    get_metafits_info( metafits, &mi );

    if (!weights)
        for (i = 0; i < nstation*npol; i++)
            mi.weights_array[i] = 1.0;

    double wgt_sum = 0;
    for (i = 0; i < nstation*npol; i++)
        wgt_sum += mi.weights_array[i];
    double invw = 1.0/wgt_sum;

    // Get first second's worth of phases and Jones matrices
    printf("[%f]  Setting up output header information\n", omp_get_wtime()-begintime);
    struct delays delay_vals;
    get_delays(
            dec_ddmmss,    // dec as a string "dd:mm:ss"
            ra_hhmmss,     // ra  as a string "hh:mm:ss"
            frequency,     // middle of the first frequency channel in Hz
            &cal,          // struct holding info about calibration
            sample_rate,   // = 10000 samples per sec
            time_utc,      // utc time string
            0.0,           // seconds offset from time_utc at which to calculate delays
            &delay_vals,   // Populate psrfits header info
            &mi,           // Struct containing info from metafits file
            complex_weights_array,  // complex weights array (answer will be output here)
            invJi          // invJi array           (answer will be output here)
    );

    // now we need to create a fits file and populate its header
    populate_psrfits_header( &pf, metafits, obsid, time_utc, sample_rate,
            frequency, nchan, mi.chan_width, outpol, summed_polns,
            rec_channel, &delay_vals );

    unsigned int nspec = 1;
    size_t items_to_read = nstation*npol*nchan*2;

    char *heap = NULL;

    heap = (char *) malloc(nspec*items_to_read*sample_rate);

    assert(heap);

    int8_t *out_buffer_8_psrfits = (int8_t *)malloc( outpol*nchan*pf.hdr.nsblk * sizeof(int8_t) );
    float  *data_buffer_psrfits  =  (float *)malloc( nchan*outpol*pf.hdr.nsblk * sizeof(float) );

    int index = 0;
    int offset_in_psrfits;

    int file_no = 0;
    int sample;

    printf("[%f]  **BEGINNING BEAMFORMING**\n", omp_get_wtime()-begintime);
    for (file_no = 0; file_no < nfiles; file_no++) {

        printf("[%f]  Reading in data from %s [%d/%d]\n", omp_get_wtime()-begintime,
                filenames[file_no], file_no+1, nfiles);
        if ((read_pfb_call(filenames[file_no], heap)) < 0)
            break; // Exit from while loop

        // Get the next second's worth of phases / jones matrices, if needed
        printf("[%f]  Calculating delays\n", omp_get_wtime()-begintime);
        get_delays(
                dec_ddmmss,    // dec as a string "dd:mm:ss"
                ra_hhmmss,     // ra  as a string "hh:mm:ss"
                frequency,     // middle of the first frequency channel in Hz
                &cal,          // struct holding info about calibration
                sample_rate,   // = 10000 samples per sec
                time_utc,      // utc time string
                (double)file_no, // seconds offset from time_utc at which to calculate delays
                NULL,          // Don't update delay_vals
                &mi,           // Struct containing info from metafits file
                complex_weights_array,  // complex weights array (answer will be output here)
                invJi );       // invJi array           (answer will be output here)

        printf("[%f]  Calculating beam\n", omp_get_wtime()-begintime);

        offset_in_psrfits  = 0;

        for (i = 0; i < nchan*outpol*pf.hdr.nsblk; i++)
            data_buffer_psrfits[i] = 0.0;

#pragma omp parallel for private(ch)
        for (sample = 0; sample < (int)sample_rate; sample++ ) {

            complex float beam[nchan][nstation*npol];
            float spectrum[nspec*nchan*outpol];
            float noise_floor[nchan*npol*npol];
            char buffer[nspec*items_to_read];

            for (i = 0; i < nchan*npol*npol; i++)
                noise_floor[i] = 0.0;

            memcpy(buffer, heap+(items_to_read*sample), items_to_read);

            for (index = 0; index < nstation*npol;index = index + 2) {

                for (ch=0;ch<nchan;ch++) {
                    int8_t *in_ptr = (int8_t *)buffer + 2*index*nchan + 2*ch;

                    complex float e_true[2], e_dash[2];

                    e_dash[0] = (float) *in_ptr             + I*(float)(*(in_ptr+          1));
                    e_dash[1] = (float) *(in_ptr+(nchan*2)) + I*(float)(*(in_ptr+(nchan*2)+1)); // next pol is nchan*2 away

                    /* apply the inv(jones) to the e_dash */
                    e_dash[0] *= complex_weights_array[index][ch];
                    e_dash[1] *= complex_weights_array[index+1][ch];

                    e_true[0] = invJi[index/npol][0]*e_dash[0] + invJi[index/npol][1]*e_dash[1];
                    e_true[1] = invJi[index/npol][2]*e_dash[0] + invJi[index/npol][3]*e_dash[1];

                    noise_floor[ch*npol*npol]   += e_true[0] * conj(e_true[0]);
                    noise_floor[ch*npol*npol+1] += e_true[0] * conj(e_true[1]);
                    noise_floor[ch*npol*npol+2] += e_true[1] * conj(e_true[0]);
                    noise_floor[ch*npol*npol+3] += e_true[1] * conj(e_true[1]);

                    beam[ch][index]   = e_true[0];
                    beam[ch][index+1] = e_true[1];
                }
            }

            // detect the beam or prep from invert_pfb
            // reduce over each channel for the beam
            // do this by twos
            int polnum = 0;
            int step = 0;
            for (ch = 0; ch < nchan; ch++) {
                for (polnum = 0; polnum < npol; polnum++) {
                    int next_good = 2;
                    int stride = 4;

                    while (next_good < nstation*npol) {
                        for (step = polnum; step < nstation*npol; step += stride) {
                            beam[ch][step] += beam[ch][step+next_good];
                        }
                        stride    *= 2;
                        next_good *= 2;
                    }
                }
            }

            // Calculate the Stokes parameters
            double beam00, beam11;
            double noise0, noise1, noise3;
            complex double beam01;
            unsigned int stokesIidx, stokesQidx, stokesUidx, stokesVidx;
            for (ch = 0; ch < nchan; ch++) {

                beam00 = (double)(beam[ch][0] * conj(beam[ch][0]));
                beam11 = (double)(beam[ch][1] * conj(beam[ch][1]));
                beam01 = beam[ch][0] * conj(beam[ch][1]);

                noise0 = noise_floor[ch*npol*npol];
                noise1 = noise_floor[ch*npol*npol+1];
                noise3 = noise_floor[ch*npol*npol+3];

                stokesIidx = 0*nchan + ch;
                stokesQidx = 1*nchan + ch;
                stokesUidx = 2*nchan + ch;
                stokesVidx = 3*nchan + ch;

                // Looking at the dspsr loader the expected order is <ntime><npol><nchan>
                // so for a single timestep we do not have to interleave - I could just stack these
                spectrum[stokesIidx]  = (beam00 + beam11 - noise0 - noise3) * invw;
                spectrum[stokesQidx]  = (beam00 - beam11 - noise0 - noise3) * invw;
                spectrum[stokesUidx] = 2.0 * (creal(beam01) - noise1)*invw;
                spectrum[stokesVidx] = -2.0 * cimag((beam01 - noise1)*invw);
            }

            offset_in_psrfits  = sizeof(float)*nchan*outpol * sample;

            memcpy((void *)((char *)data_buffer_psrfits + offset_in_psrfits), spectrum, sizeof(float)*nchan*outpol);

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

        printf("[%f]  Writing data to file\n", omp_get_wtime()-begintime);
        if (psrfits_write_subint(&pf) != 0) {
            fprintf(stderr, "Write subint failed file exists?\n");
            break; // Exit from while loop
        }

        pf.sub.offs = roundf(pf.tot_rows * pf.sub.tsubint) + 0.5*pf.sub.tsubint;
        pf.sub.lst += pf.sub.tsubint;

    }

    printf("[%f]  **FINISHED BEAMFORMING**\n", omp_get_wtime()-begintime);
    printf("[%f]  Starting clean-up\n", omp_get_wtime()-begintime);

    if (pf.status == 0) {
        /* now we have to correct the STT_SMJD/STT_OFFS as they will have been broken by the write_psrfits*/
        int itmp = 0;
        int itmp2 = 0;
        double dtmp = 0;
        int status = 0;

        //fits_open_file(&(pf.fptr), pf.filename, READWRITE, &status);

        fits_read_key(pf.fptr, TDOUBLE, "STT_OFFS", &dtmp, NULL, &status);
        fits_read_key(pf.fptr, TINT, "STT_SMJD", &itmp, NULL, &status);
        fits_read_key(pf.fptr, TINT, "STT_IMJD", &itmp2, NULL, &status);

        if (dtmp > 0.5) {
            itmp = itmp+1;
            if (itmp == 86400) {
                itmp = 0;
                itmp2++;
            }
        }
        dtmp = 0.0;

        fits_update_key(pf.fptr, TINT, "STT_SMJD", &itmp, NULL, &status);
        fits_update_key(pf.fptr, TINT, "STT_IMJD", &itmp2, NULL, &status);
        fits_update_key(pf.fptr, TDOUBLE, "STT_OFFS", &dtmp, NULL, &status);

        //fits_close_file(pf.fptr, &status);
        fprintf(stdout, "[%f]  Done.  Wrote %d subints (%f sec) in %d files.\n",
               omp_get_wtime()-begintime, pf.tot_rows, pf.T, pf.filenum);

        // free some memory
        flatten_bandpass(pf.hdr.nsblk, nchan, outpol, data_buffer_psrfits, pf.sub.dat_scales, pf.sub.dat_offsets, 32, 0, 0, 0, 0, 1);

    }

    // Free up memory for filenames
    if (datadirroot) {
        int second;
        for (second = 0; second < nfiles; second++)
            free( filenames[second] );
        free( filenames );
    }

    destroy_metafits_info( &mi );
    free( out_buffer_8_psrfits );
    free( data_buffer_psrfits  );

    return 0;
}
