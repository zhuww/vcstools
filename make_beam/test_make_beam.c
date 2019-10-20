/********************************************************
 *                                                      *
 * Licensed under the Academic Free License version 3.0 *
 *                                                      *
 ********************************************************/

// TODO: Remove superfluous #includes
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include "slalib.h"
#include "slamac.h"
#include "ascii_header.h"
#include "mwa_header.h"
#include <glob.h>
#include <fcntl.h>
#include <assert.h>
#include "beam_common.h"
#include "beam_psrfits.h"
#include "beam_vdif.h"
#include "make_beam.h"
#include "vdifio.h"
#include "filter.h"
#include "psrfits.h"
#include "mycomplex.h"
#include "form_beam.h"
#include <omp.h>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include "ipfb.h"
#define NOW  ((double)clock()/(double)CLOCKS_PER_SEC)

#else

#define NOW  (omp_get_wtime())

#endif

int main(int argc, char **argv)
{
    #ifndef HAVE_CUDA
    // Initialise FFTW with OpenMP
    fftw_init_threads();
    fftw_plan_with_nthreads( omp_get_max_threads() );
    #endif

    ComplexDouble ****complex_weights_array = create_complex_weights( 2, 128, 128, 2 );
    // A place to hold the beamformer settings
    struct make_beam_opts opts;

    /* Set default beamformer settings */

    // Variables for required options
    opts.obsid = (char *) malloc(64);
    strcpy(opts.obsid, "1166459712"); // The observation ID
    opts.begin       = 1166460000;    // GPS time -- when to start beamforming
    opts.end         = 1166460000;    // GPS time -- when to stop beamforming
    opts.time_utc = (char *) malloc(64);
    strcpy(opts.time_utc, "2016-12-22T16:39:43"); // utc time string "yyyy-mm-ddThh:mm:ss"
    opts.pointings = (char *) malloc(64);
    strcpy(opts.pointings, "07:42:49.00_-28:21:43.00,07:42:49.00_-28:21:43.01");
                       // list of pointings "dd:mm:ss_hh:mm:ss,dd:mm:ss_hh:mm:ss"
    //opts.datadir     = NULL; // The path to where the recombined data live
    //opts.metafits    = NULL; // filename of the metafits file
    opts.rec_channel = (char *) malloc(64);
    strcpy(opts.rec_channel, "132"); // 0 - 255 receiver 1.28MHz channel
    opts.frequency   = 132 * 1.28e6 - 640e3;    // = rec_channel expressed in Hz

    // Variables for MWA/VCS configuration
    opts.nstation      = 128;    // The number of antennas
    opts.nchan         = 128;    // The number of fine channels (per coarse channel)
    opts.chan_width    = 10000;  // The bandwidth of an individual fine chanel (Hz)
    opts.sample_rate   = 10000;  // The VCS sample rate (Hz)
    opts.custom_flags  = NULL;   // Use custom list for flagging antennas

    // Output options
    opts.out_incoh     = 1;  // Default = PSRFITS (incoherent) output turned OFF
    opts.out_coh       = 1;  // Default = PSRFITS (coherent)   output turned OFF
    opts.out_vdif      = 1;  // Default = VDIF                 output turned OFF
    opts.out_summed    = 0;  // Default = output only Stokes I output turned OFF
    opts.write         = 0;  // Default = write new text files output turned OFF 


    // Variables for calibration settings
    //opts.cal.filename          = NULL;
    opts.cal.bandpass_filename = NULL;
    opts.cal.chan_width        = 40000;
    opts.cal.nchan             = 0;
    opts.cal.cal_type          = NO_CALIBRATION;
    opts.cal.offr_chan_num     = 0;

    // Parse command line arguments
    fprintf( stderr, "Parsing command line\n");
    make_beam_parse_cmdline( argc, argv, &opts );
    fprintf( stderr, "Parsed\n");
    

    // Create "shorthand" variables for options that are used frequently
    int nstation             = 128;
    int nchan                = 128;
    int npol                 = 2;   // (X,Y)
    int outpol_coh           = 4;  // (I,Q,U,V)
    if ( opts.out_summed )
        outpol_coh           = 1;  // (I)
    const int outpol_incoh   = 1;  // ("I")

    float vgain = 1.0; // This is re-calculated every second for the VDIF output

    // Set up test stuff
    fprintf( stderr, "%s\n", opts.datadir);
    opts.metafits = (char *) malloc(64);
    strcpy(opts.metafits, opts.datadir);
    fprintf( stderr, "inbetween\n");
    strcat(opts.metafits, "/1166459712_metafits_ppds.fits");
    fprintf( stderr, "Parsed\n");
    opts.cal.filename = (char *) malloc(64);
    strcpy(opts.cal.filename, opts.datadir);
    strcat(opts.cal.filename, "/DI_JonesMatrices_node021.dat");
    
    fprintf( stderr, "%s\n", opts.metafits);
    fprintf( stderr, "%s\n", opts.cal.filename);
    

    // Start counting time from here (i.e. after parsing the command line)
    double begintime = NOW;
    fprintf( stderr, "[%f]  Starting %s with GPU acceleration\n", NOW-begintime, argv[0] );

    // Calculate the number of files
    int nfiles = opts.end - opts.begin + 1;
    if (nfiles <= 0) {
        fprintf(stderr, "Cannot beamform on %d files (between %lu and %lu)\n", nfiles, opts.begin, opts.end);
        exit(EXIT_FAILURE);
    }

    // Parse input pointings
    int max_npointing = 120; // Could be more
    char RAs[max_npointing][64];
    char DECs[max_npointing][64];
    int npointing = sscanf( opts.pointings, 
            "%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,],%[^_]_%[^,]," , 
                            RAs[0],  DECs[0],  RAs[1],  DECs[1],  RAs[2],  DECs[2],
                            RAs[3],  DECs[3],  RAs[4],  DECs[4],  RAs[5],  DECs[5],
                            RAs[6],  DECs[6],  RAs[7],  DECs[7],  RAs[8],  DECs[8],
                            RAs[9],  DECs[9],  RAs[10], DECs[10], RAs[11], DECs[11],
                            RAs[12], DECs[12], RAs[13], DECs[13], RAs[14], DECs[14],
                            RAs[15], DECs[15], RAs[16], DECs[16], RAs[17], DECs[17],
                            RAs[18], DECs[18], RAs[19], DECs[19], RAs[20], DECs[20],
                            RAs[21], DECs[21], RAs[22], DECs[22], RAs[23], DECs[23],
                            RAs[24], DECs[24], RAs[25], DECs[25], RAs[26], DECs[26],
                            RAs[27], DECs[27], RAs[28], DECs[28], RAs[29], DECs[29],
                            RAs[30], DECs[30], RAs[31], DECs[31], RAs[32], DECs[32],
                            RAs[33], DECs[33], RAs[34], DECs[34], RAs[35], DECs[35],
                            RAs[36], DECs[36], RAs[37], DECs[37], RAs[38], DECs[38],
                            RAs[39], DECs[39], RAs[40], DECs[40], RAs[41], DECs[41],
                            RAs[42], DECs[42], RAs[43], DECs[43], RAs[44], DECs[44],
                            RAs[45], DECs[45], RAs[46], DECs[46], RAs[47], DECs[47],
                            RAs[48], DECs[48], RAs[49], DECs[49], RAs[50], DECs[50],
                            RAs[51], DECs[51], RAs[52], DECs[52], RAs[53], DECs[53],
                            RAs[54], DECs[54], RAs[55], DECs[55], RAs[56], DECs[56],
                            RAs[57], DECs[57], RAs[58], DECs[58], RAs[59], DECs[59],
                            RAs[60], DECs[60], RAs[61], DECs[61], RAs[62], DECs[62],
                            RAs[63], DECs[63], RAs[64], DECs[64], RAs[65], DECs[65],
                            RAs[66], DECs[66], RAs[67], DECs[67], RAs[68], DECs[68],
                            RAs[69], DECs[69], RAs[70], DECs[70], RAs[71], DECs[71],
                            RAs[72], DECs[72], RAs[73], DECs[73], RAs[74], DECs[74],
                            RAs[75], DECs[75], RAs[76], DECs[76], RAs[77], DECs[77],
                            RAs[78], DECs[78], RAs[79], DECs[79], RAs[80], DECs[80],
                            RAs[81], DECs[81], RAs[82], DECs[82], RAs[83], DECs[83],
                            RAs[84], DECs[84], RAs[85], DECs[85], RAs[86], DECs[86],
                            RAs[87], DECs[87], RAs[88], DECs[88], RAs[89], DECs[89],
                            RAs[90], DECs[90], RAs[91], DECs[91], RAs[92], DECs[92],
                            RAs[93], DECs[93], RAs[94], DECs[94], RAs[95], DECs[95],
                            RAs[96], DECs[96], RAs[97], DECs[97], RAs[98], DECs[98],
                            RAs[99], DECs[99], RAs[100], DECs[100], RAs[101], DECs[101],
                            RAs[102], DECs[102], RAs[103], DECs[103], RAs[104], DECs[104],
                            RAs[105], DECs[105], RAs[106], DECs[106], RAs[107], DECs[107],
                            RAs[108], DECs[108], RAs[109], DECs[109], RAs[110], DECs[110],
                            RAs[111], DECs[111], RAs[112], DECs[112], RAs[113], DECs[113],
                            RAs[114], DECs[114], RAs[115], DECs[115], RAs[116], DECs[116],
                            RAs[117], DECs[117], RAs[118], DECs[118], RAs[119], DECs[119] );

    if (npointing%2 == 1)
    {
        fprintf(stderr, "Number of RAs do not equal the number of Decs given. Exiting\n");
        fprintf(stderr, "npointings : %d\n", npointing);
        fprintf(stderr, "RAs[0] : %s\n", RAs[0]);
        fprintf(stderr, "DECs[0] : %s\n", DECs[0]);
        exit(0);
    }
    else
        npointing /= 2; // converting from number of RAs and DECs to number of pointings

    char pointing_array[npointing][2][64];
    int p;
    for ( p = 0; p < npointing; p++) 
    {
       strcpy( pointing_array[p][0], RAs[p] );
       strcpy( pointing_array[p][1], DECs[p] );
       fprintf(stderr, "[%f]  Pointing Num: %i  RA: %s  Dec: %s\n", NOW-begintime,
                             p, pointing_array[p][0], pointing_array[p][1]);
    }

    // Allocate memory
    fprintf(stderr, "Before\n");
    char **filenames = create_filenames( &opts );
    fprintf(stderr, "int attempt\n");
    //ComplexDouble ****complex_weights_array = create_complex_weights( 2, 128, 128, 2 );
    fprintf(stderr, "After\n");
    //ComplexDouble ****complex_weights_array = create_complex_weights( npointing, nstation, nchan, npol ); // [npointing][nstation][nchan][npol]
    fprintf(stderr, "After\n");
    ComplexDouble *****invJi = create_invJi( npointing, nstation, nchan, npol ); // [npointing][nstation][nchan][npol][npol]
    fprintf(stderr, "After\n");
    ComplexDouble ****detected_beam = create_detected_beam( npointing, 3*opts.sample_rate, nchan, npol ); // [npointing][3*opts.sample_rate][nchan][npol]

    // Read in info from metafits file
    fprintf( stderr, "[%f]  Reading in metafits file information from %s\n", NOW-begintime, opts.metafits);
    struct metafits_info mi;
    get_metafits_info( opts.metafits, &mi, opts.chan_width );

    // If using bandpass calibration solutions, calculate number of expected bandpass channels
    if (opts.cal.cal_type == RTS_BANDPASS)
        opts.cal.nchan = (nchan * opts.chan_width) / opts.cal.chan_width;

    // If a custom flag file has been provided, use that instead of the metafits flags
    int i;
    if (opts.custom_flags != NULL)
    {
        // Reset the weights to 1
        for (i = 0; i < nstation*npol; i++)
            mi.weights_array[i] = 1.0;

        // Open custom flag file for reading
        FILE *flagfile = fopen( opts.custom_flags, "r" );
        if (flagfile == NULL)
        {
            fprintf( stderr, "error: couldn't open flag file \"%s\" for "
                             "reading\n", opts.custom_flags );
            exit(EXIT_FAILURE);
        }

        // Read in flags from file
        int nitems;
        int flag, ant;
        while (!feof(flagfile))
        {
            // Read in next item
            nitems = fscanf( flagfile, "%d", &ant );
            if (nitems != 1 && !feof(flagfile))
            {
                fprintf( stderr, "error: couldn't parse flag file \"%s\"\n",
                        opts.custom_flags );
                exit(EXIT_FAILURE);
            }

            // Flag both polarisations of the antenna in question
            flag = ant*2;
            mi.weights_array[flag]   = 0.0;
            mi.weights_array[flag+1] = 0.0;
        }

        // Close file
        fclose( flagfile );
    }

    // Issue warnings if any antennas are being used which are flagged in the metafits file
    for (i = 0; i < nstation*npol; i++)
    {
        if (mi.weights_array[i] != 0.0 &&
            mi.flag_array[i]    != 0.0)
        {
            fprintf( stderr, "warning: antenna %3d, pol %d is included even "
                             "though it is flagged in the metafits file\n",
                             i / npol,
                             i % npol );
        }
    }
    
    double wgt_sum = 0;
    for (i = 0; i < nstation*npol; i++)
        wgt_sum += mi.weights_array[i];
    double invw = 1.0/wgt_sum;

    // Run get_delays to populate the delay_vals struct
    fprintf( stderr, "[%f]  Setting up output header information\n", NOW-begintime);
    struct delays delay_vals[npointing];
    get_delays(
            pointing_array,     // an array of pointings [pointing][ra/dec][characters]
            npointing,          // number of pointings
            opts.frequency,     // middle of the first frequency channel in Hz
            &opts.cal,          // struct holding info about calibration
            opts.sample_rate,   // = 10000 samples per sec
            opts.time_utc,      // utc time string
            0.0,                // seconds offset from time_utc at which to calculate delays
            delay_vals,        // Populate psrfits header info
            &mi,                // Struct containing info from metafits file
            NULL,               // complex weights array (ignore this time)
            NULL                // invJi array           (ignore this time)
    );

    // Create structures for holding header information
    struct psrfits  *pf;
    struct psrfits  *pf_incoh;
    pf = (struct psrfits *)malloc(npointing * sizeof(struct psrfits));
    pf_incoh = (struct psrfits *)malloc(1 * sizeof(struct psrfits));
    vdif_header     vhdr;
    struct vdifinfo *vf;
    vf = (struct vdifinfo *)malloc(npointing * sizeof(struct vdifinfo));


    // Create structures for the PFB filter coefficients
    int ntaps    = 12;
    int fil_size = ntaps * nchan; // = 12 * 128 = 1536

    // Populate the relevant header structs
    populate_psrfits_header( pf,       opts.metafits, opts.obsid,
            opts.time_utc, opts.sample_rate, opts.frequency, nchan,
            opts.chan_width,outpol_coh, opts.rec_channel, delay_vals,
            mi, npointing, 1 );
    populate_psrfits_header( pf_incoh, opts.metafits, opts.obsid,
            opts.time_utc, opts.sample_rate, opts.frequency, nchan,
            opts.chan_width, outpol_incoh, opts.rec_channel, delay_vals,
            mi, 1, 0 );

    populate_vdif_header( vf, &vhdr, opts.metafits, opts.obsid,
            opts.time_utc, opts.sample_rate, opts.frequency, nchan,
            opts.chan_width, opts.rec_channel, delay_vals, npointing );

    // To run asynchronously we require two memory allocations for each data 
    // set so multiple parts of the memory can be worked on at once.
    // We control this by changing the pointer to alternate between
    // the two memory allocations
    
    // Create array for holding the raw data
    int bytes_per_file = opts.sample_rate * nstation * npol * nchan;
    uint8_t *data;
    //uint8_t *data1;
    //uint8_t *data2;
    //cudaMallocHost( (void**)&data1, bytes_per_file * sizeof(uint8_t) );
    //cudaMallocHost( (void**)&data2, bytes_per_file * sizeof(uint8_t) );
    uint8_t *data1 = (uint8_t *)malloc( bytes_per_file * sizeof(uint8_t) );
    uint8_t *data2 = (uint8_t *)malloc( bytes_per_file * sizeof(uint8_t) );
    assert(data1);
    assert(data2);

    // Create output buffer arrays
    float *data_buffer_coh    = NULL;
    float *data_buffer_coh1   = NULL;
    float *data_buffer_coh2   = NULL; 
    float *data_buffer_incoh  = NULL;
    float *data_buffer_incoh1 = NULL;
    float *data_buffer_incoh2 = NULL;
    float *data_buffer_vdif   = NULL;
    float *data_buffer_vdif1  = NULL;
    float *data_buffer_vdif2  = NULL;

    /*data_buffer_coh1   = create_pinned_data_buffer_psrfits( npointing * nchan *
                                                            outpol_coh * pf[0].hdr.nsblk );
    data_buffer_coh2   = create_pinned_data_buffer_psrfits( npointing * nchan * 
                                                            outpol_coh * pf[0].hdr.nsblk );
    data_buffer_incoh1 = create_pinned_data_buffer_psrfits( nchan * outpol_incoh *
                                                            pf_incoh[0].hdr.nsblk );
    data_buffer_incoh2 = create_pinned_data_buffer_psrfits( nchan * outpol_incoh *
                                                            pf_incoh[0].hdr.nsblk );
    data_buffer_vdif1  = create_pinned_data_buffer_vdif( vf->sizeof_buffer *
                                                         npointing );
    data_buffer_vdif2  = create_pinned_data_buffer_vdif( vf->sizeof_buffer *
                                                         npointing );
    */
    /* Allocate host and device memory for the use of the cu_form_beam function */
    // Declaring pointers to the structs so the memory can be alternated
    struct gpu_formbeam_arrays gf;
    struct gpu_ipfb_arrays gi;
    malloc_formbeam( &gf, opts.sample_rate, nstation, nchan, npol,
            outpol_coh, outpol_incoh, npointing, NOW-begintime );

    if (opts.out_vdif)
    {
        malloc_ipfb( &gi, ntaps, opts.sample_rate, nchan, npol, fil_size, npointing );
    }

    // Set up parrel streams
    cudaStream_t streams[npointing];

    for ( p = 0; p < npointing; p++ )
        cudaStreamCreate(&(streams[p])) ;
    
    // TODO work out why the below won't work. It should save a bit of time 
    // instead of doing the same thing every second
    //populate_weights_johnes( &gf, complex_weights_array, invJi,
    //                         npointing, nstation, nchan, npol );
        

    fprintf( stderr, "[%f]  **BEGINNING BEAMFORMING**\n", NOW-begintime);
    
    // Set up sections checks that allow the asynchronous sections know when 
    // other sections have completed
    int file_no;
    int *read_check;
    int *calc_check;
    int **write_check;
    read_check = (int*)malloc(nfiles*sizeof(int));
    calc_check = (int*)malloc(nfiles*sizeof(int));
    write_check = (int**)malloc(nfiles*sizeof(int *));
    for ( file_no = 0; file_no < nfiles; file_no++ )
    {
        read_check[file_no]  = 0;//False
        calc_check[file_no]  = 0;//False
        write_check[file_no] = (int*)malloc(npointing*sizeof(int));
        for ( p = 0; p < npointing; p++ ) write_check[file_no][p] = 0;//False
    } 
    
    // Set up timing for each section
    long read_total_time, calc_total_time, write_total_time;

    int nthread;
    #pragma omp parallel 
    {
        #pragma omp master
        {
            nthread = omp_get_num_threads();
            fprintf( stderr, "[%f]  Number of CPU threads: %d\n", NOW-begintime, nthread);
        }
    }
    int thread_no;
    int exit_check = 0;
    // Sets up a parallel for loop for each of the available thread and 
    // assigns a section to each thread
    #pragma omp parallel for shared(read_check, calc_check, write_check, pf) private( thread_no, file_no, p, exit_check, data, data_buffer_coh, data_buffer_incoh, data_buffer_vdif )
    for (thread_no = 0; thread_no < nthread; ++thread_no)
    {
        // Read section -------------------------------------------------------
        if (thread_no == 0)
        {
            for (file_no = 0; file_no < nfiles; file_no++)
            {
                //Work out which memory allocation it's requires
                if (file_no%2 == 0) data = data1;
                else data = data2;
                
                //Waits until it can read 
                exit_check = 0; 
                while (1) 
                { 
                    #pragma omp critical (read_queue) 
                    { 
                        if (file_no == 0) 
                            exit_check = 1;//First read 
                        else if ( (read_check[file_no - 1] == 1) && (file_no == 1))  
                            exit_check = 1;//Second read
                        else if ( (read_check[file_no - 1] == 1) && (calc_check[file_no - 2] == 1) )
                            exit_check = 1;//Rest of the reads
                        else
                            exit_check = 0;
                    } 
                    if (exit_check) break; 
                }
                clock_t start = clock();
                #pragma omp critical (read_queue)
                {
                    // Read in data from next file
                    fprintf( stderr, "[%f] [%d/%d] Reading in data from %s \n", NOW-begintime,
                            file_no+1, nfiles, filenames[file_no]);
                    read_data( filenames[file_no], data, bytes_per_file  );
                    
                    // Records that this read section is complete
                    read_check[file_no] = 1;
                }
                read_total_time += clock() - start;
            }
        }

        // Calc section -------------------------------------------------------
        if (thread_no == 1)
        {
            int write_array_check = 1;
            for (file_no = 0; file_no < nfiles; file_no++)
            {
                //Work out which memory allocation it's requires
                if (file_no%2 == 0)
                {
                   data = data1;
                   data_buffer_coh   = data_buffer_coh1;
                   data_buffer_incoh = data_buffer_incoh1;
                   data_buffer_vdif  = data_buffer_vdif1;
                }
                else
                {
                   data = data2;
                   data_buffer_coh   = data_buffer_coh2;
                   data_buffer_incoh = data_buffer_incoh2;
                   data_buffer_vdif  = data_buffer_vdif2;
                }

                // Waits until it can start the calc
                exit_check = 0;
                while (1)
                {
                    #pragma omp critical (calc_queue)
                    {
                        // First two checks
                        if ( (file_no < 2) && (read_check[file_no] == 1) ) exit_check = 1;
                        // Rest of the checks. Checking if output memory is ready to be changed
                        else if (read_check[file_no] == 1) 
                        {    
                            write_array_check = 1;
                            // Loop through each pointing's write_check
                            for (int pc=0; pc<npointing; pc++)
                            {   
                                if (write_check[file_no - 2][pc] == 0) 
                                {
                                    // Not complete so changing check to False
                                    write_array_check = 0;
                                }
                            }
                            if (write_array_check == 1) exit_check = 1;
                        }
                    }
                    if (exit_check == 1) break; 
                }
                clock_t start = clock();
                // Get the next second's worth of phases / jones matrices, if needed
                fprintf( stderr, "[%f] [%d/%d] Calculating delays\n", NOW-begintime,
                                        file_no+1, nfiles );
                get_delays(
                        pointing_array,     // an array of pointings [pointing][ra/dec][characters]
                        npointing,          // number of pointings
                        opts.frequency,         // middle of the first frequency channel in Hz
                        &opts.cal,              // struct holding info about calibration
                        opts.sample_rate,       // = 10000 samples per sec
                        opts.time_utc,          // utc time string
                        (double)file_no,        // seconds offset from time_utc at which to calculate delays
                        NULL,                   // Don't update delay_vals
                        &mi,                    // Struct containing info from metafits file
                        complex_weights_array,  // complex weights array (answer will be output here)
                        invJi );                // invJi array           (answer will be output here)
                
                int complex_len = npointing * nstation * nchan * npol;
                int bi = 0;
                
                ComplexDouble *complex_weights_write = (ComplexDouble *)malloc((complex_len)*sizeof(ComplexDouble));
                for (int p   = 0; p < npointing;  p++)
                for (int ant = 0; ant < nstation; ant++)
                for (int ch  = 0; ch < nchan;     ch++)
                for (int pol = 0; pol < npol;     pol++)
                {
                    complex_weights_write[bi] = complex_weights_array[p][ant][ch][pol];
                    bi++;
                }

                FILE *f = fopen("complex_weights.data", "w");
                fwrite(complex_weights_write, sizeof(ComplexDouble), complex_len, f);
                fclose(f);

                f = fopen("complex_weights.data", "r");
                fseek(f, 0, SEEK_END);
                long filelen = ftell(f);
                rewind(f);
                ComplexDouble *buffer = (ComplexDouble *)malloc((filelen+1)*sizeof(ComplexDouble));
                fread(buffer, sizeof(ComplexDouble), filelen, f);
                //ComplexDouble ****file_complex_weights_array = create_complex_weights(
                //                                        npointing, nstation, nchan, npol );
                bi = 0;
                for (int p   = 0; p < npointing;  p++)
                for (int ant = 0; ant < nstation; ant++)
                for (int ch  = 0; ch < nchan;     ch++)
                for (int pol = 0; pol < npol;     pol++)
                {
                    if (CReald(complex_weights_array[p][ant][ch][pol]) != CReald(buffer[bi]) 
                     && CImagd(complex_weights_array[p][ant][ch][pol]) != CImagd(buffer[bi])) 
                    {
                        fprintf( stderr,  "NOT EQUAL %i\n", bi);
                        fprintf( stderr,  "%f   !=  %f\n",
                                CReald(complex_weights_array[p][ant][ch][pol]),
                                CReald(buffer[bi]));
                        fprintf( stderr,  "%f   !=  %f\n",
                                CImagd(complex_weights_array[p][ant][ch][pol]),
                                CImagd(buffer[bi]));
                    }
                    bi++;
                }
                exit(0);

                //f = fopen("invJi.data", "wb");
                //fwrite(clientdata, sizeof(char), sizeof(clientdata), f);
                //fclose(f);

                /*for (i = 0; i < npointing * nchan * outpol_coh * opts.sample_rate; i++)
                    data_buffer_coh[i] = 0.0;

                for (i = 0; i < npointing * nchan * outpol_incoh * opts.sample_rate; i++)
                    data_buffer_incoh[i] = 0.0;*/
                fprintf( stderr, "[%f] [%d/%d] Calculating beam\n", NOW-begintime,
                                        file_no+1, nfiles);
                
                cu_form_beam( data, &opts, complex_weights_array, invJi, file_no,
                              npointing, nstation, nchan, npol, outpol_coh, invw, &gf,
                              detected_beam, data_buffer_coh, data_buffer_incoh,
                              streams );

                // Invert the PFB, if requested
                if (opts.out_vdif)
                {
                    fprintf( stderr, "[%f] [%d/%d]   Inverting the PFB (full)\n", 
                                     NOW-begintime, file_no+1, nfiles);
                    cu_invert_pfb_ord( detected_beam, file_no, npointing, 
                            opts.sample_rate, nchan, npol, &gi, data_buffer_vdif );
                }

                // Records that this calc section is complete
                calc_check[file_no] = 1;
                calc_total_time += clock() - start;
            }
        }    
        // Write section ------------------------------------------------------
        if (thread_no == 2) //(thread_no > 1 && thread_no < npointing + 2)
        {
            p = thread_no - 2;
            for (file_no = 0; file_no < nfiles; file_no++)
            {
                //Work out which memory allocation it's requires
                if (file_no%2 == 0)
                {
                   data_buffer_coh   = data_buffer_coh1;
                   data_buffer_incoh = data_buffer_incoh1;
                   data_buffer_vdif  = data_buffer_vdif1;
                }
                else
                {
                   data_buffer_coh   = data_buffer_coh2;
                   data_buffer_incoh = data_buffer_incoh2;
                   data_buffer_vdif  = data_buffer_vdif2;
                }
                
                // Waits until it's time to write
                exit_check = 0;
                while (1)
                {
                    #pragma omp critical (write_queue)
                    if (calc_check[file_no] == 1) exit_check = 1;
                    if (exit_check == 1) break;
                }
                
                clock_t start = clock();

                for ( p = 0; p < npointing; p++)
                {
                    //printf_psrfits(&pf[p]);
                    fprintf( stderr, "[%f] [%d/%d] [%d/%d] Writing data to file(s)\n",
                            NOW-begintime, file_no+1, nfiles, p+1, npointing );

                    if (opts.out_coh)
                        psrfits_write_second( &pf[p], data_buffer_coh, nchan,
                                              outpol_coh, p );
                    if (opts.out_incoh && p == 0)
                        psrfits_write_second( &pf_incoh[p], data_buffer_incoh,
                                              nchan, outpol_incoh, p );
                    if (opts.out_vdif)
                        vdif_write_second( &vf[p], &vhdr, data_buffer_vdif,
                                           &vgain, p );

                    // Records that this write section is complete
                    write_check[file_no][p] = 1;
                }
                write_total_time += clock() - start;
            }
        }
    }

    fprintf( stderr, "[%f]  **FINISHED BEAMFORMING**\n", NOW-begintime);
    int read_ms = read_total_time * 1000 / CLOCKS_PER_SEC;
    fprintf( stderr, "[%f]  Total read  processing time: %3d.%3d s\n", 
                NOW-begintime, read_ms/1000, read_ms%1000);
    int calc_ms = calc_total_time * 1000 / CLOCKS_PER_SEC;
    fprintf( stderr, "[%f]  Total calc  processing time: %3d.%3d s\n", 
                NOW-begintime, calc_ms/1000, calc_ms%1000);
    int write_ms = write_total_time * 1000 / CLOCKS_PER_SEC;
    fprintf( stderr, "[%f]  Total write processing time: %3d.%3d s\n", 
                NOW-begintime, write_ms/1000, write_ms%1000);
    fprintf( stderr, "[%f]  Starting clean-up\n", NOW-begintime);

    // Free up memory
    destroy_filenames( filenames, &opts );
    destroy_complex_weights( complex_weights_array, npointing, nstation, nchan );
    destroy_invJi( invJi, npointing, nstation, nchan, npol );
    destroy_detected_beam( detected_beam, npointing, 3*opts.sample_rate, nchan );
    
    destroy_metafits_info( &mi );
    free( data_buffer_coh    );
    free( data_buffer_incoh  );
    free( data_buffer_vdif   );
    cudaFreeHost( data_buffer_coh1   );
    cudaFreeHost( data_buffer_coh2   );
    cudaFreeHost( data_buffer_incoh1 );
    cudaFreeHost( data_buffer_incoh2 );
    cudaFreeHost( data_buffer_vdif1  );
    cudaFreeHost( data_buffer_vdif2  );
    cudaFreeHost( data1 );
    cudaFreeHost( data2 );
        
    free( opts.obsid        );
    free( opts.time_utc     );
    free( opts.pointings    );
    free( opts.datadir      );
    free( opts.metafits     );
    free( opts.rec_channel  );
    free( opts.cal.filename );
    if (opts.out_incoh)
    {
        free( pf_incoh[0].sub.data        );
        free( pf_incoh[0].sub.dat_freqs   );
        free( pf_incoh[0].sub.dat_weights );
        free( pf_incoh[0].sub.dat_offsets );
        free( pf_incoh[0].sub.dat_scales  );
    }
    for (p = 0; p < npointing; p++)
    {
        if (opts.out_coh)
        {
            free( pf[p].sub.data        );
            free( pf[p].sub.dat_freqs   );
            free( pf[p].sub.dat_weights );
            free( pf[p].sub.dat_offsets );
            free( pf[p].sub.dat_scales  );
        }
        if (opts.out_vdif)
        {
            free( vf[p].b_scales  );
            free( vf[p].b_offsets );
        }
    }
    free_formbeam( &gf );
    if (opts.out_vdif)
    {
        free_ipfb( &gi );
    }
    #ifndef HAVE_CUDA
    // Clean up FFTW OpenMP
    fftw_cleanup_threads();
    #endif

    return 0;
}


void usage() {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: test_make_beam [OPTIONS]\n");

    fprintf(stderr, "\n");
    fprintf(stderr, "REQUIRED OPTIONS\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\t-d, --data-location=PATH  ");
    fprintf(stderr, "PATH is the directory containing the test data\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "OTHER OPTIONS\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\t-w, --write                ");
    fprintf(stderr, "Write out new test data files to compare to future tests\n");
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
                {"data-location",   required_argument, 0, 'd'},
                {"write",           no_argument      , 0, 'w'},
                {"help",            required_argument, 0, 'h'},
                {"version",         required_argument, 0, 'V'}
            };

            int option_index = 0;
            c = getopt_long( argc, argv,
                             "d:hVwd",
                             long_options, &option_index);
            if (c == -1)
                break;

            switch(c) {

                case 'd':
                    opts->datadir = strdup(optarg);
                    break;
                case 'w':
                    opts->write = 1;
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'V':
                    fprintf( stderr, "MWA Beamformer %s\n", VERSION_BEAMFORMER);
                    exit(0);
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
    assert( opts->datadir      != NULL );
}
