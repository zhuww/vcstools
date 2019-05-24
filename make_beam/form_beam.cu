/********************************************************
 *                                                      *
 * Licensed under the Academic Free License version 3.0 *
 *                                                      *
 ********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

extern "C" {
#include "beam_common.h"
#include "form_beam.h"
#include "mycomplex.h"
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    /* Wrapper function for GPU/CUDA error handling. Every CUDA call goes through
       this function. It will return a message giving your the error string,
       file name and line of the error. Aborts on error. */

    if (code != 0)
    {
        fprintf(stderr, "GPUAssert:: %s - %s (%d)\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}

// define a macro for accessing gpuAssert
#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__, true);}


// define constants to be used in the kernel
#define NSTATION  128
#define NPOL      2
#define NSTOKES   4
// maximum number of pointings (currently)



__global__ void beamform_kernel( uint8_t *data,
                                 ComplexDouble *W,
                                 ComplexDouble *J,
                                 ComplexDouble *Bx,
                                 ComplexDouble *By,
                                 ComplexDouble *Nxx,
                                 ComplexDouble *Nxy,
                                 ComplexDouble *Nyy,
                                 double *Ia)
/* Layout for input arrays:
 *   data [nsamples] [nchan] [NPFB] [NREC] [NINC] -- see docs
 *   W    [NSTATION] [nchan] [NPOL]               -- weights array
 *   J    [NSTATION] [nchan] [NPOL] [NPOL]        -- jones matrix
 */
{
    // Translate GPU block/thread numbers into meaningful names
    int s   = threadIdx.x;  /* The (s)ample number */
    int ns  = blockDim.x;   /* The (n)umber of (s)amples */
        
    int c    = blockIdx.x;  /* The (c)hannel number */
    int nc   = gridDim.x;   /* The (n)umber of (c)hannels (=128) */
    int ant  = blockIdx.y;  /* The (ant)enna number */
    int p    = blockIdx.z;  /* The (p)ointing number */

    // Calculate the beam and the noise floor
    ComplexDouble Dx, Dy;
    ComplexDouble WDx, WDy;

    //if ( s == 0 && ant == 0 && p == 0 && c == 0 ) printf("np %d, nc %d, nant %d, ns %d\n", np, nc, gridDim.x, blockDim.x);

    /* Fix from Maceij regarding NaNs in output when running on Athena, 13 April 2018.
       Apparently the different compilers and architectures are treating what were 
       unintialised variables very differently */
    Dx  = CMaked( 0.0, 0.0 );
    Dy  = CMaked( 0.0, 0.0 );

    WDx = CMaked( 0.0, 0.0 );
    WDy = CMaked( 0.0, 0.0 );
    
    Bx[V_IDX(p,ant,c,s,ns,nc)]  = CMaked( 0.0, 0.0 );
    By[V_IDX(p,ant,c,s,ns,nc)]  = CMaked( 0.0, 0.0 );

    Nxx[V_IDX(p,ant,c,s,ns,nc)] = CMaked( 0.0, 0.0 );
    Nxy[V_IDX(p,ant,c,s,ns,nc)] = CMaked( 0.0, 0.0 );
    Nyy[V_IDX(p,ant,c,s,ns,nc)] = CMaked( 0.0, 0.0 );

    //if ( s == 0 && ant == 0 && p == 0 && c == 0 ) printf("in kernal Bx[%d] %f %f\n",V_IDX(p,ant,c,s,ns,nc), CReald(Bx[V_IDX(p,ant,c,s,ns,nc)]), CImagd(Bx[V_IDX(p,ant,c,s,ns,nc)]));
    
    if (p == 0) Ia[V_IDX(p,ant,c,s,ns,nc)] = 0.0;

    // Calculate beamform products for each antenna, and then add them together
    // Calculate the coherent beam (B = J*W*D)
    Dx  = UCMPLX4_TO_CMPLX_FLT(data[D_IDX(s,c,ant,0,nc)]);
    Dy  = UCMPLX4_TO_CMPLX_FLT(data[D_IDX(s,c,ant,1,nc)]);

    if (p == 0) Ia[V_IDX(p,ant,c,s,ns,nc)] = DETECT(Dx) + DETECT(Dy);

    WDx = CMuld( W[W_IDX(p,ant,c,0,nc)], Dx );
    WDy = CMuld( W[W_IDX(p,ant,c,1,nc)], Dy );

    Bx[V_IDX(p,ant,c,s,ns,nc)] = CAddd( CMuld( J[J_IDX(p,ant,c,0,0,nc)], WDx ),
                                        CMuld( J[J_IDX(p,ant,c,1,0,nc)], WDy ) );
    By[V_IDX(p,ant,c,s,ns,nc)] = CAddd( CMuld( J[J_IDX(p,ant,c,0,1,nc)], WDx ),
                                        CMuld( J[J_IDX(p,ant,c,1,1,nc)], WDy ) );

    Nxx[V_IDX(p,ant,c,s,ns,nc)] = CMuld( Bx[V_IDX(p,ant,c,s,ns,nc)],
                                         CConjd(Bx[V_IDX(p,ant,c,s,ns,nc)]) );
    Nxy[V_IDX(p,ant,c,s,ns,nc)] = CMuld( Bx[V_IDX(p,ant,c,s,ns,nc)], 
                                         CConjd(By[V_IDX(p,ant,c,s,ns,nc)]) );
    Nyy[V_IDX(p,ant,c,s,ns,nc)] = CMuld( By[V_IDX(p,ant,c,s,ns,nc)],
                                         CConjd(By[V_IDX(p,ant,c,s,ns,nc)]) );

}


__global__ void ant_sum(double *Ia,
                        ComplexDouble *Bx, 
                        ComplexDouble *By, 
                        ComplexDouble *Nxx, 
                        ComplexDouble *Nxy, 
                        ComplexDouble *Nyy)
{    
    // Translate GPU block/thread numbers into meaningful names
    int s    = threadIdx.x; /* The (s)ample number */
    int ns   = blockDim.x;   /* The (n)umber of (s)amples */

    int c    = blockIdx.x;  /* The (c)hannel number */
    int nc   = gridDim.x;   /* The (n)umber of (c)hannels (=128) */
    int ant  = blockIdx.y;  /* The (ant)enna number */
    int nant = gridDim.y;   /* The (n)umber of (p)ointings */
    int p    = blockIdx.z;  /* The (p)ointing number */
    
    // A summation over an array is faster on a GPU if you add half on array 
    // to its other half as than can be done in parallel. Then this is repeated
    // with half of the previous array until the array is down to 1.
    //if ( s == 0 && ant == 0 && p == 0 && c == 0 ) printf("in ant kernal Bx[%d] %f %f\n",V_IDX(p,ant,c,s,ns,nc), CReald(Bx[V_IDX(p,ant,c,s,ns,nc)]), CImagd(Bx[V_IDX(p,ant,c,s,ns,nc)]));
    
    if (p == 0) Ia[V_IDX(p,ant,c,s,ns,nc)] += Ia[V_IDX(p,ant+nant,c,s,ns,nc)];
    Bx[V_IDX(p,ant,c,s,ns,nc)]  = CAddd( Bx[V_IDX(p,ant,     c,s,ns,nc)], 
                                         Bx[V_IDX(p,ant+nant,c,s,ns,nc)] );
    By[V_IDX(p,ant,c,s,ns,nc)]  = CAddd( By[V_IDX(p,ant,     c,s,ns,nc)], 
                                         By[V_IDX(p,ant+nant,c,s,ns,nc)] );
    Nxx[V_IDX(p,ant,c,s,ns,nc)] = CAddd( Nxx[V_IDX(p,ant,     c,s,ns,nc)], 
                                         Nxx[V_IDX(p,ant+nant,c,s,ns,nc)] );
    Nxy[V_IDX(p,ant,c,s,ns,nc)] = CAddd( Nxy[V_IDX(p,ant,     c,s,ns,nc)], 
                                         Nxy[V_IDX(p,ant+nant,c,s,ns,nc)] );
    Nyy[V_IDX(p,ant,c,s,ns,nc)] = CAddd( Nyy[V_IDX(p,ant,     c,s,ns,nc)], 
                                         Nyy[V_IDX(p,ant+nant,c,s,ns,nc)] );
} 

__global__ void form_stokes(ComplexDouble *Bd,
                            float *C,
                            float *I,
                            ComplexDouble *Bx, 
                            ComplexDouble *By, 
                            ComplexDouble *Nxx, 
                            ComplexDouble *Nxy, 
                            ComplexDouble *Nyy, 
                            double *Ia,
                            double invw)
/* Layout for output arrays:
 *   Bd   [nsamples] [nchan]   [NPOL]             -- detected beam
 *   C    [nsamples] [NSTOKES] [nchan]            -- coherent full stokes
 *   I    [nsamples] [nchan]                      -- incoherent
 */

{    
    // Translate GPU block/thread numbers into meaningful names
    int s   = threadIdx.x;  /* The (s)ample number */
    int ns  = blockDim.x;   /* The (n)umber of (s)amples */

    int ant = 0;            /* The (ant)enna number */
    int c   = blockIdx.x;   /* The (c)hannel number */
    int nc  = gridDim.x;    /* The (n)umber of (c)hannels (=128) */
    int p   = blockIdx.y;   /* The (p)ointing number */
    int np  = gridDim.y;    /* The (n)umber of (p)ointings */
    
    
    // Clear output mem
    C[C_IDX(p,s,0,c,ns,nc)] = 0.0;
    C[C_IDX(p,s,1,c,ns,nc)] = 0.0;
    C[C_IDX(p,s,2,c,ns,nc)] = 0.0;
    C[C_IDX(p,s,3,c,ns,nc)] = 0.0;

    Bd[B_IDX(p,s,c,0,ns,nc)] = CMaked( 0.0, 0.0 );
    Bd[B_IDX(p,s,c,1,ns,nc)] = CMaked( 0.0, 0.0 );


    // Form the stokes parameters for the coherent beam
    // Only doing it for ant 0 so that it only prints once
    float bnXX = DETECT(Bx[V_IDX(p,ant,c,s,ns,nc)]) - CReald(Nxx[V_IDX(p,ant,c,s,ns,nc)]);
    float bnYY = DETECT(By[V_IDX(p,ant,c,s,ns,nc)]) - CReald(Nyy[V_IDX(p,ant,c,s,ns,nc)]);
    ComplexDouble bnXY = CSubd( CMuld( Bx[V_IDX(p,ant,c,s,ns,nc)], 
                                       CConjd( By[V_IDX(p,ant,c,s,ns,nc)] ) ),
                                Nxy[V_IDX(p,ant,c,s,ns,nc)] );

    // The incoherent beam
    I[I_IDX(s,c,nc)] = Ia[V_IDX(0,s,ant,c,np,nc)];

    // Stokes I, Q, U, V:
    C[C_IDX(p,s,0,c,ns,nc)] = invw*(bnXX + bnYY);
    C[C_IDX(p,s,1,c,ns,nc)] = invw*(bnXX - bnYY);
    C[C_IDX(p,s,2,c,ns,nc)] =  2.0*invw*CReald( bnXY );
    C[C_IDX(p,s,3,c,ns,nc)] = -2.0*invw*CImagd( bnXY );

    // The beamformed products
    Bd[B_IDX(p,s,c,0,ns,nc)] = Bx[V_IDX(p,ant,c,s,ns,nc)];
    Bd[B_IDX(p,s,c,1,ns,nc)] = By[V_IDX(p,ant,c,s,ns,nc)];
}

__global__ void flatten_bandpass_I_kernel(float *I,
                                     int nstep)/* uint8_t *Iout ) */
{
    // For just doing stokes I
    // One block
    // 128 threads each thread will do one channel
    // (we have already summed over all ant)

    // For doing the C array (I,Q,U,V)
    // ... figure it out later.

    // Translate GPU block/thread numbers into meaningful names
    int chan = threadIdx.x; /* The (c)hannel number */
    int nchan = blockDim.x; /* The total number of channels */
    float band;

    int new_var = 32; /* magic number */
    int i;

    float *data_ptr;

    // initialise the band 'array'
    band = 0.0;

    // accumulate abs(data) over all time samples and save into band
    data_ptr = I + I_IDX(0, chan, nchan);
    for (i=0;i<nstep;i++) { // time steps
        band += fabsf(*data_ptr);
        data_ptr = I + I_IDX(i,chan,nchan);
    }

    // now normalise the incoherent beam
    data_ptr = I + I_IDX(0, chan, nchan);
    for (i=0;i<nstep;i++) { // time steps
        *data_ptr = (*data_ptr)/( (band/nstep)/new_var );
        data_ptr = I + I_IDX(i,chan,nchan);
    }

}


__global__ void flatten_bandpass_C_kernel(float *C,
                                          int nstep)/* uint8_t *Iout ) */
{
    // For just doing stokes I
    // One block
    // 128 threads each thread will do one channel
    // (we have already summed over all ant)

    // For doing the C array (I,Q,U,V)
    // ... figure it out later.

    // Translate GPU block/thread numbers into meaningful names
    int chan   = threadIdx.x; /* The (c)hannel number */
    int nchan  = blockDim.x; /* The total number of channels */
    int p      = blockIdx.x;
    int np  = gridDim.x;    /* The (n)umber of (p)ointings */

    int stokes = threadIdx.y;
    //int nstokes = blockDim.y;
    float band;

    int new_var = 32; /* magic number */
    int i;

    float *data_ptr;

    // initialise the band 'array'
    band = 0.0;

    // accumulate abs(data) over all time samples and save into band
    //data_ptr = C + C_IDX(0,stokes,chan,nchan);
    for (i=0;i<nstep;i++) { // time steps
        data_ptr = C + C_IDX(p,i,stokes,chan,np,nchan);
        band += fabsf(*data_ptr);
    }

    // now normalise the coherent beam
    //data_ptr = C + C_IDX(0,stokes,chan,nchan);
    for (i=0;i<nstep;i++) { // time steps
        data_ptr = C + C_IDX(p,i,stokes,chan,np,nchan);
        *data_ptr = (*data_ptr)/( (band/nstep)/new_var );
    }

}


void cu_form_beam( uint8_t *data, struct make_beam_opts *opts,
                   ComplexDouble ****complex_weights_array,
                   ComplexDouble *****invJi, int file_no, 
                   int npointing, int nstation, int nchan,
                   int npol, int outpol_coh, double invw,
                   struct gpu_formbeam_arrays *g,
                   ComplexDouble ****detected_beam, float *coh, float *incoh,
                   int nchunk )
/* The CPU version of the beamforming operations, using OpenMP for
 * parallelisation.
 *
 * Inputs:
 *   data    = array of 4bit+4bit complex numbers. For data order, refer to the
 *             documentation.
 *   opts    = passed option parameters, containing meta information about the
 *             obs and the data
 *   W       = complex weights array. [npointing][nstation][nchan][npol]
 *   J       = inverse Jones matrix. [npointing][nstation][nchan][npol][npol]
 *   file_no = number of file we are processing, starting at 0.
 *   nstation     = 128
 *   nchan        = 128
 *   npol         = 2 (X,Y)
 *   outpol_coh   = 4 (I,Q,U,V)
 *   invw         = the reciprocal of the sum of the antenna weights
 *   g            = struct containing pointers to various arrays on
 *                  both host and device
 *
 * Outputs:
 *   detected_beam = result of beamforming operation, summed over antennas
 *                   [2*nsamples][nchan][npol]
 *   coh           = result in Stokes parameters (minus noise floor)
 *                   [nsamples][nstokes][nchan]
 *   incoh         = result (just Stokes I)
 *                   [nsamples][nchan]
 *
 * Assumes "coh" and "incoh" contain only zeros.
 */
{
    // Setup input values (= populate W and J)
    int p, s, ant, ch, pol, pol2;
    int Wi, Ji;
    for (p   = 0; p   < npointing; p++  )
    for (ant = 0; ant < nstation ; ant++)
    for (ch  = 0; ch  < nchan    ; ch++ )
    for (pol = 0; pol < npol     ; pol++)
    {
        Wi = p   * (npol*nchan*nstation) +
             ant * (npol*nchan) +
             ch  * (npol) +
             pol;
        g->W[Wi] = complex_weights_array[p][ant][ch][pol];

        for (pol2 = 0; pol2 < npol; pol2++)
        {
            Ji = Wi*npol + pol2;
            g->J[Ji] = invJi[p][ant][ch][pol][pol2];
        }
    }

    // Copy the data to the device
    gpuErrchk(cudaMemcpy( g->d_W,    g->W, g->W_size,    cudaMemcpyHostToDevice ));
    gpuErrchk(cudaMemcpy( g->d_J,    g->J, g->J_size,    cudaMemcpyHostToDevice ));
    
    // Divide the gpu calculation into multiple time chunks so there is enough room on the GPU
    for (int ichunk = 0; ichunk < nchunk; ichunk++)
    {    
        gpuErrchk(cudaMemcpy( g->d_data,
                              data + ichunk * g->data_size / sizeof(uint8_t),
                              g->data_size,  cudaMemcpyHostToDevice ));
        
        // Call the kernels
        // sammples_chan(index=blockIdx.x  size=gridDim.x,
        //               index=blockIdx.y  size=gridDim.y)
        // stat_point   (index=threadIdx.x size=blockDim.x,
        //               index=threadIdx.y size=blockDim.y)
        dim3 samples(opts->sample_rate / nchunk);
        dim3 chan_ant_point(nchan, NSTATION, npointing);
        cudaDeviceSynchronize();

        beamform_kernel<<<chan_ant_point, samples>>>( g->d_data, 
                          g->d_W, g->d_J, g->d_Bx, g->d_By, 
                          g->d_Nxx, g->d_Nxy, g->d_Nyy, g->d_incoh_volt);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        
        // sum over antennas/tile
        for (int nant = NSTATION / 2; nant > 0; nant = nant / 2)
        {
            dim3 chan_ant_point(nchan, nant, npointing);
            ant_sum<<<chan_ant_point, samples>>>( g->d_incoh_volt, 
                                                   g->d_Bx, g->d_By, 
                                                   g->d_Nxx, g->d_Nxy, g->d_Nyy );
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        }

        //form stokes
        dim3 chan_point(nchan, npointing);
        form_stokes<<<chan_point, samples>>>( g->d_Bd, g->d_coh, g->d_incoh, 
                                                  g->d_Bx, g->d_By,
                                                  g->d_Nxx, g->d_Nxy, g->d_Nyy,
                                                  g->d_incoh_volt, invw);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        // 1 block per pointing direction, hence the 1 for now
        // TODO check if these actually work, can't see them return values.
        // The incoh kernal also takes 40 second for some reason so commenting out
        //flatten_bandpass_I_kernel<<<1, nchan>>>(g->d_incoh, opts->sample_rate);
        
        // now do the same for the coherent beam
        //dim3 chan_stokes(nchan, outpol_coh);
        //flatten_bandpass_C_kernel<<<npointing, chan_stokes>>>(g->d_coh, opts->sample_rate);
        //cudaDeviceSynchronize(); // Memcpy acts as a synchronize step so don't sync here
    }
    // Copy the results back into host memory
    gpuErrchk(cudaMemcpy( g->Bd,
                          g->d_Bd,
                          g->Bd_size, cudaMemcpyDeviceToHost ));
    gpuErrchk(cudaMemcpy( incoh,    
                          g->d_incoh,
                          g->incoh_size, cudaMemcpyDeviceToHost ));
    gpuErrchk(cudaMemcpy( coh,      
                          g->d_coh,
                          g->coh_size,   cudaMemcpyDeviceToHost ));
    //fprintf( stderr, "coh 0,1,2,3 %f %f %f %f\n", coh[0], coh[1], coh[2], coh[3]);
    // Copy the data back from Bd back into the detected_beam array
    // Make sure we put it back into the correct half of the array, depending
    // on whether this is an even or odd second.
    int offset, i;
    offset = file_no % 3 * opts->sample_rate;
    
    for (p   = 0; p   < npointing        ; p++  )
    for (s   = 0; s   < opts->sample_rate; s++  )
    for (ch  = 0; ch  < nchan            ; ch++ )
    for (pol = 0; pol < npol             ; pol++)
    {
        i = p  * npol * nchan * opts->sample_rate +
            s  * npol * nchan                     +
            ch * npol                             +
            pol;

        /*if ( CReald(g->Bd[i]) != 0.0 && CImagd(g->Bd[i]) != 0.0) {
            fprintf( stderr, "Bd[%d]: %f %f \n", i, CReald(g->Bd[i]), CImagd(g->Bd[i]));
        }*/
        detected_beam[p][s+offset][ch][pol] = g->Bd[i];
    }
}

void malloc_formbeam( struct gpu_formbeam_arrays *g, unsigned int sample_rate,
                      int nstation, int nchan, int npol, int outpol_coh, 
                      int outpol_incoh, int npointing, int nchunk, float clock )
{
    // Calculate array sizes for host and device
    g->coh_size   = npointing * sample_rate * outpol_coh * nchan * sizeof(float);
    g->incoh_size = sample_rate * outpol_incoh * nchan* sizeof(float);
    g->data_size  = sample_rate * nstation * nchan * npol / nchunk * sizeof(uint8_t);
    g->Bd_size    = npointing * sample_rate * nchan * npol * sizeof(ComplexDouble);
    g->W_size     = npointing * nstation * nchan * npol * sizeof(ComplexDouble);
    g->J_size     = npointing * nstation * nchan * npol * npol * sizeof(ComplexDouble);
    g->volt_size  = npointing * sample_rate * nstation * nchan / nchunk * sizeof(ComplexDouble);

    // Allocate host memory
    g->W  = (ComplexDouble *)malloc( g->W_size );
    g->J  = (ComplexDouble *)malloc( g->J_size );
    g->Bd = (ComplexDouble *)malloc( g->Bd_size * nchunk );
    fprintf( stderr, "[%f]  %d GB GPU memory allocated\n", clock, (g->W_size + 
                              g->J_size + g->Bd_size + 2 * g->data_size +
                              g->coh_size + g->incoh_size + 5 * g->volt_size) 
                              /1000000000 );

    // Allocate device memory
    gpuErrchk(cudaMalloc( (void **)&g->d_W,     g->W_size ));
    gpuErrchk(cudaMalloc( (void **)&g->d_J,     g->J_size ));
    gpuErrchk(cudaMalloc( (void **)&g->d_Bd,    g->Bd_size ));
    gpuErrchk(cudaMalloc( (void **)&g->d_data,  g->data_size ));
    gpuErrchk(cudaMalloc( (void **)&g->d_coh,   g->coh_size ));
    gpuErrchk(cudaMalloc( (void **)&g->d_incoh, g->incoh_size ));
    gpuErrchk(cudaMalloc( (void **)&g->d_Bx,    g->volt_size ));
    gpuErrchk(cudaMalloc( (void **)&g->d_By,    g->volt_size ));
    gpuErrchk(cudaMalloc( (void **)&g->d_Nxx,   g->volt_size ));
    gpuErrchk(cudaMalloc( (void **)&g->d_Nxy,   g->volt_size ));
    gpuErrchk(cudaMalloc( (void **)&g->d_Nyy,   g->volt_size ));
    gpuErrchk(cudaMalloc( (void **)&g->d_incoh_volt, g->volt_size / npointing));

}

void free_formbeam( struct gpu_formbeam_arrays *g )
{
    // Free memory on host and device
    free( g->W );
    free( g->J );
    free( g->Bd );
    cudaFree( g->d_W );
    cudaFree( g->d_J );
    cudaFree( g->d_Bd );
    cudaFree( g->d_data );
    cudaFree( g->d_coh );
    cudaFree( g->d_incoh );
}
