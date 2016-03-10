#ifndef GLOBAL_DEFS_H
#define GLOBAL_DEFS_H
#include "Python.h"
#include <boost/shared_ptr.hpp>
typedef double num;



#define GPUMEMBER __host__ __device__
#define SHARED(X) boost::shared_ptr<X>
//some files get grumpy if this is within the if.  doesn't hurt to have it declared multiple times
 
#define TESTING

#if defined( TESTING )
#define CUCHECK(c) {auto res = c; if (res!=cudaSuccess) { printf("CUDA runtime error %sn", cudaGetErrorString(res));}; assert(res == cudaSuccess);}
#else
#define CUCHECK(c) c
#endif

#if defined( TESTING )
#define SAFECALL(f, tag) {\
        cudaError_t errSync = cudaGetLastError();\
        cudaError_t errAsync = cudaDeviceSynchronize(); \
        if (errSync != cudaSuccess) {\
            cout << "sync error \"" << cudaGetErrorString(errSync) << "\" before call " << tag << endl;\
        };\
        if (errAsync != cudaSuccess) {\
            cout << "async error \"" << cudaGetErrorString(errAsync) << "\" before call " << tag << endl;\
        };\
        {f;}\
        errSync = cudaGetLastError();\
        errAsync = cudaDeviceSynchronize(); \
        if (errSync != cudaSuccess) {\
            cout << "sync error \"" << cudaGetErrorString(errSync) << "\" after call " << tag << endl;\
            assert(false);\
        };\
        if (errAsync != cudaSuccess) {\
            cout << "async error \"" << cudaGetErrorString(errAsync) << "\" after call " << tag << endl;\
            assert(false);\
        };\
}
        
#endif
     
#define GETIDX() (blockIdx.x*blockDim.x + threadIdx.x)
#define PERLINE 65536
#define XIDX(x, SIZE) (x % (PERLINE / SIZE))
#define YIDX(y, SIZE) (y / (PERLINE / SIZE))
#define PERBLOCK 256
#define NBLOCK(x) ((int) (ceil(x / (float) PERBLOCK)))
#define NBLOCKTEAM(x, threadsPerAtom) ((int) (ceil(x / (float) (PERBLOCK/threadsPerAtom))))

#define LINEARIDX(idx, ns) (ns.z*ns.y*idx.x + ns.z*idx.y + idx.z)
#endif

