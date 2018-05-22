#include "FixChargeEwald.h"

#include "BoundsGPU.h"
#include "cutils_func.h"
#include "cutils_math.h"
#include "GridGPU.h"
#include "State.h"
#include <cufft.h>
#include "globalDefs.h"
#include <fstream>
#include "Virial.h"
#include "helpers.h"

#include "PairEvaluatorNone.h"
#include "EvaluatorWrapper.h"
// #include <cmath>
namespace py = boost::python;
const std::string chargeEwaldType = "ChargeEwald";

// MW: Note that this function is a verbatim copy of that which appears in GridGPU.cu
//     consider combining
__global__ void computeCentroid(real4 *centroids, real4 *xs, int nAtoms, int nPerRingPoly, BoundsGPU bounds) {
   int idx = GETIDX();
    int nRingPoly = nAtoms / nPerRingPoly;
    if (idx < nRingPoly) {
        int baseIdx = idx * nPerRingPoly;
        real3 init = make_real3(xs[baseIdx]);
        real3 diffSum = make_real3(0, 0, 0);
        for (int i=baseIdx+1; i<baseIdx + nPerRingPoly; i++) {
            real3 next = make_real3(xs[i]);
            real3 dx = bounds.minImage(next - init);
            diffSum += dx;
        }
        diffSum /= nPerRingPoly;
        real3 unwrappedPos = init + diffSum;
        real3 trace = bounds.trace();
        real3 diffFromLo = unwrappedPos - bounds.lo;
// so, we need to use either floor() or floorf(), depending on typedef of real
#ifdef DASH_DOUBLE
        real3 imgs = floor(diffFromLo / trace); //are unskewed at this point
#else
        real3 imgs = floorf(diffFromLo / trace); //are unskewed at this point
#endif
        real3 wrappedPos = unwrappedPos - trace * imgs * bounds.periodic;

        centroids[idx] = make_real4(wrappedPos);
    }

}

// MW: This is a duplicated function from GridGPU.cu
 __global__ void periodicWrapCpy(real4 *xs, int nAtoms, BoundsGPU bounds) {
 
     int idx = GETIDX(); 
     if (idx < nAtoms) {
         
         real4 pos = xs[idx];
         
         real id = pos.w;
         real3 trace = bounds.trace();
         real3 diffFromLo = make_real3(pos) - bounds.lo;
// same as above - floor() or floorf(), depending on typedef of real
#ifdef DASH_DOUBLE
         real3 imgs = floor(diffFromLo / trace); //are unskewed at this point
#else
         real3 imgs = floorf(diffFromLo / trace); //are unskewed at this point
#endif /* DASH_DOUBLE */
         pos -= make_real4(trace * imgs * bounds.periodic);
         pos.w = id;
         //if (not(pos.x==orig.x and pos.y==orig.y and pos.z==orig.z)) { //sigh
         if (imgs.x != 0 or imgs.y != 0 or imgs.z != 0) {
             xs[idx] = pos;
         }
     }
 
 }
//different implementation for different interpolation orders
//TODO template
//order 1 nearest point
__global__ void map_charge_to_grid_order_1_cu(int nRingPoly, int nPerRingPoly, real4 *xs,  real *qs,  BoundsGPU bounds,
                                      int3 sz,real *grid/*convert to real for cufffComplex*/,real  Qunit) {

    int idx = GETIDX();
    if (idx < nRingPoly) {
        real4 posWhole = xs[idx];
        real3 pos = make_real3(posWhole)-bounds.lo;

        real qi = Qunit*qs[idx * nPerRingPoly];
        
        //find nearest grid point
// might be possible to change this back to real3; but, it was complaining before
#ifdef DASH_DOUBLE
        double3 h=bounds.trace()/make_double3(sz);
#else 
        float3 h = bounds.trace()/make_float3(sz);
#endif
        int3 nearest_grid_point=make_int3((pos+0.5*h)/h);
        //or
        int3 p=nearest_grid_point;
        if (p.x>0) p.x-=int(p.x/sz.x)*sz.x;
        if (p.y>0) p.y-=int(p.y/sz.y)*sz.y;
        if (p.z>0) p.z-=int(p.z/sz.z)*sz.z;
        if (p.x<0) p.x-=int((p.x+1)/sz.x-1)*sz.x;
        if (p.y<0) p.y-=int((p.y+1)/sz.y-1)*sz.y;
        if (p.z<0) p.z-=int((p.z+1)/sz.z-1)*sz.z;
        atomicAdd(&grid[p.x*sz.y*sz.z*2+p.y*sz.z*2+p.z*2], 1.0*qi);
    }
}

// coefficients here are from appendix E of paper referenced in header file
inline __host__ __device__ real W_p_3(int i,real x){
    if (i==-1) return 0.125-0.5*x+0.5*x*x;
    if (i== 0) return 0.75-x*x;
    /*if (i== 1)*/ return 0.125+0.5*x+0.5*x*x;
}


__global__ void map_charge_to_grid_order_3_cu(int nRingPoly, int nPerRingPoly, real4 *xs,  real *qs,  BoundsGPU bounds,
                                      int3 sz,real *grid/*convert to real for cufffComplex*/,real  Qunit) {

    int idx = GETIDX();
    if (idx < nRingPoly) {
        real4 posWhole = xs[idx];
        real3 pos = make_real3(posWhole)-bounds.lo;

        real qi = Qunit*qs[idx * nPerRingPoly];
        
        //find nearest grid point
        real3 h=bounds.trace()/make_real3(sz);
        int3 nearest_grid_point=make_int3((pos+0.5*h)/h);
        
        //distance from nearest_grid_point /h
        real3 d=pos/h-make_real3(nearest_grid_point);
        
        int3 p=nearest_grid_point;
        for (int ix=-1;ix<=1;ix++){
          p.x=nearest_grid_point.x+ix;
          real charge_yz_w=qi*W_p_3(ix,d.x);
          for (int iy=-1;iy<=1;iy++){
            p.y=nearest_grid_point.y+iy;
            real charge_z_w=charge_yz_w*W_p_3(iy,d.y);
            for (int iz=-1;iz<=1;iz++){
                p.z=nearest_grid_point.z+iz;
                real charge_w=charge_z_w*W_p_3(iz,d.z);
                if (p.x>0) p.x-=int(p.x/sz.x)*sz.x;
                if (p.y>0) p.y-=int(p.y/sz.y)*sz.y;
                if (p.z>0) p.z-=int(p.z/sz.z)*sz.z;
                if (p.x<0) p.x-=int((p.x+1)/sz.x-1)*sz.x;
                if (p.y<0) p.y-=int((p.y+1)/sz.y-1)*sz.y;
                if (p.z<0) p.z-=int((p.z+1)/sz.z-1)*sz.z;
                if ((p.x<0) or (p.x>sz.x-1)) printf("grid point miss x  %d, %d, %d, %f \n", idx,p.x,nearest_grid_point.x,pos.x);
                if ((p.y<0) or (p.y>sz.y-1)) printf("grid point miss y  %d, %d, %d, %f \n", idx,p.y,nearest_grid_point.y,pos.y);
                if ((p.z<0) or (p.z>sz.z-1)) printf("grid point miss z  %d, %d, %d, %f \n", idx,p.z,nearest_grid_point.z,pos.z);
                
                atomicAdd(&grid[p.x*sz.y*sz.z*2+p.y*sz.z*2+p.z*2], charge_w);
                
            }
          }
        }
    }
}

inline __host__ __device__ real W_p_5(int i,real x){

    real x2 = x*x;
    real x4 = x2*x2;
    if (i==-2) return ( (1.0/384.0) * (1.0 - 8.0*x + 24.0*x2 - 32.0*x*x2 + 16.0*x4));
    if (i==-1) return ( (1.0/96.0 ) * (19.0 - 44.0*x + 24.0*x2 + 16.0*x*x2 - 16.0*x4));
    if (i==0)  return ( (1.0/192.0) * (115.0 - 120.0*x2 + 48.0*x4));
    if (i==1)  return ( (1.0/96.0)  * (19.0 + 44.0*x + 24.0*x2 - 16.0*x*x2 - 16.0*x4));
    return ((1.0/384.0) * (1.0 + 8.0*x + 24.0*x2 + 32.0*x*x2 + 16.0*x4));
}

__global__ void map_charge_to_grid_order_5_cu(int nRingPoly, int nPerRingPoly, real4 *xs,  real *qs,  BoundsGPU bounds,
                                      int3 sz,real *grid/*convert to real for cufffComplex*/,real  Qunit) {

    int idx = GETIDX();
    if (idx < nRingPoly) {
        real4 posWhole = xs[idx];
        real3 pos = make_real3(posWhole)-bounds.lo;

        real qi = Qunit*qs[idx * nPerRingPoly];
        
        //find nearest grid point
        real3 h=bounds.trace()/make_real3(sz);
        int3 nearest_grid_point=make_int3((pos+0.5*h)/h);
        
        //distance from nearest_grid_point /h
        real3 d=pos/h-make_real3(nearest_grid_point);
        
        int3 p=nearest_grid_point;
        for (int ix=-2;ix<=2;ix++){
          p.x=nearest_grid_point.x+ix;
          real charge_yz_w=qi*W_p_5(ix,d.x);
          for (int iy=-2;iy<=2;iy++){
            p.y=nearest_grid_point.y+iy;
            real charge_z_w=charge_yz_w*W_p_5(iy,d.y);
            for (int iz=-2;iz<=2;iz++){
                p.z=nearest_grid_point.z+iz;
                real charge_w=charge_z_w*W_p_5(iz,d.z);
                if (p.x>0) p.x-=int(p.x/sz.x)*sz.x;
                if (p.y>0) p.y-=int(p.y/sz.y)*sz.y;
                if (p.z>0) p.z-=int(p.z/sz.z)*sz.z;
                if (p.x<0) p.x-=int((p.x+1)/sz.x-1)*sz.x;
                if (p.y<0) p.y-=int((p.y+1)/sz.y-1)*sz.y;
                if (p.z<0) p.z-=int((p.z+1)/sz.z-1)*sz.z;
                if ((p.x<0) or (p.x>sz.x-1)) printf("grid point miss x  %d, %d, %d, %f \n", idx,p.x,nearest_grid_point.x,pos.x);
                if ((p.y<0) or (p.y>sz.y-1)) printf("grid point miss y  %d, %d, %d, %f \n", idx,p.y,nearest_grid_point.y,pos.y);
                if ((p.z<0) or (p.z>sz.z-1)) printf("grid point miss z  %d, %d, %d, %f \n", idx,p.z,nearest_grid_point.z,pos.z);
                
                atomicAdd(&grid[p.x*sz.y*sz.z*2+p.y*sz.z*2+p.z*2], charge_w);
                
            }
          }
        }
    }
}

// COMPLETE
inline __host__ __device__ real W_p_7(int i,real x){
    real x2 = x*x;
    real x4 = x2*x2;
    real x6 = x2*x4;
    if (i==0) return ( (1.0 / 11520.0) * (5887.0 - 4620.0*x2 + 1680.0 * x4 - 320.0*x6));
    real x3 = x2*x;
    real x5 = x3*x2;
    if (i==-3) return ( (1.0 / 46080.0) * (1.0 - 12.0*x + 60.0*x2 - 160.0*x3 + 240.0*x4 - 192.0*x5 + 64.0*x6));
    if (i==-2) return ( (1.0 / 23040.0) * (361.0 - 1416.0*x + 2220.0*x2 - 1600.0*x3 + 240.0*x4 + 384.0*x5 - 192.0*x6));
    if (i==-1) return ( (1.0 / 46080.0) * (10543.0 - 17340.0*x + 4740.0*x2 + 6880.0*x3 - 4080.0*x4 - 960.0*x5 + 960.0*x6));

    if (i==1) return ( (1.0 / 46080.0) * (10543.0 + 17340.0*x + 4740.0*x2 - 6880.0*x3 - 4080.0*x4 + 960.0*x5 + 960.0*x6));
    if (i==2) return ( (1.0 / 23040.0) * (361.0 + 1416.0*x + 2220.0*x2 + 1600.0*x3 + 240.0*x4 - 384.0*x5 - 192.0*x6));
    return  ((1.0/46080.0) * (1.0 + 12.0*x + 60.0*x2 + 160.0*x3 + 240.0*x4 + 192.0*x5 + 64.0*x6));
}


__global__ void map_charge_to_grid_order_7_cu(int nRingPoly, int nPerRingPoly, real4 *xs,  real *qs,  BoundsGPU bounds,
                                      int3 sz,real *grid/*convert to real for cufffComplex*/,real  Qunit) {

    int idx = GETIDX();
    if (idx < nRingPoly) {
        real4 posWhole = xs[idx];
        real3 pos = make_real3(posWhole)-bounds.lo;

        real qi = Qunit*qs[idx * nPerRingPoly];
        
        //find nearest grid point
        real3 h=bounds.trace()/make_real3(sz);
        int3 nearest_grid_point=make_int3((pos+0.5*h)/h);
        
        //distance from nearest_grid_point /h
        real3 d=pos/h-make_real3(nearest_grid_point);
        
        int3 p=nearest_grid_point;
        for (int ix=-3;ix<=3;ix++){
          p.x=nearest_grid_point.x+ix;
          real charge_yz_w=qi*W_p_7(ix,d.x);
          for (int iy=-3;iy<=3;iy++){
            p.y=nearest_grid_point.y+iy;
            real charge_z_w=charge_yz_w*W_p_7(iy,d.y);
            for (int iz=-3;iz<=3;iz++){
                p.z=nearest_grid_point.z+iz;
                real charge_w=charge_z_w*W_p_7(iz,d.z);
                if (p.x>0) p.x-=int(p.x/sz.x)*sz.x;
                if (p.y>0) p.y-=int(p.y/sz.y)*sz.y;
                if (p.z>0) p.z-=int(p.z/sz.z)*sz.z;
                if (p.x<0) p.x-=int((p.x+1)/sz.x-1)*sz.x;
                if (p.y<0) p.y-=int((p.y+1)/sz.y-1)*sz.y;
                if (p.z<0) p.z-=int((p.z+1)/sz.z-1)*sz.z;
                if ((p.x<0) or (p.x>sz.x-1)) printf("grid point miss x  %d, %d, %d, %f \n", idx,p.x,nearest_grid_point.x,pos.x);
                if ((p.y<0) or (p.y>sz.y-1)) printf("grid point miss y  %d, %d, %d, %f \n", idx,p.y,nearest_grid_point.y,pos.y);
                if ((p.z<0) or (p.z>sz.z-1)) printf("grid point miss z  %d, %d, %d, %f \n", idx,p.z,nearest_grid_point.z,pos.z);
                
                atomicAdd(&grid[p.x*sz.y*sz.z*2+p.y*sz.z*2+p.z*2], charge_w);
                
            }
          }
        }
    }
}



#ifdef DASH_DOUBLE
__global__ void map_charge_set_to_zero_cu(int3 sz,cufftDoubleComplex *grid) {
#else
__global__ void map_charge_set_to_zero_cu(int3 sz,cufftComplex *grid) {
#endif /* DASH_DOUBLE */
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.y<sz.y)&&(id.z<sz.z))             

#ifdef DASH_DOUBLE
         grid[id.x*sz.y*sz.z+id.y*sz.z+id.z]=make_cuDoubleComplex (0.0, 0.0);    
#else
         grid[id.x*sz.y*sz.z+id.y*sz.z+id.z]=make_cuComplex (0.0f, 0.0f);    
#endif
}

__device__ real sinc(real x){
  if ((x<0.1)&&(x>-0.1)){
    real x2=x*x;
#ifdef DASH_DOUBLE
    return 1.0 - x2*0.16666666667 + x2*x2*0.008333333333333333 - x2*x2*x2*0.00019841269841269841;    
#else 
    return 1.0 - x2*0.16666666667f + x2*x2*0.008333333333333333f - x2*x2*x2*0.00019841269841269841f;    
#endif
  }
    else return sin(x)/x;
}

__global__ void Green_function_cu(BoundsGPU bounds, int3 sz,real *Green_function,real alpha,
                                  //now some parameter for Gf calc
                                  int sum_limits, int intrpl_order) {
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.y<sz.y)&&(id.z<sz.z)){
          real3 h =bounds.trace()/make_real3(sz);
          
          //         2*PI
          real3 k= 6.28318530717958647693f*make_real3(id)/bounds.trace();
          if (id.x>sz.x/2) k.x= 6.28318530717958647693f*(id.x-sz.x)/bounds.trace().x;
          if (id.y>sz.y/2) k.y= 6.28318530717958647693f*(id.y-sz.y)/bounds.trace().y;
          if (id.z>sz.z/2) k.z= 6.28318530717958647693f*(id.z-sz.z)/bounds.trace().z;
          

          //OK GF(k)  = 4Pi/K^2 [SumforM(W(K+M)^2  exp(-(K+M)^2/4alpha) dot(K,K+M)/(K+M^2))] / 
          //                    [SumforM^2(W(K+M)^2)]
             
             
          real sum1=0.0f;   
          real sum2=0.0f;   
          real k2=lengthSqr(k);
          real Fouralpha2inv=0.25/alpha/alpha;
          if (k2!=0.0){
              for (int ix=-sum_limits;ix<=sum_limits;ix++){//TODO different limits 
                for (int iy=-sum_limits;iy<=sum_limits;iy++){
                  for (int iz=-sum_limits;iz<=sum_limits;iz++){
                      real3 kpM=k+6.28318530717958647693f*make_real3(ix,iy,iz)/h;
//                             kpM.x+=6.28318530717958647693f/h.x*ix;//TODO rewrite
//                             kpM.y+=6.28318530717958647693f/h.y*iy;
//                             kpM.z+=6.28318530717958647693f/h.z*iz;
                            real kpMlen=lengthSqr(kpM);
                            real W=sinc(kpM.x*h.x*0.5)*sinc(kpM.y*h.y*0.5)*sinc(kpM.z*h.z*0.5);
//                             for(int p=1;p<intrpl_order;p++)
//                                   W*=W;
    //                          W*=h;//not need- cancels out
//                             real W2=W*W;
                             real W2=pow(W,intrpl_order*2);
                            //4*PI
                            sum1+=12.56637061435917295385*exp(-kpMlen*Fouralpha2inv)*dot(k,kpM)/kpMlen*W2;
                            sum2+=W2;
                  }
                }
              }
              Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z]=sum1/(sum2*sum2)/k2;
          }else{
              Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z]=0.0f;
          }
      }
             
}

#ifdef DASH_DOUBLE
__global__ void potential_cu(int3 sz,real *Green_function,
                                    cufftDoubleComplex *FFT_qs, cufftDoubleComplex *FFT_phi){
#else
__global__ void potential_cu(int3 sz,real *Green_function,
                                    cufftComplex *FFT_qs, cufftComplex *FFT_phi){

#endif /* DASH_DOUBLE */
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.y<sz.y)&&(id.z<sz.z)){
        FFT_phi[id.x*sz.y*sz.z+id.y*sz.z+id.z]=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z]*Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z];
//TODO after Inverse FFT divide by volume
      }
}

#ifdef DASH_DOUBLE
__global__ void E_field_cu(BoundsGPU bounds, int3 sz,real *Green_function, cufftDoubleComplex *FFT_qs,
                           cufftDoubleComplex *FFT_Ex,cufftDoubleComplex *FFT_Ey,cufftDoubleComplex *FFT_Ez){
#else
__global__ void E_field_cu(BoundsGPU bounds, int3 sz,real *Green_function, cufftComplex *FFT_qs,
                           cufftComplex *FFT_Ex,cufftComplex *FFT_Ey,cufftComplex *FFT_Ez){

#endif
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.y<sz.y)&&(id.z<sz.z)){
          //K vector
#ifdef DASH_DOUBLE
          real3 k= 6.28318530717958647693*make_real3(id)/bounds.trace();
          if (id.x>sz.x/2) k.x= 6.28318530717958647693*(id.x-sz.x)/bounds.trace().x;
          if (id.y>sz.y/2) k.y= 6.28318530717958647693*(id.y-sz.y)/bounds.trace().y;
          if (id.z>sz.z/2) k.z= 6.28318530717958647693*(id.z-sz.z)/bounds.trace().z;        
 
          //ik*q(k)*Gf(k)
          cufftDoubleComplex Ex,Ey,Ez;
          real GF=Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z];
          cufftDoubleComplex q=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z];

#else
          real3 k= 6.28318530717958647693f*make_real3(id)/bounds.trace();
          if (id.x>sz.x/2) k.x= 6.28318530717958647693f*(id.x-sz.x)/bounds.trace().x;
          if (id.y>sz.y/2) k.y= 6.28318530717958647693f*(id.y-sz.y)/bounds.trace().y;
          if (id.z>sz.z/2) k.z= 6.28318530717958647693f*(id.z-sz.z)/bounds.trace().z;        
          //ik*q(k)*Gf(k)
          cufftComplex Ex,Ey,Ez;
          real GF=Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z];
          cufftComplex q=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z];
#endif

          Ex.y= k.x*q.x*GF;
          Ex.x=-k.x*q.y*GF;
          Ey.y= k.y*q.x*GF;
          Ey.x=-k.y*q.y*GF;
          Ez.y= k.z*q.x*GF;
          Ez.x=-k.z*q.y*GF;
          
          FFT_Ex[id.x*sz.y*sz.z+id.y*sz.z+id.z]=Ex;
          FFT_Ey[id.x*sz.y*sz.z+id.y*sz.z+id.z]=Ey;
          FFT_Ez[id.x*sz.y*sz.z+id.y*sz.z+id.z]=Ez;
          //TODO after Inverse FFT divide by -volume
      }
}

#ifdef DASH_DOUBLE
__global__ void Ewald_long_range_forces_order_1_cu(int nRingPoly, int nPerRingPoly, real4 *xs, real4 *fs, 
                                                   real *qs, BoundsGPU bounds,
                                                   int3 sz, cufftDoubleComplex *FFT_Ex,
                                                    cufftDoubleComplex *FFT_Ey,cufftDoubleComplex *FFT_Ez,real  Qunit,
                                                   bool storeForces, uint *ids, real4 *storedForces) {
#else
__global__ void Ewald_long_range_forces_order_1_cu(int nRingPoly, int nPerRingPoly, real4 *xs, real4 *fs, 
                                                   real *qs, BoundsGPU bounds,
                                                   int3 sz, cufftComplex *FFT_Ex,
                                                    cufftComplex *FFT_Ey,cufftComplex *FFT_Ez,real  Qunit,
                                                   bool storeForces, uint *ids, real4 *storedForces) {
#endif /* DASH_DOUBLE */    
    int idx = GETIDX();
    if (idx < nRingPoly) {
        real4 posWhole= xs[idx];
        real3 pos     = make_real3(posWhole)-bounds.lo;
        int    baseIdx = idx*nPerRingPoly;
        real  qi      = qs[baseIdx];
        
        //find nearest grid point
        real3 h=bounds.trace()/make_real3(sz);
        int3 nearest_grid_point=make_int3((pos+0.5*h)/h);

        int3 p=nearest_grid_point;        
        if (p.x>0) p.x-=int(p.x/sz.x)*sz.x;
        if (p.y>0) p.y-=int(p.y/sz.y)*sz.y;
        if (p.z>0) p.z-=int(p.z/sz.z)*sz.z;
        if (p.x<0) p.x-=int((p.x+1)/sz.x-1)*sz.x;
        if (p.y<0) p.y-=int((p.y+1)/sz.y-1)*sz.y;
        if (p.z<0) p.z-=int((p.z+1)/sz.z-1)*sz.z;
        
        //get E field
        real3 E;
        real volume=bounds.trace().x*bounds.trace().y*bounds.trace().z;
        E.x= -FFT_Ex[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
        E.y= -FFT_Ey[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
        E.z= -FFT_Ez[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
        
        // Apply force on centroid to all time slices for given atom
        real3 force= Qunit*qi*E;
        for (int i = 0; i< nPerRingPoly; i++) {
            fs[baseIdx + i] += force; 
        }

        if (storeForces) {
            for (int i = 0; i < nPerRingPoly; i++) {
                storedForces[ids[baseIdx+i]] = make_real4(force.x, force.y, force.z, 0);
            }
        }
    }
}


#ifdef DASH_DOUBLE
__global__ void Ewald_long_range_forces_order_3_cu(int nRingPoly, int nPerRingPoly, real4 *xs, real4 *fs, 
                                                   real *qs, BoundsGPU bounds,
                                                   int3 sz, cufftDoubleComplex *FFT_Ex,
                                                    cufftDoubleComplex *FFT_Ey,cufftDoubleComplex *FFT_Ez,real  Qunit,
                                                   bool storeForces, uint *ids, real4 *storedForces) {
#else
__global__ void Ewald_long_range_forces_order_3_cu(int nRingPoly, int nPerRingPoly, real4 *xs, real4 *fs, 
                                                   real *qs, BoundsGPU bounds,
                                                   int3 sz, cufftComplex *FFT_Ex,
                                                    cufftComplex *FFT_Ey,cufftComplex *FFT_Ez,real  Qunit,
                                                   bool storeForces, uint *ids, real4 *storedForces) {

#endif
    int idx = GETIDX();
    if (idx < nRingPoly) {
        real4 posWhole= xs[idx];
        real3 pos     = make_real3(posWhole)-bounds.lo;
        int    baseIdx = idx*nPerRingPoly;
        real  qi      = qs[baseIdx];

        //find nearest grid point
        real3 h=bounds.trace()/make_real3(sz);
        int3 nearest_grid_point=make_int3((pos+0.5*h)/h);
        
        //distance from nearest_grid_point /h
        real3 d=pos/h-make_real3(nearest_grid_point);

        real3 E=make_real3(0,0,0);
        real volume=bounds.trace().x*bounds.trace().y*bounds.trace().z;

        int3 p=nearest_grid_point;
        for (int ix=-1;ix<=1;ix++){
          p.x=nearest_grid_point.x+ix;
          for (int iy=-1;iy<=1;iy++){
            p.y=nearest_grid_point.y+iy;
            for (int iz=-1;iz<=1;iz++){
                p.z=nearest_grid_point.z+iz;
                if (p.x>0) p.x-=int(p.x/sz.x)*sz.x;
                if (p.y>0) p.y-=int(p.y/sz.y)*sz.y;
                if (p.z>0) p.z-=int(p.z/sz.z)*sz.z;
                if (p.x<0) p.x-=int((p.x+1)/sz.x-1)*sz.x;
                if (p.y<0) p.y-=int((p.y+1)/sz.y-1)*sz.y;
                if (p.z<0) p.z-=int((p.z+1)/sz.z-1)*sz.z;
                real3 Ep;
                real W_xyz=W_p_3(ix,d.x)*W_p_3(iy,d.y)*W_p_3(iz,d.z);
                
                Ep.x= -FFT_Ex[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                Ep.y= -FFT_Ey[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                Ep.z= -FFT_Ez[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                E+=W_xyz*Ep;
            }
          }
        }
               
        real3 force= Qunit*qi*E;
        // Apply force on centroid to all time slices for given atom
        for (int i = 0; i < nPerRingPoly; i++) {
            fs[baseIdx + i] += force;
        }

        if (storeForces) {
            for (int i = 0; i < nPerRingPoly; i++) {
                storedForces[ids[baseIdx+i]] = make_real4(force.x, force.y, force.z, 0);
            }
        }
    }
}


#ifdef DASH_DOUBLE
__global__ void Ewald_long_range_forces_order_5_cu(int nRingPoly, int nPerRingPoly, real4 *xs, real4 *fs, 
                                                   real *qs, BoundsGPU bounds,
                                                   int3 sz, cufftDoubleComplex *FFT_Ex,
                                                    cufftDoubleComplex *FFT_Ey,cufftDoubleComplex *FFT_Ez,real  Qunit,
                                                   bool storeForces, uint *ids, real4 *storedForces) {
#else
__global__ void Ewald_long_range_forces_order_5_cu(int nRingPoly, int nPerRingPoly, real4 *xs, real4 *fs, 
                                                   real *qs, BoundsGPU bounds,
                                                   int3 sz, cufftComplex *FFT_Ex,
                                                    cufftComplex *FFT_Ey,cufftComplex *FFT_Ez,real  Qunit,
                                                   bool storeForces, uint *ids, real4 *storedForces) {

#endif
    int idx = GETIDX();
    if (idx < nRingPoly) {
        real4 posWhole= xs[idx];
        real3 pos     = make_real3(posWhole)-bounds.lo;
        int    baseIdx = idx*nPerRingPoly;
        real  qi      = qs[baseIdx];

        //find nearest grid point
        real3 h=bounds.trace()/make_real3(sz);
        int3 nearest_grid_point=make_int3((pos+0.5*h)/h);
        
        //distance from nearest_grid_point /h
        real3 d=pos/h-make_real3(nearest_grid_point);

        real3 E=make_real3(0,0,0);
        real volume=bounds.trace().x*bounds.trace().y*bounds.trace().z;

        int3 p=nearest_grid_point;
        for (int ix=-2;ix<=2;ix++){
          p.x=nearest_grid_point.x+ix;
          for (int iy=-2;iy<=2;iy++){
            p.y=nearest_grid_point.y+iy;
            for (int iz=-2;iz<=2;iz++){
                p.z=nearest_grid_point.z+iz;
                if (p.x>0) p.x-=int(p.x/sz.x)*sz.x;
                if (p.y>0) p.y-=int(p.y/sz.y)*sz.y;
                if (p.z>0) p.z-=int(p.z/sz.z)*sz.z;
                if (p.x<0) p.x-=int((p.x+1)/sz.x-1)*sz.x;
                if (p.y<0) p.y-=int((p.y+1)/sz.y-1)*sz.y;
                if (p.z<0) p.z-=int((p.z+1)/sz.z-1)*sz.z;
                real3 Ep;
                real W_xyz=W_p_5(ix,d.x)*W_p_5(iy,d.y)*W_p_5(iz,d.z);
                
                Ep.x= -FFT_Ex[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                Ep.y= -FFT_Ey[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                Ep.z= -FFT_Ez[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                E+=W_xyz*Ep;
            }
          }
        }
               
        real3 force= Qunit*qi*E;
        // Apply force on centroid to all time slices for given atom
        for (int i = 0; i < nPerRingPoly; i++) {
            fs[baseIdx + i] += force;
        }

        if (storeForces) {
            for (int i = 0; i < nPerRingPoly; i++) {
                storedForces[ids[baseIdx+i]] = make_real4(force.x, force.y, force.z, 0);
            }
        }
    }
}


#ifdef DASH_DOUBLE
__global__ void Ewald_long_range_forces_order_7_cu(int nRingPoly, int nPerRingPoly, real4 *xs, real4 *fs, 
                                                   real *qs, BoundsGPU bounds,
                                                   int3 sz, cufftDoubleComplex *FFT_Ex,
                                                    cufftDoubleComplex *FFT_Ey,cufftDoubleComplex *FFT_Ez,real  Qunit,
                                                   bool storeForces, uint *ids, real4 *storedForces) {
#else
__global__ void Ewald_long_range_forces_order_7_cu(int nRingPoly, int nPerRingPoly, real4 *xs, real4 *fs, 
                                                   real *qs, BoundsGPU bounds,
                                                   int3 sz, cufftComplex *FFT_Ex,
                                                    cufftComplex *FFT_Ey,cufftComplex *FFT_Ez,real  Qunit,
                                                   bool storeForces, uint *ids, real4 *storedForces) {

#endif
    int idx = GETIDX();
    if (idx < nRingPoly) {
        real4 posWhole= xs[idx];
        real3 pos     = make_real3(posWhole)-bounds.lo;
        int    baseIdx = idx*nPerRingPoly;
        real  qi      = qs[baseIdx];

        //find nearest grid point
        real3 h=bounds.trace()/make_real3(sz);
        int3 nearest_grid_point=make_int3((pos+0.5*h)/h);
        
        //distance from nearest_grid_point /h
        real3 d=pos/h-make_real3(nearest_grid_point);

        real3 E=make_real3(0,0,0);
        real volume=bounds.trace().x*bounds.trace().y*bounds.trace().z;

        int3 p=nearest_grid_point;
        for (int ix=-3;ix<=3;ix++){
          p.x=nearest_grid_point.x+ix;
          for (int iy=-3;iy<=3;iy++){
            p.y=nearest_grid_point.y+iy;
            for (int iz=-3;iz<=3;iz++){
                p.z=nearest_grid_point.z+iz;
                if (p.x>0) p.x-=int(p.x/sz.x)*sz.x;
                if (p.y>0) p.y-=int(p.y/sz.y)*sz.y;
                if (p.z>0) p.z-=int(p.z/sz.z)*sz.z;
                if (p.x<0) p.x-=int((p.x+1)/sz.x-1)*sz.x;
                if (p.y<0) p.y-=int((p.y+1)/sz.y-1)*sz.y;
                if (p.z<0) p.z-=int((p.z+1)/sz.z-1)*sz.z;
                real3 Ep;
                real W_xyz=W_p_7(ix,d.x)*W_p_7(iy,d.y)*W_p_7(iz,d.z);
                
                Ep.x= -FFT_Ex[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                Ep.y= -FFT_Ey[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                Ep.z= -FFT_Ez[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                E+=W_xyz*Ep;
            }
          }
        }
               
        real3 force= Qunit*qi*E;
        // Apply force on centroid to all time slices for given atom
        for (int i = 0; i < nPerRingPoly; i++) {
            fs[baseIdx + i] += force;
        }

        if (storeForces) {
            for (int i = 0; i < nPerRingPoly; i++) {
                storedForces[ids[baseIdx+i]] = make_real4(force.x, force.y, force.z, 0);
            }
        }
    }
}




// XXX: may need to template this cufftComplex & double prec analog, if they are different;
//      -- consult NVIDIA docs to find out
#ifdef DASH_DOUBLE
__global__ void Energy_cu(int3 sz,real *Green_function,
                                    cufftDoubleComplex *FFT_qs, cufftDoubleComplex *E_grid){
#else
__global__ void Energy_cu(int3 sz,real *Green_function,
                                    cufftComplex *FFT_qs, cufftComplex *E_grid){
#endif
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.y<sz.y)&&(id.z<sz.z)){
#ifdef DASH_DOUBLE
        cufftDoubleComplex qi=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z];
        E_grid[id.x*sz.y*sz.z+id.y*sz.z+id.z]
            =make_cuDoubleComplex((qi.x*qi.x+qi.y*qi.y)*Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z],0.0);
#else
        cufftComplex qi=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z];
        E_grid[id.x*sz.y*sz.z+id.y*sz.z+id.z]
            =make_cuComplex((qi.x*qi.x+qi.y*qi.y)*Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z],0.0);

#endif
//TODO after Inverse FFT divide by volume
      }
}

#ifdef DASH_DOUBLE
__global__ void virials_cu(BoundsGPU bounds,int3 sz,Virial *dest,real alpha, real *Green_function,cufftDoubleComplex *FFT_qs,int warpSize){
#else
__global__ void virials_cu(BoundsGPU bounds,int3 sz,Virial *dest,real alpha, real *Green_function,cufftComplex *FFT_qs,int warpSize){
#endif
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.y<sz.y)&&(id.z<sz.z)){
#ifdef DASH_DOUBLE
          real3 k= 6.28318530717958647693*make_real3(id)/bounds.trace();
          if (id.x>sz.x/2) k.x= 6.28318530717958647693*(id.x-sz.x)/bounds.trace().x;
          if (id.y>sz.y/2) k.y= 6.28318530717958647693*(id.y-sz.y)/bounds.trace().y;
          if (id.z>sz.z/2) k.z= 6.28318530717958647693*(id.z-sz.z)/bounds.trace().z;        
          real klen=lengthSqr(k);
          cufftDoubleComplex qi=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z];

#else 
          
          real3 k= 6.28318530717958647693f*make_real3(id)/bounds.trace();
          if (id.x>sz.x/2) k.x= 6.28318530717958647693f*(id.x-sz.x)/bounds.trace().x;
          if (id.y>sz.y/2) k.y= 6.28318530717958647693f*(id.y-sz.y)/bounds.trace().y;
          if (id.z>sz.z/2) k.z= 6.28318530717958647693f*(id.z-sz.z)/bounds.trace().z;        
          real klen=lengthSqr(k);
          cufftComplex qi=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z];
          
#endif      
          real E=(qi.x*qi.x+qi.y*qi.y)*Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z];
          
          real differential=-2.0*(1.0/klen+0.25/(alpha*alpha));
          if (klen==0.0) {differential=0.0;E=0.0;}
          
          Virial virialstmp = Virial(0, 0, 0, 0, 0, 0);   
          virialstmp[0]=(1.0+differential*k.x*k.x)*E; //xx
          virialstmp[1]=(1.0+differential*k.y*k.y)*E; //yy
          virialstmp[2]=(1.0+differential*k.z*k.z)*E; //zz
          virialstmp[3]=(differential*k.x*k.y)*E; //xy
          virialstmp[4]=(differential*k.x*k.z)*E; //xz
          virialstmp[5]=(differential*k.y*k.z)*E; //yz

//           virials[id.x*sz.y*sz.z+id.y*sz.z+id.z]=virialstmp;
//           __syncthreads();
          extern __shared__ Virial tmpV[]; 
//           const int copyBaseIdx = blockDim.x*blockIdx.x * N_DATA_PER_THREAD + threadIdx.x;
//           const int copyIncrement = blockDim.x;
          tmpV[threadIdx.x*blockDim.y*blockDim.z+threadIdx.y*blockDim.z+threadIdx.z]=virialstmp;
          int curLookahead=1;
          int numLookaheadSteps = log2f(blockDim.x*blockDim.y*blockDim.z-1);
          const int sumBaseIdx = threadIdx.x*blockDim.y*blockDim.z+threadIdx.y*blockDim.z+threadIdx.z;
          __syncthreads();
          for (int i=0; i<=numLookaheadSteps; i++) {
              if (! (sumBaseIdx % (curLookahead*2))) {
                  tmpV[sumBaseIdx] += tmpV[sumBaseIdx + curLookahead];
              }
              curLookahead *= 2;
//               if (curLookahead >= (warpSize)) {//Doesn't work in 3D case 
                  __syncthreads();
//               }
          } 

          if (sumBaseIdx  == 0) {
            
              atomicAdd(&(dest[0].vals[0]), tmpV[0][0]);
              atomicAdd(&(dest[0].vals[1]), tmpV[0][1]);
              atomicAdd(&(dest[0].vals[2]), tmpV[0][2]);
              atomicAdd(&(dest[0].vals[3]), tmpV[0][3]);
              atomicAdd(&(dest[0].vals[4]), tmpV[0][4]);
              atomicAdd(&(dest[0].vals[5]), tmpV[0][5]);
          }          
      }
      
}

__global__ void applyStoredForces(int  nAtoms,
                real4 *fs,
                uint *ids, real4 *fsStored) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        real4 cur = fs[idx];
        real3 stored = make_real3(fsStored[ids[idx]]);
        cur += stored;
        fs[idx] = cur;
    }
}
__global__ void mapVirialToSingleAtom(Virial *atomVirials, Virial *fieldVirial, real volume) {
    //just mapping to one atom for now.  If we're looking at per-atom properties, should change to mapping to all atoms evenly
    atomVirials[0][threadIdx.x] += 0.5 * fieldVirial[0][threadIdx.x] / volume;
}


__global__ void mapEngToParticles(int nAtoms, real eng, real *engs) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        engs[idx] += eng;
    }
}

FixChargeEwald::FixChargeEwald(SHARED(State) state_, std::string handle_, std::string groupHandle_): FixCharge(state_, handle_, groupHandle_, chargeEwaldType, true){
    cufftCreate(&plan);
    prepared = false;
    canOffloadChargePairCalc = true;
    modeIsError = false;
    sz = make_int3(64, 64, 64);
    malloced = false;
    longRangeInterval = 1;
    interpolation_order = 5;
    r_cut = -1;
    setEvalWrapper();
}


FixChargeEwald::~FixChargeEwald(){
    cufftDestroy(plan);
    if (malloced) {
        cudaFree(FFT_Qs);
        cudaFree(FFT_Ex);
        cudaFree(FFT_Ey);
        cudaFree(FFT_Ez);
    }
}


//Root mean square force error estimation
const double amp_table[][7] = {
        {2.0/3.0,           0,                 0,                    0,                        0,                         0,                                0},
        {1.0/50.0,          5.0/294.0,         0,                    0,                        0,                         0,                                0},
        {1.0/588.0,         7.0/1440.0,        21.0/3872.0,          0,                        0,                         0,                                0},
        {1.0/4320.0,        3.0/1936.0,        7601.0/2271360.0,     143.0/28800.0,            0,                         0,                                0},
        {1.0/23232.0,       7601.0/12628160.0, 143.0/69120.0,        517231.0/106536960.0,     106640677.0/11737571328.0, 0,                                0},
        {691.0/68140800.0,  13.0/57600.0,      47021.0/35512320.0,   9694607.0/2095994880.0,   733191589.0/59609088000.0, 326190917.0/11700633600.0,        0},
        {1.0/345600.0,      3617.0/35512320.0, 745739.0/838397952.0, 56399353.0/12773376000.0, 25091609.0/1560084480.0,   1755948832039.0/36229939200000.0, 48887769399.0/37838389248.0}
}; 


double FixChargeEwald :: DeltaF_k(double t_alpha){
    int nAtoms = state->atoms.size(); 
   double sumx=0.0,sumy=0.0,sumz=0.0;
   for( int m=0;m<interpolation_order;m++){
       double amp=amp_table[interpolation_order-1][m];
       sumx+=amp*pow(h.x*t_alpha,2*m);
       sumy+=amp*pow(h.y*t_alpha,2*m);
       sumz+=amp*pow(h.z*t_alpha,2*m);
   }
   return total_Q2/3.0*(1.0/(L.x*L.x)*pow(t_alpha*h.x,interpolation_order)*sqrt(t_alpha*L.x/nAtoms*sqrt(2.0*M_PI)*sumx)+
                        1.0/(L.y*L.y)*pow(t_alpha*h.y,interpolation_order)*sqrt(t_alpha*L.y/nAtoms*sqrt(2.0*M_PI)*sumy)+
                        1.0/(L.z*L.z)*pow(t_alpha*h.z,interpolation_order)*sqrt(t_alpha*L.z/nAtoms*sqrt(2.0*M_PI)*sumz));
 }
 
double  FixChargeEwald :: DeltaF_real(double t_alpha){  
    int nAtoms = state->atoms.size(); 
   return 2*total_Q2/sqrt(nAtoms*r_cut*L.x*L.y*L.z)*exp(-t_alpha*t_alpha*r_cut*r_cut);
 } 
 
 
void FixChargeEwald::setTotalQ2() {
    int nAtoms = state->atoms.size();    
    GPUArrayGlobal<real>tmp(1);
    tmp.memsetByVal(0.0);
    real conversion = state->units.qqr_to_eng;


    accumulate_gpu<real,real, SumSqr, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(real)*PERBLOCK>>>
        (
         tmp.getDevData(),
         state->gpd.qs(state->gpd.activeIdx()),
         nAtoms,
         state->devManager.prop.warpSize,
         SumSqr());
    tmp.dataToHost();   
    total_Q2=conversion*tmp.h_data[0]/state->nPerRingPoly;

    tmp.memsetByVal(0.0);

    accumulate_gpu<real,real, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(real)*PERBLOCK>>>
        (
         tmp.getDevData(),
         state->gpd.qs(state->gpd.activeIdx()),
         nAtoms,
         state->devManager.prop.warpSize,
         SumSingle());

    tmp.dataToHost();   
    total_Q=sqrt(conversion)*tmp.h_data[0]/state->nPerRingPoly;   
    
    std::cout<<"total_Q "<<total_Q<<'\n';
    std::cout<<"total_Q2 "<<total_Q2<<'\n';
}
double FixChargeEwald::find_optimal_parameters(bool printError){

    int nAtoms = state->atoms.size();    
    L=state->boundsGPU.trace();
    h=make_real3(L.x/sz.x,L.y/sz.y,L.z/sz.z);
//     std::cout<<"Lx "<<L.x<<'\n';
//     std::cout<<"hx "<<h.x<<'\n';
//     std::cout<<"nA "<<nAtoms<<'\n';

//now root solver 
//solving DeltaF_k=DeltaF_real
//        Log(DeltaF_k)=Log(DeltaF_real)
//        Log(DeltaF_k)-Log(DeltaF_real)=0

//lets try secant
    //two initial points
    double x_a=0.0;
    double x_b=4.79853/r_cut;
    
    double y_a=DeltaF_k(x_a)-DeltaF_real(x_a);
    double y_b=DeltaF_k(x_b)-DeltaF_real(x_b);
//           std::cout<<x_a<<' '<<y_a<<'\n';
//           std::cout<<x_b<<' '<<y_b<<' '<<DeltaF_real(x_b)<<'\n';

    double tol=1E-5;
    int n_iter=0,max_iter=100;
    while((fabs(y_b)/DeltaF_real(x_b)>tol)&&(n_iter<max_iter)){
      double kinv=(x_b-x_a)/(y_b-y_a);
      y_a=y_b;
      x_a=x_b;
      x_b=x_a-y_a*kinv;
      y_b=DeltaF_k(x_b)-DeltaF_real(x_b);
//       std::cout<<x_b<<' '<<y_b<<'\n';
      n_iter++;
    }
    if (n_iter==max_iter) std::cout<<"Ewald RMS Root finder failed, max_iter "<<max_iter<<" reached\n";
    alpha=x_b;
    setEvalWrapper();
    //set orig!
    //alpha = 1.0;
    double error = DeltaF_k(alpha)+DeltaF_real(alpha);
    if (printError) {

        std::cout<<"Ewald alpha="<<alpha<<'\n';
        std::cout<<"Ewald RMS error is  "<<error<<'\n';
    }
    return error;
    
    
}

void FixChargeEwald::setParameters(int szx_,int szy_,int szz_,real rcut_,int interpolation_order_)
{
    //for now support for only 2^N sizes
    //TODO generalize for non cubic boxes
    if (rcut_==-1) {
        rcut_ = state->rCut;
    }
    if ((szx_!=32)&&(szx_!=64)&&(szx_!=128)&&(szx_!=256)&&(szx_!=512)&&(szx_!=1024)){
        std::cout << szx_ << " is not supported, sorry. Only 2^N grid size works for charge Ewald\n";
        exit(2);
    }
    if ((szy_!=32)&&(szy_!=64)&&(szy_!=128)&&(szy_!=256)&&(szy_!=512)&&(szy_!=1024)){
        std::cout << szy_ << " is not supported, sorry. Only 2^N grid size works for charge Ewald\n";
        exit(2);
    }
    if ((szz_!=32)&&(szz_!=64)&&(szz_!=128)&&(szz_!=256)&&(szz_!=512)&&(szz_!=1024)){
        std::cout << szz_ << " is not supported, sorry. Only 2^N grid size works for charge Ewald\n";
        exit(2);
    }
    sz=make_int3(szx_,szy_,szz_);
    r_cut=rcut_;

#ifdef DASH_DOUBLE
    cudaMalloc((void**)&FFT_Qs, sizeof(cufftDoubleComplex)*sz.x*sz.y*sz.z);

    cufftPlan3d(&plan, sz.x,sz.y, sz.z, CUFFT_Z2Z);

    
    cudaMalloc((void**)&FFT_Ex, sizeof(cufftDoubleComplex)*sz.x*sz.y*sz.z);
    cudaMalloc((void**)&FFT_Ey, sizeof(cufftDoubleComplex)*sz.x*sz.y*sz.z);
    cudaMalloc((void**)&FFT_Ez, sizeof(cufftDoubleComplex)*sz.x*sz.y*sz.z);
   
#else

    cudaMalloc((void**)&FFT_Qs, sizeof(cufftComplex)*sz.x*sz.y*sz.z);

    cufftPlan3d(&plan, sz.x,sz.y, sz.z, CUFFT_C2C);

    
    cudaMalloc((void**)&FFT_Ex, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
    cudaMalloc((void**)&FFT_Ey, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
    cudaMalloc((void**)&FFT_Ez, sizeof(cufftComplex)*sz.x*sz.y*sz.z);

#endif
    Green_function=GPUArrayGlobal<real>(sz.x*sz.y*sz.z);
    CUT_CHECK_ERROR("setParameters execution failed");
    

    interpolation_order=interpolation_order_;

    malloced = true;

}


void FixChargeEwald::setGridToErrorTolerance(bool printMsg) {
    int3 szOld = sz;
    int nTries = 0;
    double error = find_optimal_parameters(false);
    Vector trace = state->bounds.rectComponents;
    while (nTries < 100 and (error > errorTolerance or error!=error or error < 0)) { //<0 tests for -inf
        Vector sVec = Vector(make_real3(sz));
        Vector ratio = sVec / trace;
        double minRatio = ratio[0];
        int minIdx = 0;
        for (int i=0; i<3; i++) {
            if (ratio[i] < minRatio) {
                minRatio = ratio[i];
                minIdx = i;
            }
        }
        sVec[minIdx] *= 2;
        //sz *= 2;//make_int3(sVec.asreal3());
        sz = make_int3(sVec.asreal3());
        error = find_optimal_parameters(false);
        nTries++;
    }
    //DOESN'T REDUCE GRID SIZE EVER
    if (printMsg) {
        printf("Using ewald grid of %d %d %d with interpolation order %d and error %f\n", sz.x, sz.y, sz.z,interpolation_order, error);
    }

    if (!malloced or szOld != sz) {
        if (malloced) {
            cufftDestroy(plan);
            cudaFree(FFT_Qs);
            cudaFree(FFT_Ex);
            cudaFree(FFT_Ey);
            cudaFree(FFT_Ez);
        }

#ifdef DASH_DOUBLE
        cudaMalloc((void**)&FFT_Qs, sizeof(cufftDoubleComplex)*sz.x*sz.y*sz.z);
        cufftPlan3d(&plan, sz.x,sz.y, sz.z, CUFFT_Z2Z);


        cudaMalloc((void**)&FFT_Ex, sizeof(cufftDoubleComplex)*sz.x*sz.y*sz.z);
        cudaMalloc((void**)&FFT_Ey, sizeof(cufftDoubleComplex)*sz.x*sz.y*sz.z);
        cudaMalloc((void**)&FFT_Ez, sizeof(cufftDoubleComplex)*sz.x*sz.y*sz.z);
#else

        cudaMalloc((void**)&FFT_Qs, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
        cufftPlan3d(&plan, sz.x,sz.y, sz.z, CUFFT_C2C);


        cudaMalloc((void**)&FFT_Ex, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
        cudaMalloc((void**)&FFT_Ey, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
        cudaMalloc((void**)&FFT_Ez, sizeof(cufftComplex)*sz.x*sz.y*sz.z);


#endif
        Green_function=GPUArrayGlobal<real>(sz.x*sz.y*sz.z);
        malloced = true;
    }


}
void FixChargeEwald::setError(double targetError, real rcut_, int interpolation_order_) {
    if (rcut_==-1) {
        rcut_ = state->rCut;
    }
    r_cut=rcut_;
    interpolation_order=interpolation_order_;
    errorTolerance = targetError;
    modeIsError = true;

}

void FixChargeEwald::calc_Green_function(){

    
    dim3 dimBlock(8,8,8);
    dim3 dimGrid((sz.x + dimBlock.x - 1) / dimBlock.x,(sz.y + dimBlock.y - 1) / dimBlock.y,(sz.z + dimBlock.z - 1) / dimBlock.z);    
    int sum_limits=int(alpha*pow(h.x*h.y*h.z,1.0/3.0)/3.14159*(sqrt(-log(10E-7))))+1;
    Green_function_cu<<<dimGrid, dimBlock>>>(state->boundsGPU, sz,Green_function.getDevData(),alpha,
                                             sum_limits,interpolation_order);//TODO parameters unknown
    CUT_CHECK_ERROR("Green_function_cu kernel execution failed");
    
        //test area
//     Green_function.dataToHost();
//     ofstream ofs;
//     ofs.open("test_Green_function.dat",ios::out );
//     for(int i=0;i<sz.x;i++)
//             for(int j=0;j<sz.y;j++){
//                 for(int k=0;k<sz.z;k++){
//                     std::cout<<Green_function.h_data[i*sz.y*sz.z+j*sz.z+k]<<'\t';
//                     ofs<<Green_function.h_data[i*sz.y*sz.z+j*sz.z+k]<<'\t';
//                 }
//                 ofs<<'\n';
//                 std::cout<<'\n';
//             }
//     ofs.close();

}


#ifdef DASH_DOUBLE
void FixChargeEwald::calc_potential(cufftDoubleComplex *phi_buf){
#else
void FixChargeEwald::calc_potential(cufftComplex *phi_buf){
#endif
     BoundsGPU b=state->boundsGPU;
    real volume=b.volume();
    
    dim3 dimBlock(8,8,8);
    dim3 dimGrid((sz.x + dimBlock.x - 1) / dimBlock.x,(sz.y + dimBlock.y - 1) / dimBlock.y,(sz.z + dimBlock.z - 1) / dimBlock.z);    
    potential_cu<<<dimGrid, dimBlock>>>(sz,Green_function.getDevData(), FFT_Qs,phi_buf);
    CUT_CHECK_ERROR("potential_cu kernel execution failed");    

#ifdef DASH_DOUBLE
    cufftExecZ2Z(plan, phi_buf, phi_buf,  CUFFT_INVERSE);
#else 
    cufftExecC2C(plan, phi_buf, phi_buf,  CUFFT_INVERSE);
#endif
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("cufftExecC2C (Z2Z) execution failed");

//     //test area
//     real *buf=new real[sz.x*sz.y*sz.z*2];
//     cudaMemcpy((void *)buf,phi_buf,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
//     ofstream ofs;
//     ofs.open("test_phi.dat",ios::out );
//     for(int i=0;i<sz.x;i++)
//             for(int j=0;j<sz.y;j++){
//                 for(int k=0;k<sz.z;k++){
//                     std::cout<<buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
//                      ofs<<buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
//                 }
//                 ofs<<'\n';
//                 std::cout<<'\n';
//             }
//     ofs.close();
//     delete []buf;
}

bool FixChargeEwald::prepareForRun() {
    if (!malloced)  setParameters(sz.x, sz.y, sz.z, r_cut,interpolation_order);
    virialField = GPUArrayDeviceGlobal<Virial>(1);
    setTotalQ2();
    //TODO these values for comparison are uninitialized - we should see about this


    handleBoundsChangeInternal(true,true);
    turnInit = state->turn;
    if (longRangeInterval != 1) {
        storedForces = GPUArrayDeviceGlobal<real4>(state->maxIdExisting+1);
    } else {
        storedForces = GPUArrayDeviceGlobal<real4>(1);
    }
    if (state->nPerRingPoly > 1) { 
        rpCentroids = GPUArrayDeviceGlobal<real4>(state->atoms.size() / state->nPerRingPoly);
    }
    setEvalWrapper();
    prepared = true;
    return prepared;
}

void FixChargeEwald::setEvalWrapper() {
    if (evalWrapperMode == "offload") {
        if (hasOffloadedChargePairCalc) {
            evalWrap = pickEvaluator<EvaluatorNone, 1, false>(EvaluatorNone(), nullptr); //nParams arg is 1 rather than zero b/c can't have zero sized argument on device
        } else {
            evalWrap = pickEvaluator<EvaluatorNone, 1, false>(EvaluatorNone(), this);
        }
    } else if (evalWrapperMode == "self") {
        evalWrap = pickEvaluator<EvaluatorNone, 1, false>(EvaluatorNone(), this);
    }

}


void FixChargeEwald::handleBoundsChange() {
    handleBoundsChangeInternal(false);
}

void FixChargeEwald::handleBoundsChangeInternal(bool printError, bool forceChange) {

    // just to avoid comparing uninitialized value (boundsLastOptimize, on ::prepareForRun())
    if (forceChange) {
        if (modeIsError) {
            setGridToErrorTolerance(printError);
        } else {
            find_optimal_parameters(printError);
        }
        calc_Green_function();
        boundsLastOptimize = state->boundsGPU;
        total_Q2LastOptimize=total_Q2;
        return;
    }

    if ((state->boundsGPU != boundsLastOptimize)||(total_Q2!=total_Q2LastOptimize)) {
        if (modeIsError) {
            setGridToErrorTolerance(printError);
        } else {
            find_optimal_parameters(printError);
        }
        calc_Green_function();
        boundsLastOptimize = state->boundsGPU;
        total_Q2LastOptimize=total_Q2;
    }
}

void FixChargeEwald::compute(int virialMode) {
 //   CUT_CHECK_ERROR("before FixChargeEwald kernel execution failed");

//     std::cout<<"FixChargeEwald::compute..\n";
    int nAtoms       = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int nRingPoly    = nAtoms / nPerRingPoly;
    GPUData &gpd     = state->gpd;
    GridGPU &grid    = state->gridGPU;
    int activeIdx    = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    
 
    real Qconversion = sqrt(state->units.qqr_to_eng);

    //first update grid from atoms positions
    //set qs to 0
    dim3 dimBlock(8,8,8);
    // e.g., sz ~ (64,64,64)... so, 64 + 8 - 1 / 8, (64 + 8 - 1) / 8
    dim3 dimGrid((sz.x + dimBlock.x - 1) / dimBlock.x,(sz.y + dimBlock.y - 1) / dimBlock.y,(sz.z + dimBlock.z - 1) / dimBlock.z);    
    if (not ((state->turn - turnInit) % longRangeInterval)) {
        map_charge_set_to_zero_cu<<<dimGrid, dimBlock>>>(sz,FFT_Qs);
        //  CUT_CHECK_ERROR("map_charge_set_to_zero_cu kernel execution failed");

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // Compute centroids of all ring polymers for use on grid
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        real4 *centroids;
        BoundsGPU bounds         = state->boundsGPU;
        BoundsGPU boundsUnskewed = bounds.unskewed();
        if (nPerRingPoly >1) {
            computeCentroid<<<NBLOCK(nRingPoly),PERBLOCK>>>(rpCentroids.data(),gpd.xs(activeIdx),nAtoms,nPerRingPoly,boundsUnskewed);
            centroids = rpCentroids.data();
        } else {
            centroids = gpd.xs(activeIdx);
        }
        switch (interpolation_order){
            case 1:{map_charge_to_grid_order_1_cu
                       <<<NBLOCK(nRingPoly), PERBLOCK>>>( nRingPoly, nPerRingPoly, 
                               centroids,                                                      
                               gpd.qs(activeIdx),
                               state->boundsGPU,
                               sz,
                               (real *)FFT_Qs,
                               Qconversion);
                       break;}
            case 3:{map_charge_to_grid_order_3_cu
                       <<<NBLOCK(nRingPoly), PERBLOCK>>>(nRingPoly, nPerRingPoly,
                               centroids,
                               gpd.qs(activeIdx),
                               state->boundsGPU,
                               sz,
                               (real *)FFT_Qs,
                               Qconversion);
                       break;}
            case 5:{map_charge_to_grid_order_5_cu
                       <<<NBLOCK(nRingPoly), PERBLOCK>>>(nRingPoly, nPerRingPoly,
                               centroids,
                               gpd.qs(activeIdx),
                               state->boundsGPU,
                               sz,
                               (real *)FFT_Qs,
                               Qconversion);
                       break;}
            case 7:{map_charge_to_grid_order_7_cu
                       <<<NBLOCK(nRingPoly), PERBLOCK>>>(nRingPoly, nPerRingPoly,
                               centroids,
                               gpd.qs(activeIdx),
                               state->boundsGPU,
                               sz,
                               (real *)FFT_Qs,
                               Qconversion);
                       break;}
        }    
        // CUT_CHECK_ERROR("map_charge_to_grid_cu kernel execution failed");
#ifdef DASH_DOUBLE
        cufftExecZ2Z(plan, FFT_Qs, FFT_Qs, CUFFT_FORWARD);
#else
        cufftExecC2C(plan, FFT_Qs, FFT_Qs, CUFFT_FORWARD);

#endif
        // cudaDeviceSynchronize();
        //  CUT_CHECK_ERROR("cufftExecC2C Qs execution failed");


        //     //test area
        //     real buf[sz.x*sz.y*sz.z*2];
        //     cudaMemcpy(buf,FFT_Qs,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
        //     ofstream ofs;
        //     ofs.open("test_FFT.dat",ios::out );
        //     for(int i=0;i<sz.x;i++)
        //             for(int j=0;j<sz.y;j++){
        //                 for(int k=0;k<sz.z;k++){
        //                     std::cout<<buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]<<'\t';
        //                     ofs <<buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]<<'\t';
        //                 }
        //                 ofs<<'\n';
        //                 std::cout<<'\n';
        //             }


        //next potential calculation: just going to use Ex to store it for now
        //       calc_potential(FFT_Ex);

        //calc E field
        E_field_cu<<<dimGrid, dimBlock>>>(state->boundsGPU,sz,Green_function.getDevData(), FFT_Qs,FFT_Ex,FFT_Ey,FFT_Ez);
        CUT_CHECK_ERROR("E_field_cu kernel execution failed");    

#ifdef DASH_DOUBLE
        cufftExecZ2Z(plan, FFT_Ex, FFT_Ex,  CUFFT_INVERSE);
        cufftExecZ2Z(plan, FFT_Ey, FFT_Ey,  CUFFT_INVERSE);
        cufftExecZ2Z(plan, FFT_Ez, FFT_Ez,  CUFFT_INVERSE);
#else
        cufftExecC2C(plan, FFT_Ex, FFT_Ex,  CUFFT_INVERSE);
        cufftExecC2C(plan, FFT_Ey, FFT_Ey,  CUFFT_INVERSE);
        cufftExecC2C(plan, FFT_Ez, FFT_Ez,  CUFFT_INVERSE);
#endif
        //  cudaDeviceSynchronize();
        // CUT_CHECK_ERROR("cufftExecC2C  E_field execution failed");


        /*//test area
          Bounds b=state->bounds;
          real volume=b.trace[0]*b.trace[1]*b.trace[2];    
          real *buf=new real[sz.x*sz.y*sz.z*2];
          cudaMemcpy((void *)buf,FFT_Ex,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
          ofstream ofs;
          ofs.open("test_Ex.dat",ios::out );
          for(int i=0;i<sz.x;i++)
          for(int j=0;j<sz.y;j++){
          for(int k=0;k<sz.z;k++){
          std::cout<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
          ofs<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
          }
          ofs<<'\n';
          std::cout<<'\n';
          }
          ofs.close();
          cudaMemcpy((void *)buf,FFT_Ey,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
          ofs.open("test_Ey.dat",ios::out );
          for(int i=0;i<sz.x;i++)
          for(int j=0;j<sz.y;j++){
          for(int k=0;k<sz.z;k++){
          std::cout<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
          ofs<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
          }
          ofs<<'\n';
          std::cout<<'\n';
          }
          ofs.close();    
          cudaMemcpy((void *)buf,FFT_Ez,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
          ofs.open("test_Ez.dat",ios::out );
          for(int i=0;i<sz.x;i++)
          for(int j=0;j<sz.y;j++){
          for(int k=0;k<sz.z;k++){
          std::cout<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
          ofs<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
          }
          ofs<<'\n';
          std::cout<<'\n';
          }
          ofs.close();    
          delete []buf;   */ 




        //calc forces
        //printf("Forces!\n");
        // Performing an "effective" ring polymer contraction means that we should evaluate the forces
        // for the centroids
        bool storeForces = longRangeInterval != 1;
        switch (interpolation_order){
            case 1:{Ewald_long_range_forces_order_1_cu<<<NBLOCK(nRingPoly), PERBLOCK>>>( nRingPoly, nPerRingPoly,
                           centroids,                                                      
                           gpd.fs(activeIdx),
                           gpd.qs(activeIdx),
                           state->boundsGPU,
                           sz,
                           FFT_Ex,FFT_Ey,FFT_Ez,Qconversion,
                           storeForces, gpd.ids(activeIdx), storedForces.data()
                           );
                       break;}
            case 3:{Ewald_long_range_forces_order_3_cu<<<NBLOCK(nRingPoly), PERBLOCK>>>( nRingPoly, nPerRingPoly,
                           centroids,                                                      
                           gpd.fs(activeIdx),
                           gpd.qs(activeIdx),
                           state->boundsGPU,
                           sz,
                           FFT_Ex,FFT_Ey,FFT_Ez,Qconversion,
                           storeForces, gpd.ids(activeIdx), storedForces.data()
                           );

                       break;}
            case 5:{Ewald_long_range_forces_order_5_cu<<<NBLOCK(nRingPoly), PERBLOCK>>>( nRingPoly, nPerRingPoly,
                           centroids,                                                      
                           gpd.fs(activeIdx),
                           gpd.qs(activeIdx),
                           state->boundsGPU,
                           sz,
                           FFT_Ex,FFT_Ey,FFT_Ez,Qconversion,
                           storeForces, gpd.ids(activeIdx), storedForces.data()
                           );

                       break;}
            case 7:{Ewald_long_range_forces_order_7_cu<<<NBLOCK(nRingPoly), PERBLOCK>>>( nRingPoly, nPerRingPoly,
                           centroids,                                                      
                           gpd.fs(activeIdx),
                           gpd.qs(activeIdx),
                           state->boundsGPU,
                           sz,
                           FFT_Ex,FFT_Ey,FFT_Ez,Qconversion,
                           storeForces, gpd.ids(activeIdx), storedForces.data()
                           );

                       break;}
        }
    } else {
        applyStoredForces<<<NBLOCK(nAtoms), PERBLOCK>>>( nAtoms,
                gpd.fs(activeIdx),
                gpd.ids(activeIdx), storedForces.data());
    }
    CUT_CHECK_ERROR("Ewald_long_range_forces_cu  execution failed");
    //SHORT RANGE
    if (virialMode) {
        int warpSize = state->devManager.prop.warpSize;
        BoundsGPU &b=state->boundsGPU;
        real volume=b.volume();          
        virialField.memset(0); 
        virials_cu<<<dimGrid, dimBlock,sizeof(Virial)*dimBlock.x*dimBlock.y*dimBlock.z>>>(state->boundsGPU,sz,virialField.data(),alpha,Green_function.getDevData(), FFT_Qs, warpSize); 
        CUT_CHECK_ERROR("virials_cu kernel execution failed");    



        mapVirialToSingleAtom<<<1, 6>>>(gpd.virials.d_data.data(), virialField.data(), volume);
    }

    real *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->compute(nAtoms,nPerRingPoly,gpd.xs(activeIdx), gpd.fs(activeIdx),
                  neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                  state->devManager.prop.warpSize, nullptr, 0, state->boundsGPU, //PASSING NULLPTR TO GPU MAY CAUSE ISSUES
    //ALTERNATIVELy, COULD JUST GIVE THE PARMS SOME OTHER RANDOM POINTER, AS LONG AS IT'S VALID
                  neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), gpd.qs(activeIdx), r_cut, virialMode, nThreadPerBlock(), nThreadPerAtom());


    CUT_CHECK_ERROR("Ewald_short_range_forces_cu  execution failed");

}


void FixChargeEwald::singlePointEng(real * perParticleEng) {
    CUT_CHECK_ERROR("before FixChargeEwald kernel execution failed");

    if (state->boundsGPU != boundsLastOptimize) {
        handleBoundsChange();
    }
//     std::cout<<"FixChargeEwald::compute..\n";
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int nRingPoly    = nAtoms / nPerRingPoly;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    
    
     
    real Qconversion = sqrt(state->units.qqr_to_eng);


    //first update grid from atoms positions
    //set qs to 0
    real field_energy_per_particle = 0;
    dim3 dimBlock(8,8,8);
    dim3 dimGrid((sz.x + dimBlock.x - 1) / dimBlock.x,(sz.y + dimBlock.y - 1) / dimBlock.y,(sz.z + dimBlock.z - 1) / dimBlock.z);    
    map_charge_set_to_zero_cu<<<dimGrid, dimBlock>>>(sz,FFT_Qs);
    CUT_CHECK_ERROR("map_charge_set_to_zero_cu kernel execution failed");
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Compute centroids of all ring polymers for use on grid
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    real4 *centroids;
    BoundsGPU bounds         = state->boundsGPU;
    BoundsGPU boundsUnskewed = bounds.unskewed();
    if (nPerRingPoly >1) {
        rpCentroids = GPUArrayDeviceGlobal<real4>(nRingPoly);
        computeCentroid<<<NBLOCK(nRingPoly),PERBLOCK>>>(rpCentroids.data(),gpd.xs(activeIdx),nAtoms,nPerRingPoly,boundsUnskewed);
        centroids = rpCentroids.data();
        periodicWrapCpy<<<NBLOCK(nRingPoly), PERBLOCK>>>(centroids, nRingPoly, boundsUnskewed);
    } else {
        centroids = gpd.xs(activeIdx);
    }

      switch (interpolation_order){
      case 1:{map_charge_to_grid_order_1_cu
              <<<NBLOCK(nRingPoly), PERBLOCK>>>(nRingPoly, nPerRingPoly,
                                              centroids,
                                              gpd.qs(activeIdx),
                                              state->boundsGPU,
                                              sz,
                                              (real *)FFT_Qs,Qconversion);
              break;}
      case 3:{map_charge_to_grid_order_3_cu
              <<<NBLOCK(nRingPoly), PERBLOCK>>>(nRingPoly, nPerRingPoly, 
                                              centroids,
                                              gpd.qs(activeIdx),
                                              state->boundsGPU,
                                              sz,
                                              (real *)FFT_Qs,Qconversion);
              break;}
      case 5:{map_charge_to_grid_order_5_cu
              <<<NBLOCK(nRingPoly), PERBLOCK>>>(nRingPoly, nPerRingPoly, 
                                              centroids,
                                              gpd.qs(activeIdx),
                                              state->boundsGPU,
                                              sz,
                                              (real *)FFT_Qs,Qconversion);
              break;}
      case 7:{map_charge_to_grid_order_7_cu
              <<<NBLOCK(nRingPoly), PERBLOCK>>>(nRingPoly, nPerRingPoly, 
                                              centroids,
                                              gpd.qs(activeIdx),
                                              state->boundsGPU,
                                              sz,
                                              (real *)FFT_Qs,Qconversion);
              break;}
    }    
    CUT_CHECK_ERROR("map_charge_to_grid_cu kernel execution failed");

#ifdef DASH_DOUBLE
    cufftExecZ2Z(plan, FFT_Qs, FFT_Qs, CUFFT_FORWARD);
#else 
    cufftExecC2C(plan, FFT_Qs, FFT_Qs, CUFFT_FORWARD);
#endif
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("cufftExecC2C Qs execution failed");

    

    //calc field energy 
    BoundsGPU &b=state->boundsGPU;
    real volume=b.volume();
    
    Energy_cu<<<dimGrid, dimBlock>>>(sz,Green_function.getDevData(), FFT_Qs,FFT_Ex);//use Ex as buffer
    CUT_CHECK_ERROR("Energy_cu kernel execution failed");    
  
    GPUArrayGlobal<real>field_E(1);
    field_E.memsetByVal(0.0);
    int warpSize = state->devManager.prop.warpSize;
    accumulate_gpu<real,real, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(2*sz.x*sz.y*sz.z/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(real)*PERBLOCK>>>
        (
         field_E.getDevData(),
         (real *)FFT_Ex,
         2*sz.x*sz.y*sz.z,
         warpSize,
         SumSingle()
         );   
/*
    sumSingle<real,real, N_DATA_PER_THREAD> <<<NBLOCK(2*sz.x*sz.y*sz.z/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(real)*PERBLOCK>>>(
                                            field_E.getDevData(),
                                            (real *)FFT_Ex,
                                            2*sz.x*sz.y*sz.z,
                                            warpSize);   
                                            */
    field_E.dataToHost();

    //field_energy_per_particle=0.5*field_E.h_data[0]/volume/nAtoms;
    field_energy_per_particle=0.5*field_E.h_data[0]/volume/nRingPoly;
//         std::cout<<"field_E "<<field_E.h_data[0]<<'\n';

    field_energy_per_particle-=alpha/sqrt(M_PI)*total_Q2/nRingPoly;
//      std::cout<<"self correction "<<alpha/sqrt(M_PI)*total_Q2<<'\n';

//pair energies
    mapEngToParticles<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, field_energy_per_particle, perParticleEng);
    real *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->energy(nAtoms,nPerRingPoly, gpd.xs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, nullptr, 0, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), r_cut, nThreadPerBlock(), nThreadPerAtom());


    CUT_CHECK_ERROR("Ewald_short_range_forces_cu  execution failed");

}


int FixChargeEwald::setLongRangeInterval(int interval) {
    if (interval) {
        longRangeInterval = interval;
    }
    return longRangeInterval;
}



ChargeEvaluatorEwald FixChargeEwald::generateEvaluator() {
    return ChargeEvaluatorEwald(alpha, state->units.qqr_to_eng);
}

void (FixChargeEwald::*setParameters_xyz)(int ,int ,int ,real ,int) = &FixChargeEwald::setParameters;
void (FixChargeEwald::*setParameters_xxx)(int ,real ,int) = &FixChargeEwald::setParameters;
void export_FixChargeEwald() {
    py::class_<FixChargeEwald,
                          SHARED(FixChargeEwald),
                          py::bases<FixCharge> > (
         "FixChargeEwald", 
         py::init<SHARED(State), std::string, std::string> (
              py::args("state", "handle", "groupHandle"))
        )
        .def("setParameters", setParameters_xyz,
                (py::arg("szx"),py::arg("szy"),py::arg("szz"), py::arg("r_cut")=-1,py::arg("interpolation_order")=5)
          
            )
        .def("setParameters", setParameters_xxx,
                (py::arg("sz"),py::arg("r_cut")=-1,py::arg("interpolation_order")=5)
            )        
        .def("setError", &FixChargeEwald::setError, (py::arg("error"), py::arg("rCut")=-1, py::arg("interpolation_order")=5)
            )
        .def("setLongRangeInterval", &FixChargeEwald::setLongRangeInterval, (py::arg("interval")=0))
        ;
}

