#include "FixChargeEwald.h"
#include "cutils_func.h"
#include <cufft.h>
#include "cuda_call.h"
#include <fstream>

// #include <cmath>
using namespace std;

// #define THREADS_PER_BLOCK_


//different implementation for different interpolation orders
//TODO template
//order 1 nearest point
__global__ void map_charge_to_grid_order_1_cu(int nAtoms, float4 *xs,  float *qs,  BoundsGPU bounds,
                                      int3 sz,float *grid/*convert to float for cufffComplex*/) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 posWhole = xs[idx];
        float3 pos = make_float3(posWhole)-bounds.lo;

        float qi = qs[idx];
        
        //find nearest grid point
        float3 h=bounds.trace()/make_float3(sz);
        int3 nearest_grid_point=make_int3(pos/h);//TODO looks unsafe. should round down
        //or
        int3 p=nearest_grid_point;
        p.x-=p.x>=sz.x? sz.x:0;
        p.y-=p.y>=sz.y? sz.y:0;
        p.z-=p.z>=sz.z? sz.z:0;
        atomicAdd(&grid[p.x*sz.y*sz.z*2+p.y*sz.z*2+p.z*2], 1.0*qi);
    }
}

inline __host__ __device__ float W_p_3(int i,float x){
    if (i==-1) return 0.125-0.5*x+0.5*x*x;
    if (i== 0) return 0.75-x*x;
    /*if (i== 1)*/ return 0.125+0.5*x+0.5*x*x;
}


__global__ void map_charge_to_grid_order_3_cu(int nAtoms, float4 *xs,  float *qs,  BoundsGPU bounds,
                                      int3 sz,float *grid/*convert to float for cufffComplex*/) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 posWhole = xs[idx];
        float3 pos = make_float3(posWhole)-bounds.lo;

        float qi = qs[idx];
        
        //find nearest grid point
        float3 h=bounds.trace()/make_float3(sz);
        int3 nearest_grid_point=make_int3(pos/h);//TODO looks unsafe. should round down or handle for actual grid point where assignment happens
        
        //distance from nearest_grid_point /h
        float3 d=pos/h-make_float3(nearest_grid_point);
        
        int3 p=nearest_grid_point;
        for (int ix=-1;ix<=1;ix++){
          p.x=nearest_grid_point.x+ix;
          float charge_yz_w=qi*W_p_3(ix,d.x);
          for (int iy=-1;iy<=1;iy++){
            p.y=nearest_grid_point.y+iy;
            float charge_z_w=charge_yz_w*W_p_3(iy,d.y);
            for (int iz=-1;iz<=1;iz++){
                p.z=nearest_grid_point.z+iz;
                float charge_w=charge_z_w*W_p_3(iz,d.z);
                p.x-= p.x>=sz.x? sz.x : 0;
                p.y-= p.y>=sz.y? sz.y : 0;
                p.z-= p.z>=sz.z? sz.z : 0;
                p.x+= p.x<0    ? sz.x : 0;
                p.y+= p.y<0    ? sz.y : 0;
                p.z+= p.z<0    ? sz.z : 0;
                atomicAdd(&grid[p.x*sz.y*sz.z*2+p.y*sz.z*2+p.z*2], charge_w);
            }
          }
        }
    }
}


__global__ void map_charge_set_to_zero_cu(int3 sz,cufftComplex *grid) {
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.x<sz.y)&&(id.x<sz.z))                  
         grid[id.x*sz.y*sz.z+id.y*sz.z+id.z]=make_cuComplex (0.0f, 0.0f);    
}

__device__ float sinc(float x){
  if ((x<0.1)&&(x>-0.1)){
    float x2=x*x;
    return 1.0 - x2*0.16666666667f + x2*x2*0.008333333333333333f - x2*x2*x2*0.00019841269841269841f;    
  }
    else return sin(x)/x;
}

__global__ void Green_function_cu(BoundsGPU bounds, int3 sz,float *Green_function,float alpha,
                                  //now some parameter for Gf calc
                                  int sum_limits, int intrpl_order) {
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.x<sz.y)&&(id.x<sz.z)){
          float3 h =bounds.trace()/make_float3(sz);
          
          //         2*PI
          float3 k= 6.28318530717958647693f*make_float3(id)/bounds.trace();
          if (id.x>sz.x/2) k.x= 6.28318530717958647693f*(id.x-sz.x)/bounds.trace().x;
          if (id.y>sz.y/2) k.y= 6.28318530717958647693f*(id.y-sz.y)/bounds.trace().y;
          if (id.z>sz.z/2) k.z= 6.28318530717958647693f*(id.z-sz.z)/bounds.trace().z;
          

          //OK GF(k)  = 4Pi/K^2 [SumforM(W(K+M)^2  exp(-(K+M)^2/4alpha) dot(K,K+M)/(K+M^2))] / 
          //                    [SumforM^2(W(K+M)^2)]
             
             
          float sum1=0.0f;   
          float sum2=0.0f;   
          float k2=lengthSqr(k);
          if (k2!=0.0){
              for (int ix=-sum_limits;ix<=sum_limits;ix++){//TODO different limits 
                for (int iy=-sum_limits;iy<=sum_limits;iy++){
                  for (int iz=-sum_limits;iz<=sum_limits;iz++){
                      float3 kpM=k+6.28318530717958647693f*make_float3(ix,iy,iz)/h;
//                             kpM.x+=6.28318530717958647693f/h.x*ix;//TODO rewrite
//                             kpM.y+=6.28318530717958647693f/h.y*iy;
//                             kpM.z+=6.28318530717958647693f/h.z*iz;
                            float kpMlen=lengthSqr(kpM);
                            float W=sinc(kpM.x*h.x*0.5)*sinc(kpM.y*h.y*0.5)*sinc(kpM.z*h.z*0.5);
                            for(int p=1;p<intrpl_order;p++)
                                  W*=W;
    //                          W*=h;//not need- cancels out
                            float W2=W*W;
                            
                            //4*PI
                            sum1+=12.56637061435917295385*exp(-kpMlen*0.25/alpha/alpha)*dot(k,kpM)/kpMlen*W2;
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

__global__ void potential_cu(int3 sz,float *Green_function,
                                    cufftComplex *FFT_qs, cufftComplex *FFT_phi){
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.x<sz.y)&&(id.x<sz.z)){
        FFT_phi[id.x*sz.y*sz.z+id.y*sz.z+id.z]=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z]*Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z];
//TODO after Inverse FFT divide by volume
      }
}

__global__ void E_field_cu(BoundsGPU bounds, int3 sz,float *Green_function, cufftComplex *FFT_qs,
                           cufftComplex *FFT_Ex,cufftComplex *FFT_Ey,cufftComplex *FFT_Ez){
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.x<sz.y)&&(id.x<sz.z)){
          //K vector
          float3 k= 6.28318530717958647693f*make_float3(id)/bounds.trace();
          if (id.x>sz.x/2) k.x= 6.28318530717958647693f*(id.x-sz.x)/bounds.trace().x;
          if (id.y>sz.y/2) k.y= 6.28318530717958647693f*(id.y-sz.y)/bounds.trace().y;
          if (id.z>sz.z/2) k.z= 6.28318530717958647693f*(id.z-sz.z)/bounds.trace().z;        
          
          //ik*q(k)*Gf(k)
          cufftComplex Ex,Ey,Ez;
          float GF=Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z];
          cufftComplex q=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z];

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


__global__ void Ewald_long_range_forces_order_1_cu(int nAtoms, float4 *xs, float4 *fs, 
                                                   float *qs, BoundsGPU bounds,
                                                   int3 sz, cufftComplex *FFT_Ex,
                                                    cufftComplex *FFT_Ey,cufftComplex *FFT_Ez){
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 posWhole = xs[idx];
        float3 pos = make_float3(posWhole)-bounds.lo;

        float qi = qs[idx];
        
        //find nearest grid point
        float3 h=bounds.trace()/make_float3(sz);
        int3 p=make_int3(pos/h);//TODO looks unsafe. should round down
        p.x-=p.x>=sz.x? sz.x:0;
        p.y-=p.y>=sz.y? sz.y:0;
        p.z-=p.z>=sz.z? sz.z:0;
        
        //get E field
        float3 E;
        float volume=bounds.trace().x*bounds.trace().y*bounds.trace().z;
        E.x= -FFT_Ex[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
        E.y= -FFT_Ey[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
        E.z= -FFT_Ez[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
        
        float3 force=qi*E;
        fs[idx] += force;
    }
}


__global__ void Ewald_long_range_forces_order_3_cu(int nAtoms, float4 *xs, float4 *fs, 
                                                   float *qs, BoundsGPU bounds,
                                                   int3 sz, cufftComplex *FFT_Ex,
                                                    cufftComplex *FFT_Ey,cufftComplex *FFT_Ez){
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 posWhole = xs[idx];
        float3 pos = make_float3(posWhole)-bounds.lo;

        float qi = qs[idx];
        

        //find nearest grid point
        float3 h=bounds.trace()/make_float3(sz);
        int3 nearest_grid_point=make_int3(pos/h);//TODO looks unsafe. should round down or handle for actual grid point where assignment happens
        
        //distance from nearest_grid_point /h
        float3 d=pos/h-make_float3(nearest_grid_point);

        float3 E=make_float3(0,0,0);
        float volume=bounds.trace().x*bounds.trace().y*bounds.trace().z;

        int3 p=nearest_grid_point;
        for (int ix=-1;ix<=1;ix++){
          p.x=nearest_grid_point.x+ix;
          for (int iy=-1;iy<=1;iy++){
            p.y=nearest_grid_point.y+iy;
            for (int iz=-1;iz<=1;iz++){
                p.z=nearest_grid_point.z+iz;
                p.x-= p.x>=sz.x? sz.x : 0;
                p.y-= p.y>=sz.y? sz.y : 0;
                p.z-= p.z>=sz.z? sz.z : 0;
                p.x+= p.x<0    ? sz.x : 0;
                p.y+= p.y<0    ? sz.y : 0;
                p.z+= p.z<0    ? sz.z : 0;
                float3 Ep;
                float W_xyz=W_p_3(ix,d.x)*W_p_3(iy,d.y)*W_p_3(iz,d.z);
                
                Ep.x= -FFT_Ex[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                Ep.y= -FFT_Ey[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                Ep.z= -FFT_Ez[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                E+=W_xyz*Ep;
            }
          }
        }
               
        float3 force=qi*E;
        fs[idx] += force;
    }
}


__global__ void compute_short_range_forces_cu(int nAtoms, float4 *xs, float4 *fs, int *neighborCounts, uint *neighborlist, int *cumulSumMaxPerBlock, float *qs, float alpha, float rCut, BoundsGPU bounds, int warpSize, float oneFourStrength) {

    float multipliers[4] = {1, 0, 0, oneFourStrength};
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 posWhole = xs[idx];
        float3 pos = make_float3(posWhole);

        float3 forceSum = make_float3(0, 0, 0);
        float qi = qs[idx];//tex2D<float>(qs, XIDX(idx, sizeof(float)), YIDX(idx, sizeof(float)));

        //printf("start, end %d %d\n", start, end);
        int baseIdx = baseNeighlistIdx<void>(cumulSumMaxPerBlock, warpSize);
        int numNeigh = neighborCounts[idx];
        for (int i=0; i<numNeigh; i++) {
            int nlistIdx = baseIdx + warpSize * i;
            uint otherIdxRaw = neighborlist[nlistIdx];
            uint neighDist = otherIdxRaw >> 30;
            uint otherIdx = otherIdxRaw & EXCL_MASK;
            float3 otherPos = make_float3(xs[otherIdx]);
            //then wrap and compute forces!
            float3 dr = bounds.minImage(pos - otherPos);
            float lenSqr = lengthSqr(dr);
            //   printf("dist is %f %f %f\n", dr.x, dr.y, dr.z);
            if (lenSqr < rCut*rCut) {
                float multiplier = multipliers[neighDist];
                float len=sqrtf(lenSqr);
                float qj = qs[otherIdx];

                float r2inv = 1.0f/lenSqr;
                float rinv = 1.0f/len;                                   //1/Sqrt(Pi)
                float forceScalar = qi*qj*(erfcf((alpha*len))*rinv+(2.0*0.5641895835477563*alpha)*exp(-alpha*alpha*lenSqr))*r2inv* multiplier;

                
                float3 forceVec = dr * forceScalar;
                forceSum += forceVec;
            }

        }   
        fs[idx] += forceSum; //operator for float4 + float3

    }

}


FixChargeEwald::FixChargeEwald(SHARED(State) state_, string handle_, string groupHandle_): FixCharge(state_, handle_, groupHandle_, chargePairDSF),first_run(true){
//   setParameters(128,3.0);
  cufftCreate(&plan);
}


FixChargeEwald::~FixChargeEwald(){
  cufftDestroy(plan);
  cudaFree(FFT_Qs);
  cudaFree(FFT_Ex);
  cudaFree(FFT_Ey);
  cudaFree(FFT_Ez);
}


void FixChargeEwald::setParameters(int szx_,int szy_,int szz_,float rcut_,int interpolation_order_)
{
    //for now support for only 2^N sizes
    //TODO generalize for non cubic boxes
    if ((szx_!=32)||(szx_!=64)||(szx_!=128)||(szx_!=256)||(szx_!=512)||(szx_!=1024)){
        cout << szx_ << " is not supported, sorry. Only 2^N grid size works for charge Ewald\n";
    }
    if ((szy_!=32)||(szy_!=64)||(szy_!=128)||(szy_!=256)||(szy_!=512)||(szy_!=1024)){
        cout << szy_ << " is not supported, sorry. Only 2^N grid size works for charge Ewald\n";
    }
    if ((szz_!=32)||(szz_!=64)||(szz_!=128)||(szz_!=256)||(szz_!=512)||(szz_!=1024)){
        cout << szz_ << " is not supported, sorry. Only 2^N grid size works for charge Ewald\n";
    }
    sz=make_int3(szx_,szy_,szz_);
    r_cut=rcut_;
    cudaMalloc((void**)&FFT_Qs, sizeof(cufftComplex)*sz.x*sz.y*sz.z);

    cufftPlan3d(&plan, sz.x,sz.y, sz.z, CUFFT_C2C);

    
    cudaMalloc((void**)&FFT_Ex, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
    cudaMalloc((void**)&FFT_Ey, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
    cudaMalloc((void**)&FFT_Ez, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
    
    Green_function=GPUArray<float>(sz.x*sz.y*sz.z);
    CUT_CHECK_ERROR("setParameters execution failed");
    

    interpolation_order=interpolation_order_;
    //in order to find alpha we have to solve
    //Fshort(r_cut,alpha)==10^-10
    //where Fshort(r,alpha)= erfc(alpha*r)/r^2+2alpha/sqrt(pi)*exp(-alpha^2*r^2)/r
    
    //first we solve with only the  leading term exp(-alpha^2*r_cut^2)====10^-10
    //which gives us  alpha=4.79853/r_cut
    alpha=4.79853/r_cut;
    //second TODO couple of iterations of Newton root finder
    cout<<"Ewald alpha="<<alpha<<'\n';
}


void FixChargeEwald::calc_Green_function(){

    
    dim3 dimBlock(8,8,8);
    dim3 dimGrid((sz.x + dimBlock.x - 1) / dimBlock.x,(sz.y + dimBlock.y - 1) / dimBlock.y,(sz.z + dimBlock.z - 1) / dimBlock.z);    
    Green_function_cu<<<dimGrid, dimBlock>>>(state->boundsGPU, sz,Green_function.getDevData(),alpha,
                                             10,interpolation_order);//TODO parameters unknown
    CUT_CHECK_ERROR("Green_function_cu kernel execution failed");
    
        //test area
//     Green_function.dataToHost();
//     ofstream ofs;
//     ofs.open("test_Green_function.dat",ios::out );
//     for(int i=0;i<sz.x;i++)
//             for(int j=0;j<sz.y;j++){
//                 for(int k=0;k<sz.z;k++){
//                     cout<<Green_function.h_data[i*sz.y*sz.z+j*sz.z+k]<<'\t';
//                     ofs<<Green_function.h_data[i*sz.y*sz.z+j*sz.z+k]<<'\t';
//                 }
//                 ofs<<'\n';
//                 cout<<'\n';
//             }
//     ofs.close();

}


void FixChargeEwald::calc_potential(cufftComplex *phi_buf){
     Bounds b=state->bounds;
    float volume=b.trace[0]*b.trace[1]*b.trace[2];
    
    dim3 dimBlock(8,8,8);
    dim3 dimGrid((sz.x + dimBlock.x - 1) / dimBlock.x,(sz.y + dimBlock.y - 1) / dimBlock.y,(sz.z + dimBlock.z - 1) / dimBlock.z);    
    potential_cu<<<dimGrid, dimBlock>>>(sz,Green_function.getDevData(), FFT_Qs,phi_buf);
    CUT_CHECK_ERROR("potential_cu kernel execution failed");    


    cufftExecC2C(plan, phi_buf, phi_buf,  CUFFT_INVERSE);
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("cufftExecC2C execution failed");

    //test area
    float *buf=new float[sz.x*sz.y*sz.z*2];
    cudaMemcpy((void *)buf,phi_buf,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
    ofstream ofs;
    ofs.open("test_phi.dat",ios::out );
    for(int i=0;i<sz.x;i++)
            for(int j=0;j<sz.y;j++){
                for(int k=0;k<sz.z;k++){
                    cout<<buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
                    ofs<<buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
                }
                ofs<<'\n';
                cout<<'\n';
            }
    ofs.close();
    delete []buf;
}



void FixChargeEwald::compute() {
    CUT_CHECK_ERROR("before FixChargeEwald kernel execution failed");

    cout<<"FixChargeEwald::compute..\n";
    int nAtoms = state->atoms.size();
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx;
    int *neighborCounts = grid.perAtomArray.d_data.ptr;
    
    if (first_run){
        first_run=false;
        calc_Green_function();
    }
    
 

    //first update grid from atoms positions
    //set qs to 0
    dim3 dimBlock(8,8,8);
    dim3 dimGrid((sz.x + dimBlock.x - 1) / dimBlock.x,(sz.y + dimBlock.y - 1) / dimBlock.y,(sz.z + dimBlock.z - 1) / dimBlock.z);    
    map_charge_set_to_zero_cu<<<dimGrid, dimBlock>>>(sz,FFT_Qs);
    
      switch (interpolation_order){
      case 1:{map_charge_to_grid_order_1_cu
              <<<NBLOCK(nAtoms), PERBLOCK>>>( nAtoms,
                                              gpd.xs(activeIdx),                                                      
                                              gpd.qs(activeIdx),
                                              state->boundsGPU,
                                              sz,
                                              (float *)FFT_Qs);
              break;}
      case 3:{map_charge_to_grid_order_3_cu
              <<<NBLOCK(nAtoms), PERBLOCK>>>( nAtoms,
                                              gpd.xs(activeIdx),                                                      
                                              gpd.qs(activeIdx),
                                              state->boundsGPU,
                                              sz,
                                              (float *)FFT_Qs);
              break;}
    }    
    CUT_CHECK_ERROR("map_charge_to_grid_cu kernel execution failed");

    cufftExecC2C(plan, FFT_Qs, FFT_Qs, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("cufftExecC2C Qs execution failed");

    
//     //test area
//     float buf[sz.x*sz.y*sz.z*2];
//     cudaMemcpy(buf,FFT_Qs,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
//     ofstream ofs;
//     ofs.open("test_FFT.dat",ios::out );
//     for(int i=0;i<sz.x;i++)
//             for(int j=0;j<sz.y;j++){
//                 for(int k=0;k<sz.z;k++){
//                     cout<<buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]<<'\t';
//                     ofs <<buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]<<'\t';
//                 }
//                 ofs<<'\n';
//                 cout<<'\n';
//             }
//     ofs.close();

    
    //next potential calculation: just going to use Ex to store it for now
//       calc_potential(FFT_Ex);

    //calc E field
    E_field_cu<<<dimGrid, dimBlock>>>(state->boundsGPU,sz,Green_function.getDevData(), FFT_Qs,FFT_Ex,FFT_Ey,FFT_Ez);
    CUT_CHECK_ERROR("E_field_cu kernel execution failed");    


    cufftExecC2C(plan, FFT_Ex, FFT_Ex,  CUFFT_INVERSE);
    cufftExecC2C(plan, FFT_Ey, FFT_Ey,  CUFFT_INVERSE);
    cufftExecC2C(plan, FFT_Ez, FFT_Ez,  CUFFT_INVERSE);
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("cufftExecC2C  E_field execution failed");
    
    
    /*//test area
     Bounds b=state->bounds;
    float volume=b.trace[0]*b.trace[1]*b.trace[2];    
    float *buf=new float[sz.x*sz.y*sz.z*2];
    cudaMemcpy((void *)buf,FFT_Ex,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
    ofstream ofs;
    ofs.open("test_Ex.dat",ios::out );
    for(int i=0;i<sz.x;i++)
            for(int j=0;j<sz.y;j++){
                for(int k=0;k<sz.z;k++){
                    cout<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
                    ofs<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
                }
                ofs<<'\n';
                cout<<'\n';
            }
    ofs.close();
    cudaMemcpy((void *)buf,FFT_Ey,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
    ofs.open("test_Ey.dat",ios::out );
    for(int i=0;i<sz.x;i++)
            for(int j=0;j<sz.y;j++){
                for(int k=0;k<sz.z;k++){
                    cout<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
                    ofs<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
                }
                ofs<<'\n';
                cout<<'\n';
            }
    ofs.close();    
    cudaMemcpy((void *)buf,FFT_Ez,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
    ofs.open("test_Ez.dat",ios::out );
    for(int i=0;i<sz.x;i++)
            for(int j=0;j<sz.y;j++){
                for(int k=0;k<sz.z;k++){
                    cout<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
                    ofs<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
                }
                ofs<<'\n';
                cout<<'\n';
            }
    ofs.close();    
    delete []buf;   */ 
    
    
    //calc forces
    switch (interpolation_order){
      case 1:{Ewald_long_range_forces_order_1_cu<<<NBLOCK(nAtoms), PERBLOCK>>>( nAtoms,
                                              gpd.xs(activeIdx),                                                      
                                              gpd.fs(activeIdx),
                                              gpd.qs(activeIdx),
                                              state->boundsGPU,
                                              sz,
                                              FFT_Ex,FFT_Ey,FFT_Ez);
              break;}
      case 3:{Ewald_long_range_forces_order_3_cu<<<NBLOCK(nAtoms), PERBLOCK>>>( nAtoms,
                                              gpd.xs(activeIdx),                                                      
                                              gpd.fs(activeIdx),
                                              gpd.qs(activeIdx),
                                              state->boundsGPU,
                                              sz,
                                              FFT_Ex,FFT_Ey,FFT_Ez);
               break;}
    }
    CUT_CHECK_ERROR("Ewald_long_range_forces_cu  execution failed");
    
    
    compute_short_range_forces_cu<<<NBLOCK(nAtoms), PERBLOCK>>>( nAtoms,
                                              gpd.xs(activeIdx),                                                      
                                              gpd.fs(activeIdx),
                                              neighborCounts,
                                              grid.neighborlist.ptr,
                                              grid.perBlockArray.d_data.ptr,
                                              gpd.qs(activeIdx),
                                              alpha,
                                              r_cut,
                                              state->boundsGPU,
                                              state->devManager.prop.warpSize, 0.5);
    CUT_CHECK_ERROR("Ewald_short_range_forces_cu  execution failed");
    
}


void export_FixChargeEwald() {
//     class_<FixChargeEwald, SHARED(FixChargeEwald), bases<FixCharge> > ("FixChargeEwald", init<SHARED(State), string, string> (args("state", "handle", "groupHandle")))
//         .def("setParameters", &FixChargeEwald::setParameters, (python::arg("alpha"), python::arg("r_cut")))
//         ;
}
