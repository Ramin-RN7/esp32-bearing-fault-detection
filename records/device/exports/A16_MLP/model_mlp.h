/* Auto-generated MLP header (A16_MLP) */
#pragma once
#include <math.h>
#include <stdint.h>

static const int L0_IN=4;
static const int L0_OUT=10;
static const float L0_W[40]={9.67910811e-02,-1.11756384e+00,-1.21677510e-01,-1.46144733e-01,1.45555362e-01,9.03550625e-01,-5.06851494e-01,-9.33031440e-01,-1.83119923e-01,7.75308073e-01,-4.83560026e-01,-9.73466098e-01,-9.67305675e-02,-1.08527832e-01,-3.03366423e-01,2.68012732e-01,-6.22149825e-01,-2.45589554e-01,-1.11016244e-01,5.35438418e-01,-9.05775905e-01,-3.70807528e-01,6.40958697e-02,-3.35348509e-02,-2.13749543e-01,-8.81252810e-03,-4.63623047e-01,1.32454529e-01,7.01886058e-01,-2.93055832e-01,4.46062952e-01,-5.85764766e-01,1.12800635e-01,-2.87751794e-01,4.86170828e-01,4.60217595e-01,7.01672435e-01,-4.43954974e-01,-1.04995084e+00,2.46357068e-01};
static const float L0_B[10]={-1.10395133e+00,4.94672120e-01,-1.50613533e-02,8.49267185e-01,-3.04974794e-01,5.56785941e-01,-5.50719798e-01,6.24080002e-01,4.15792555e-01,-5.41505694e-01};
static const int L1_IN=10;
static const int L1_OUT=1;
static const float L1_W[10]={1.62180877e+00,6.44854426e-01,1.02375793e+00,4.45844620e-01,-1.43794045e-01,7.37061679e-01,-4.97073442e-01,-7.95493722e-01,-1.02837276e+00,-7.93570638e-01};
static const float L1_B[1]={1.52270555e-01};

static inline float act_relu(float x){return x>0?x:0;}
static inline float act_tanhf(float x){return tanhf(x);} 
static inline float act_logistic(float x){return 1.0f/(1.0f+expf(-x));}
static void dense_forward(const float *W,const float *B,int IN,int OUT,const float *x,float *y){
  for(int o=0;o<OUT;o++){float s=B[o];const float *col=&W[o*IN];for(int i=0;i<IN;i++){s+=col[i]*x[i];}y[o]=s;}}

// returns argmax class index
static int A16_MLP_predict(const float *x_in,float *tmp0,float *tmp1){
  const float *x=x_in; float *a=tmp0; float *b=tmp1; int IN,OUT;
  IN=L0_IN; OUT=L0_OUT; dense_forward(L0_W,L0_B,IN,OUT,x,a);
  for(int j=0;j<OUT;j++){a[j]=act_relu(a[j]);}
  const float *sw=x; x=a; a=(float*)sw;
  IN=L1_IN; OUT=L1_OUT; dense_forward(L1_W,L1_B,IN,OUT,x,a);
  float p=act_logistic(a[0]); return p>=0.5f?1:0;
}
