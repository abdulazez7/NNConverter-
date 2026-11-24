#include "model.h"

#ifdef RV_BARE_METAL
static void put(char c){volatile unsigned char *u=(volatile unsigned char*)0x10000000;while(!(u[5]&32)){}u[0]=(unsigned char)c;}
static void text(const char *s){while(*s)put(*s++);}
static void finish(int fail){*(volatile unsigned int*)0x100000=fail?0x00013333:0x5555;for(;;){}}
#else
#include <stdio.h>
static void text(const char *s){fputs(s,stdout);}
#endif
static int close_enough(float a,float b){float d=a-b;if(d<0)d=-d;float m=b<0?-b:b;return a==a && d<=0.0003f+0.0003f*m;}

static const int in_0_0[6]={6,8,9,0,0,2};
static const float expected_0_0[2]={-4.107896805e+00f,4.321291924e+00f};
static const int in_1_0[6]={7,11,2,11,5,6};
static const float expected_1_0[2]={-4.190371513e+00f,4.260120869e+00f};
static const int in_2_0[6]={5,4,8,3,9,10};
static const float expected_2_0[2]={4.081849098e+00f,-4.117966652e+00f};
static const int in_3_0[6]={8,1,11,11,10,8};
static const float expected_3_0[2]={4.122406006e+00f,-4.039398193e+00f};
static const int in_4_0[6]={8,8,0,5,5,11};
static const float expected_4_0[2]={-3.774802208e+00f,4.173687458e+00f};
static const int in_5_0[6]={4,11,11,2,0,9};
static const float expected_5_0[2]={-4.174999714e+00f,4.330949783e+00f};
static const int in_6_0[6]={8,6,5,5,3,6};
static const float expected_6_0[2]={-3.978082895e+00f,4.396122456e+00f};
static const int in_7_0[6]={5,0,0,4,8,0};
static const float expected_7_0[2]={4.108356953e+00f,-4.085513115e+00f};
static float out_0[2];
int main(void){int fail=0;
if(rv_model_run(in_0_0,out_0)!=0)fail++;
for(int i=0;i<2;i++)if(!close_enough(out_0[i],expected_0_0[i]))fail++;
if(rv_model_run(in_1_0,out_0)!=0)fail++;
for(int i=0;i<2;i++)if(!close_enough(out_0[i],expected_1_0[i]))fail++;
if(rv_model_run(in_2_0,out_0)!=0)fail++;
for(int i=0;i<2;i++)if(!close_enough(out_0[i],expected_2_0[i]))fail++;
if(rv_model_run(in_3_0,out_0)!=0)fail++;
for(int i=0;i<2;i++)if(!close_enough(out_0[i],expected_3_0[i]))fail++;
if(rv_model_run(in_4_0,out_0)!=0)fail++;
for(int i=0;i<2;i++)if(!close_enough(out_0[i],expected_4_0[i]))fail++;
if(rv_model_run(in_5_0,out_0)!=0)fail++;
for(int i=0;i<2;i++)if(!close_enough(out_0[i],expected_5_0[i]))fail++;
if(rv_model_run(in_6_0,out_0)!=0)fail++;
for(int i=0;i<2;i++)if(!close_enough(out_0[i],expected_6_0[i]))fail++;
if(rv_model_run(in_7_0,out_0)!=0)fail++;
for(int i=0;i<2;i++)if(!close_enough(out_0[i],expected_7_0[i]))fail++;
if(fail)text("FAIL numerical parity\n");else text("PASS 8 fixtures numerical parity\n");
#ifdef RV_BARE_METAL
finish(fail);
#endif
return fail?1:0;}