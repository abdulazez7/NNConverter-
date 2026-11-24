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
static const float expected_0_0[2]={-1.575385094e+01f,1.247894001e+01f};
static const int in_1_0[6]={7,11,2,11,5,6};
static const float expected_1_0[2]={-1.233751488e+01f,9.929297447e+00f};
static const int in_2_0[6]={5,4,8,3,9,10};
static const float expected_2_0[2]={7.736135006e+00f,-8.396194458e+00f};
static const int in_3_0[6]={8,1,11,11,10,8};
static const float expected_3_0[2]={1.290609837e+01f,-1.469277668e+01f};
static const int in_4_0[6]={8,8,0,5,5,11};
static const float expected_4_0[2]={-6.959822655e+00f,4.580469608e+00f};
static const int in_5_0[6]={4,11,11,2,0,9};
static const float expected_5_0[2]={-1.905206490e+01f,1.566279697e+01f};
static const int in_6_0[6]={8,6,5,5,3,6};
static const float expected_6_0[2]={-5.342301369e+00f,4.932683945e+00f};
static const int in_7_0[6]={5,0,0,4,8,0};
static const float expected_7_0[2]={1.232297421e+01f,-1.414985943e+01f};
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