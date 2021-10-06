#ifndef OCSVM_H_
#define OCSVM_H_

#include <stdint.h>
#define OCSVM_MAX_DIMENSION 10

typedef struct
{
	uint32_t numDim;
	uint32_t numSVs;
	float gamma;
	float rho;
}OCSVM_Params;
typedef struct
{
	float scaleRange[2];
	float lows[OCSVM_MAX_DIMENSION];
	float highs[OCSVM_MAX_DIMENSION];
}OCSVM_Scale_Params;

struct OCSVM_Ctx;
typedef float (*OCSVMKernel)(struct OCSVM_Ctx*,float*, uint32_t);
typedef struct
{
	uint8_t ID;
	OCSVM_Params params;
	OCSVM_Scale_Params scale_params;
	OCSVMKernel kernel_function;
}OCSVM_Ctx;


void OCSVM_Scale(OCSVM_Ctx* ctx, float* inputs, float* scaledOutput);
float OCSVM_Predict(OCSVM_Ctx* ctx, float* inputs);

// rbf kernel implementation
float OCSVM_RBF(OCSVM_Ctx* ctx, float* inputs, uint32_t row);

// implement your own storage retrieval
extern float OCSVM_GetModelDataAt(OCSVM_Ctx* ctx, uint32_t row, uint32_t col);

#endif
