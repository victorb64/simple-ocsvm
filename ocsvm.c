#include "ocsvm.h"
#include "math.h"
void OCSVM_Scale(OCSVM_Ctx* ctx, float* inputs, float* scaledOutput)
{
	for(int i=0;i<ctx->params.numDim;i++)
	{
	    // s0+(s1-s0*((di-li) / (hi-li)))
		scaledOutput[i] = ctx->scale_params.scaleRange[0] +
				(ctx->scale_params.scaleRange[1] -
						ctx->scale_params.scaleRange[0]*
								((inputs[i] - ctx->scale_params.lows[i]) / (ctx->scale_params.highs[i] - ctx->scale_params.lows[i])));
	}
}

float OCSVM_Predict(OCSVM_Ctx* ctx, float* inputs)
{
	float scaledInputs[OCSVM_MAX_DIMENSION];
	OCSVM_Scale(ctx, inputs, scaledInputs);
	float sum = 0;
	for(int i = 0;i<ctx->params.numSVs;i++)
	{
		sum += ctx->kernel_function((struct OCSVM_Ctx*)ctx,scaledInputs,i);
	}

	// bias
	sum -= ctx->params.rho;
	return sum;
}

float OCSVM_RBF(OCSVM_Ctx* ctx, float* inputs, uint32_t row)
{
	// exp(-gamma*(Xi-Xj)^2)
	float res = 0;
	float Xj = 0;
	for(int j=0;j<ctx->params.numDim;j++)
	{
		Xj = OCSVM_GetModelDataAt(ctx, row, j);
		float temp = Xj - inputs[j];
		res += temp * temp;
	}
	return expf(-ctx->params.gamma * res);
}
