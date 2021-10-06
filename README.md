# simple-ocsvm
a simple C OCSVM predictor mainly for micros...

Only implements RBF kernel, but you can add your own...

Retrieval of model data is up to the user. I am using this in an embedded application using SPI Flash for model storage but you can do whatever.

Just define the **OCSVM_GetModelDataAt** function
```C
    float OCSVM_GetModelDataAt(OCSVM_Ctx* ctx, uint32_t row, uint32_t col)
    {
	    uint32_t rowsize = ctx->params.numDim * sizeof(float) + sizeof(float);
	    return Storage_Read4BF(row*rowsize + sizeof(float) + col*sizeof(float));
    }
```
## Example Usage

```C
    	OCSVM_Ctx ocsvm1;
	ocsvm1.ID = 0;
	// set kernel function - RBF is already implemented
	ocsvm1.kernel_function = (OCSVMKernel)&OCSVM_RBF;

	// basic OCSVM params
	ocsvm1.params.numDim = Storage_Read4B(SVM_PARAMS_START);
	ocsvm1.params.numSVs = Storage_Read4B(SVM_PARAMS_START+4);
	ocsvm1.params.gamma = Storage_Read4BF(SVM_PARAMS_START+8);
	ocsvm1.params.rho = Storage_Read4BF(SVM_PARAMS_START+12);

	// scaling params
	ocsvm1.scale_params.scaleRange[0] = 0;
	ocsvm1.scale_params.scaleRange[1] = 1;
	for(int i=0;i<ocsvm1.params.numDim;i++)
	{
		uint32_t baseaddr = SVM_SCALE_PARAMS_START + 2*sizeof(float) + i*(2*sizeof(float));
		ocsvm1.scale_params.lows[i] = Storage_Read4BF(baseaddr);
		ocsvm1.scale_params.highs[i] = Storage_Read4BF(baseaddr+sizeof(float));
	}

	// feed in some new data and check if it matches the model or is a novelty
	float somedata[6] = {0.1,0.2,1,2,3,0.5};
	float result = OCSVM_Predict(&ocsvm1, somedata);
	if(result <= 0)
	{
		// novelty! yay.
	}

```
