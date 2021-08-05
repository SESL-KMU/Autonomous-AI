/*
 * sensor.c
 * 
 * Real-Time Workshop code generation for Simulink model "sensor.mdl".
 *
 * Model Version              : 1.4
 * Real-Time Workshop version : 6.4  (R2006a)  03-Feb-2006
 * C source code generated on : Fri Apr 20 13:54:22 2007
 */

#include "sensor.h"
#include "sensor_private.h"

/* Block signals (auto storage) */
BlockIO_sensor sensor_B;

/* Block states (auto storage) */
D_Work_sensor sensor_DWork;

/* Real-time model */
RT_MODEL_sensor sensor_M_;
RT_MODEL_sensor *sensor_M = &sensor_M_;

/* Model output function */

static void sensor_output(int_T tid)
{

  /* Level2 S-Function Block: '<Root>/PCI-6052E ' (adnipcie) */
  {
    SimStruct *rts = sensor_M->childSfunctions[0];
    sfcnOutputs(rts, 0);
  }
}

/* Model update function */

static void sensor_update(int_T tid)
{

  /* Update absolute time for base rate */
  if(!(++sensor_M->Timing.clockTick0)) ++sensor_M->Timing.clockTickH0;
  sensor_M->Timing.t[0] = sensor_M->Timing.clockTick0 *
    sensor_M->Timing.stepSize0 + sensor_M->Timing.clockTickH0 *
    sensor_M->Timing.stepSize0 * 4294967296.0;
}

/* Model initialize function */

void sensor_initialize(boolean_T firstTime)
{
  if (firstTime) {

    /* Registration code */
    /* initialize real-time model */
    (void) memset((char_T *)sensor_M,0,
     sizeof(RT_MODEL_sensor));

    /* Initialize timing info */
    {
      int_T *mdlTsMap = sensor_M->Timing.sampleTimeTaskIDArray;
      mdlTsMap[0] = 0;
      sensor_M->Timing.sampleTimeTaskIDPtr = (&mdlTsMap[0]);
      sensor_M->Timing.sampleTimes = (&sensor_M->Timing.sampleTimesArray[0]);
      sensor_M->Timing.offsetTimes = (&sensor_M->Timing.offsetTimesArray[0]);
      /* task periods */
      sensor_M->Timing.sampleTimes[0] = (0.001);

      /* task offsets */
      sensor_M->Timing.offsetTimes[0] = (0.0);
    }

    rtmSetTPtr(sensor_M, &sensor_M->Timing.tArray[0]);

    {
      int_T *mdlSampleHits = sensor_M->Timing.sampleHitArray;
      mdlSampleHits[0] = 1;
      sensor_M->Timing.sampleHits = (&mdlSampleHits[0]);
    }

    rtmSetTFinal(sensor_M, -1);
    sensor_M->Timing.stepSize0 = 0.001;

    /* Setup for data logging */
    {
      static RTWLogInfo rt_DataLoggingInfo;

      sensor_M->rtwLogInfo = &rt_DataLoggingInfo;

      rtliSetLogFormat(sensor_M->rtwLogInfo, 0);

      rtliSetLogMaxRows(sensor_M->rtwLogInfo, 1000);

      rtliSetLogDecimation(sensor_M->rtwLogInfo, 1);

      rtliSetLogVarNameModifier(sensor_M->rtwLogInfo, "rt_");

      rtliSetLogT(sensor_M->rtwLogInfo, "tout");

      rtliSetLogX(sensor_M->rtwLogInfo, "");

      rtliSetLogXFinal(sensor_M->rtwLogInfo, "");

      rtliSetSigLog(sensor_M->rtwLogInfo, "");

      rtliSetLogXSignalInfo(sensor_M->rtwLogInfo, NULL);

      rtliSetLogXSignalPtrs(sensor_M->rtwLogInfo, NULL);

      rtliSetLogY(sensor_M->rtwLogInfo, "");

      rtliSetLogYSignalInfo(sensor_M->rtwLogInfo, NULL);

      rtliSetLogYSignalPtrs(sensor_M->rtwLogInfo, NULL);
    }

    sensor_M->solverInfoPtr = (&sensor_M->solverInfo);
    sensor_M->Timing.stepSize = (0.001);
    rtsiSetFixedStepSize(&sensor_M->solverInfo, 0.001);
    rtsiSetSolverMode(&sensor_M->solverInfo, SOLVER_MODE_SINGLETASKING);

    /* block I/O */
    sensor_M->ModelData.blockIO = ((void *) &sensor_B);

    {

      ((real_T*)&sensor_B.PCI6052E)[0] = 0.0;
    }

    /* parameters */
    sensor_M->ModelData.defaultParam = ((real_T *) &sensor_P);

    /* states (dwork) */

    sensor_M->Work.dwork = ((void *) &sensor_DWork);

    (void) memset((char_T *) &sensor_DWork,0,
     sizeof(D_Work_sensor));
    {
      int_T i;
      real_T *dwork_ptr = (real_T *) &sensor_DWork.PCI6052E_RWORK[0];

      for (i = 0; i < 64; i++) {
        dwork_ptr[i] = 0.0;
      }
    }

    /* initialize non-finites */
    rt_InitInfAndNaN(sizeof(real_T));

    /* child S-Function registration */
    {
      RTWSfcnInfo *sfcnInfo = &sensor_M->NonInlinedSFcns.sfcnInfo;

      sensor_M->sfcnInfo = (sfcnInfo);

      rtssSetErrorStatusPtr(sfcnInfo, &rtmGetErrorStatus(sensor_M));
      rtssSetNumRootSampTimesPtr(sfcnInfo, &sensor_M->Sizes.numSampTimes);
      rtssSetTPtrPtr(sfcnInfo, &rtmGetTPtr(sensor_M));
      rtssSetTStartPtr(sfcnInfo, &rtmGetTStart(sensor_M));
      rtssSetTimeOfLastOutputPtr(sfcnInfo, &rtmGetTimeOfLastOutput(sensor_M));
      rtssSetStepSizePtr(sfcnInfo, &sensor_M->Timing.stepSize);
      rtssSetStopRequestedPtr(sfcnInfo, &rtmGetStopRequested(sensor_M));
      rtssSetDerivCacheNeedsResetPtr(sfcnInfo,
       &sensor_M->ModelData.derivCacheNeedsReset);
      rtssSetZCCacheNeedsResetPtr(sfcnInfo,
       &sensor_M->ModelData.zCCacheNeedsReset);
      rtssSetBlkStateChangePtr(sfcnInfo, &sensor_M->ModelData.blkStateChange);
      rtssSetSampleHitsPtr(sfcnInfo, &sensor_M->Timing.sampleHits);
      rtssSetPerTaskSampleHitsPtr(sfcnInfo, &sensor_M->Timing.perTaskSampleHits);
      rtssSetSimModePtr(sfcnInfo, &sensor_M->simMode);
      rtssSetSolverInfoPtr(sfcnInfo, &sensor_M->solverInfoPtr);
    }

    sensor_M->Sizes.numSFcns = (1);

    /* register each child */
    {
      (void) memset((void *)&sensor_M->NonInlinedSFcns.childSFunctions[0],0,
       1*sizeof(SimStruct));
      sensor_M->childSfunctions =
        (&sensor_M->NonInlinedSFcns.childSFunctionPtrs[0]);
      sensor_M->childSfunctions[0] =
        (&sensor_M->NonInlinedSFcns.childSFunctions[0]);

      /* Level2 S-Function Block: sensor/<Root>/PCI-6052E  (adnipcie) */
      {
        SimStruct *rts = sensor_M->childSfunctions[0];
        /* timing info */
        time_T *sfcnPeriod = sensor_M->NonInlinedSFcns.Sfcn0.sfcnPeriod;
        time_T *sfcnOffset = sensor_M->NonInlinedSFcns.Sfcn0.sfcnOffset;
        int_T *sfcnTsMap = sensor_M->NonInlinedSFcns.Sfcn0.sfcnTsMap;

        (void) memset((void*)sfcnPeriod,0,
         sizeof(time_T)*1);
        (void) memset((void*)sfcnOffset,0,
         sizeof(time_T)*1);
        ssSetSampleTimePtr(rts, &sfcnPeriod[0]);
        ssSetOffsetTimePtr(rts, &sfcnOffset[0]);
        ssSetSampleTimeTaskIDPtr(rts, sfcnTsMap);

        /* Set up the mdlInfo pointer */
        {
          ssSetBlkInfo2Ptr(rts, &sensor_M->NonInlinedSFcns.blkInfo2[0]);
          ssSetRTWSfcnInfo(rts, sensor_M->sfcnInfo);
        }

        /* Allocate memory of model methods 2 */
        {
          ssSetModelMethods2(rts, &sensor_M->NonInlinedSFcns.methods2[0]);
        }

        /* outputs */
        {
          ssSetPortInfoForOutputs(rts,
           &sensor_M->NonInlinedSFcns.Sfcn0.outputPortInfo[0]);
          _ssSetNumOutputPorts(rts, 1);
          /* port 0 */
          {
            _ssSetOutputPortNumDimensions(rts, 0, 1);
            ssSetOutputPortWidth(rts, 0, 1);
            ssSetOutputPortSignal(rts, 0, ((real_T *) &sensor_B.PCI6052E));
          }
        }

        /* path info */
        ssSetModelName(rts, "PCI-6052E ");
        ssSetPath(rts, "sensor/PCI-6052E ");
        ssSetRTModel(rts,sensor_M);
        ssSetParentSS(rts, NULL);
        ssSetRootSS(rts, rts);
        ssSetVersion(rts, SIMSTRUCT_VERSION_LEVEL2);

        /* parameters */
        {
          mxArray **sfcnParams = (mxArray **)
            &sensor_M->NonInlinedSFcns.Sfcn0.params;

          ssSetSFcnParamsCount(rts, 6);
          ssSetSFcnParamsPtr(rts, &sfcnParams[0]);

          ssSetSFcnParam(rts, 0, (mxArray*)&sensor_P.PCI6052E_P1_Size[0]);
          ssSetSFcnParam(rts, 1, (mxArray*)&sensor_P.PCI6052E_P2_Size[0]);
          ssSetSFcnParam(rts, 2, (mxArray*)&sensor_P.PCI6052E_P3_Size[0]);
          ssSetSFcnParam(rts, 3, (mxArray*)&sensor_P.PCI6052E_P4_Size[0]);
          ssSetSFcnParam(rts, 4, (mxArray*)&sensor_P.PCI6052E_P5_Size[0]);
          ssSetSFcnParam(rts, 5, (mxArray*)&sensor_P.PCI6052E_P6_Size[0]);
        }

        /* work vectors */
        ssSetRWork(rts, (real_T *) &sensor_DWork.PCI6052E_RWORK[0]);
        ssSetIWork(rts, (int_T *) &sensor_DWork.PCI6052E_IWORK[0]);
        {

          struct _ssDWorkRecord *dWorkRecord = (struct _ssDWorkRecord *)
            &sensor_M->NonInlinedSFcns.Sfcn0.dWork;

          ssSetSFcnDWork(rts, dWorkRecord);
          _ssSetNumDWork(rts, 2);

          /* RWORK */
          ssSetDWorkWidth(rts, 0, 64);
          ssSetDWorkDataType(rts, 0,SS_DOUBLE);
          ssSetDWorkComplexSignal(rts, 0, 0);
          ssSetDWork(rts, 0, &sensor_DWork.PCI6052E_RWORK[0]);

          /* IWORK */
          ssSetDWorkWidth(rts, 1, 66);
          ssSetDWorkDataType(rts, 1,SS_INTEGER);
          ssSetDWorkComplexSignal(rts, 1, 0);
          ssSetDWork(rts, 1, &sensor_DWork.PCI6052E_IWORK[0]);
        }

        /* registration */
        adnipcie(rts);

        sfcnInitializeSizes(rts);
        sfcnInitializeSampleTimes(rts);

        /* adjust sample time */
        ssSetSampleTime(rts, 0, 0.001);
        ssSetOffsetTime(rts, 0, 0.0);
        sfcnTsMap[0] = 0;

        /* set compiled values of dynamic vector attributes */

        ssSetNumNonsampledZCs(rts, 0);

        /* Update connectivity flags for each port */
        _ssSetOutputPortConnected(rts, 0, 1);
        _ssSetOutputPortBeingMerged(rts, 0, 0);
        /* Update the BufferDstPort flags for each input port */
      }
    }
  }
}

/* Model terminate function */

void sensor_terminate(void)
{

  /* Level2 S-Function Block: '<Root>/PCI-6052E ' (adnipcie) */
  {
    SimStruct *rts = sensor_M->childSfunctions[0];
    sfcnTerminate(rts);
  }
}

/*========================================================================*
 * Start of GRT compatible call interface                                 *
 *========================================================================*/

void MdlOutputs(int_T tid) {

  sensor_output(tid);
}

void MdlUpdate(int_T tid) {

  sensor_update(tid);
}

void MdlInitializeSizes(void) {
  sensor_M->Sizes.numContStates = (0); /* Number of continuous states */
  sensor_M->Sizes.numY = (0);           /* Number of model outputs */
  sensor_M->Sizes.numU = (0);           /* Number of model inputs */
  sensor_M->Sizes.sysDirFeedThru = (0); /* The model is not direct feedthrough */
  sensor_M->Sizes.numSampTimes = (1);   /* Number of sample times */
  sensor_M->Sizes.numBlocks = (2);      /* Number of blocks */
  sensor_M->Sizes.numBlockIO = (1);     /* Number of block outputs */
  sensor_M->Sizes.numBlockPrms = (18); /* Sum of parameter "widths" */
}

void MdlInitializeSampleTimes(void) {
}

void MdlInitialize(void) {
}

void MdlStart(void) {

  /* Level2 S-Function Block: '<Root>/PCI-6052E ' (adnipcie) */
  {
    SimStruct *rts = sensor_M->childSfunctions[0];
    sfcnStart(rts);
    if(ssGetErrorStatus(rts) != NULL) return;
  }

  MdlInitialize();
}

RT_MODEL_sensor *sensor(void) {
  sensor_initialize(1);
  return sensor_M;
}

void MdlTerminate(void) {
  sensor_terminate();
}

/*========================================================================*
 * End of GRT compatible call interface                                   *
 *========================================================================*/

