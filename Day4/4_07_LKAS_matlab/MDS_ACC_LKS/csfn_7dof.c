/* 7 degree of freedom vehicle dynamics model
   modeled by sh Lee
   modified by yj Cho
   modified by th Hwang (version 20040826)
   
   Note : The original file modified by yj Cho is used for TCS HILS.
   
**************************************************************************
   inputs =>
   u(0): steering input [rad]
**************************************************************************
   outputs =>
   y(0) : vehicle longitudinal velocity [m/s]
   y(1) : vehicle lateral velocity [m/s]
   y(2) : x position (global coordinate) [m]
   y(3) : y position (global coordinate) [m]
   y(4) : x position (local coordinate) [m]
   y(5) : y position (local coordinate) [m]
   y(6) : front wheel steer angle [rad]
   y(7) : yaw velocity [rad/sec]
   y(8) : yaw angle [rad]
   y(9) : FL slip angle [rad]
   y(10) : FR slip angle [rad]
   y(11) : RL slip angle [rad]
   y(12) : RR slip angle [rad]
   y(13) : FL wheel speed [m/s]
   y(14) : FR wheel speed [m/s]
   y(15) : RL wheel speed [m/s]
   y(16) : RR wheel speed [m/s]                
   y(17) : FL longitudinal tire force [N]
   y(18) : FR longitudinal tire force [N]
   y(19) : RL longitudinal tire force [N]
   y(20) : RR longitudinal tire force [N]
   y(21) : FL lateral tire force [N]
   y(22) : FR lateral tire force [N]
   y(23) : RL lateral tire force [N]
   y(24) : RR lateral tire force [N]
   y(25) : sideslip angle of CG [rad]
   y(26) : Longitudinal acceleration [m/s^2]
   y(27) : Laterall accleration [m/s^2]
   y(28) : FL brake torque
   y(29) : FR brake torque
   y(30) : RL brake torque
   y(31) : RR brake torque
**************************************************************************
   states = >
   x(0) : x position (local) [m]
   x(1) : vehicle longitudinal speed [m/sec]
   x(2) : y position () [m]
   x(3) : vehicle lateral speed [m/sec]
   x(4) : yaw rate [rad/sec]
   x(5) : FL wheel speed [rad/sec]
   x(6) : FR wheel speed [rad/sec]
   x(7) : RL wheel speed [rad/sec]
   x(8) : RR wheel speed [rad/sec]
   x(9): yaw angle [rad]
   x(10): x position (global coordinate) [m]
   x(11): y position (global coordinate) [m]
**************************************************************************
   parameters =>                  

**************************************************************************
   Global Variables =>

**************************************************************************
*/

#define S_FUNCTION_NAME  csfn_7dof
#define S_FUNCTION_LEVEL 2

#include "simstruc.h"
#include "vehicle_param.h"

#define U(element) (*uPtrs[element])  /* Pointer to Input Port0 */


void InitialDataSet()
{

}

/*====================*
 * S-function methods *
 *====================*/

/* Function: mdlInitializeSizes ===============================================
 * Abstract:
 *    The sizes information is used by Simulink to determine the S-function
 *    block's characteristics (number of inputs, outputs, states, etc.).
 */
static void mdlInitializeSizes(SimStruct *S)
{
    ssSetNumSFcnParams(S, 0);  /* Number of expected parameters */
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
        return; /* Parameter mismatch will be reported by Simulink */
    }

    ssSetNumContStates(S, NORDER);
    ssSetNumDiscStates(S, 0);

    if (!ssSetNumInputPorts(S, 1)) return;
    ssSetInputPortWidth(S, 0, NUMOFINPUT);
    ssSetInputPortDirectFeedThrough(S, 0, 1);

    if (!ssSetNumOutputPorts(S, 1)) return;
    ssSetOutputPortWidth(S, 0, NUMOFOUTPUT);

    ssSetNumSampleTimes(S, 1);
    ssSetNumRWork(S, 0);
    ssSetNumIWork(S, 0);
    ssSetNumPWork(S, 0);
    ssSetNumModes(S, 0);
    ssSetNumNonsampledZCs(S, 0);

    /* Take care when specifying exception free code - see sfuntmpl_doc.c */
    ssSetOptions(S, SS_OPTION_EXCEPTION_FREE_CODE);
}


/* Function: mdlInitializeSampleTimes =========================================
 * Abstract:
 *    Specifiy that we have a continuous sample time.
 */
static void mdlInitializeSampleTimes(SimStruct *S)
{
    ssSetSampleTime(S, 0, CONTINUOUS_SAMPLE_TIME);
    ssSetOffsetTime(S, 0, 0.0);
}


#define MDL_INITIALIZE_CONDITIONS
/* Function: mdlInitializeConditions ==========================================
 * Abstract:
 *    Set the initial condition
 */
static void mdlInitializeConditions(SimStruct *S)
{
    real_T *x0 = ssGetContStates(S);
    
    InitialDataSet();

    x0[0] = 0.;
    x0[1] = vehicle_vel; /* initial speed begins at vehicle_vel km/hour */
    x0[2] = 0.;
    x0[3] = 0.;
    x0[4] = 0.;
    x0[5] = vehicle_vel/RTIRE[0]; /* initial wheel velocity */
    x0[6] = vehicle_vel/RTIRE[1];
    x0[7] = vehicle_vel/RTIRE[2];
    x0[8] = vehicle_vel/RTIRE[3];
    x0[9] = 0.;
    x0[10] = 0.;
    x0[11] = 0.;
}


/* Function: mdlOutputs =======================================================
 * Abstract:
 *    In this function, you compute the outputs of your S-function
 *    block. Generally outputs are placed in the output vector, ssGetY(S).
 */
static void mdlOutputs(SimStruct *S, int_T tid)
{
    real_T            *y    = ssGetOutputPortRealSignal(S,0);
    real_T            *x    = ssGetContStates(S);
    
    InputRealPtrsType uPtrs = ssGetInputPortRealSignalPtrs(S,0);    
 
    UNUSED_ARG(tid); 	/* not used in single tasking mode */   

    y[0] = x[1];  		// vehicle longitudinal velocity [m/s]
    y[1] = x[3];  		// vehicle lateral velocity [m/s]
    y[2] = x[10]; 		// x position (global coordinate) [m]
    y[3] = x[11]; 		// y position (global coordinate) [m]
    y[4] = x[0]; 			// x position (local coordinate) [m]
    y[5] = x[2]; 			// y position (local coordinate) [m]
    y[6] = *uPtrs[0];     // front wheel steer angle [rad]
    y[7] = x[4];  		// yaw velocity [rad/sec] 
    y[8] = x[9];  		// yaw angle [rad]
    y[9] = Slip_Angle[0];  
    y[10] = Slip_Angle[1];
    y[11] = Slip_Angle[2];
    y[12] = Slip_Angle[3];
    y[13] = x[5]*RTIRE[0]; 		// FL wheel speed [m/s]
    y[14] = x[6]*RTIRE[1]; 		// FR wheel speed [m/s]
    y[15] = x[7]*RTIRE[2]; 		// RL wheel speed [m/s]
    y[16] = x[8]*RTIRE[3]; 		// RR wheel speed [m/s]
    y[17] = F_tire[0][0]; 		// FL longitudinal tire force
    y[18] = F_tire[1][0]; 		// FR longitudinal tire force
    y[19] = F_tire[2][0]; 		// RL longitudinal tire force
    y[20] = F_tire[3][0]; 		// RR longitudinal tire force
    y[21] = F_tire[0][1]; 		// FL lateral tire force
    y[22] = F_tire[1][1]; 		// FR lateral tire force
    y[23] = F_tire[2][1]; 		// RL lateral tire force
    y[24] = F_tire[3][1]; 		// RR lateral tire force
    y[25] = Sideslip_angle;			// sideslip angle of CG [rad]
    y[26] = Ax;				// Longitudinal acceleration [m/s^2]
    y[27] = Ay;				// Laterall accleration [m/s^2]
    y[28] = T_brake[0];  
    y[29] = T_brake[1];   
    y[30] = T_brake[2];   
    y[31] = T_brake[3];   
}


#define MDL_UPDATE  /* Change to #undef to remove function */
#if defined(MDL_UPDATE)
/* Function: mdlUpdate ========================================================
 * Abstract:
 *    This function is called once for every major integration time step.
 *    Discrete states are typically updated here, but this function is useful
 *    for performing any tasks that should only take place once per
 *    integration step.
 */
static void mdlUpdate(SimStruct *S, int_T tid)
{
	real_T            *x    = ssGetContStates(S);
  InputRealPtrsType uPtrs = ssGetInputPortRealSignalPtrs(S,0);
  	
}
#endif /* MDL_UPDATE */


#define MDL_DERIVATIVES
/* Function: mdlDerivatives ===================================================
 * Abstract:
 *      dx = function of(x, U)
 */
static void mdlDerivatives(SimStruct *S)
{
    // U(0) : steering angle 
    
    real_T            *dx   = ssGetdX(S);
    real_T            *x    = ssGetContStates(S);
    InputRealPtrsType uPtrs = ssGetInputPortRealSignalPtrs(S,0);
    
    Sideslip_angle = atan2(x[3], x[1]); // sideslip angle of CG [rad]
    
    // Braking Torque
    
    // get each wheel's slip angle Slip_Angle[] 
    fn_slipangle(x,U(0));
    // get each wheel's slip ratio Slip_Ratio[] 
    fn_slipratio(x);
    // get each wheel's tire force (Fx and Fy) 
    fn_dugoff_tireforce2(x[1], Ax, Ay);
    // fn_tireforce2();       
    
    
    T_brake[0] =  U(1);
    T_brake[1] =  U(2);
    T_brake[2] =  U(3);
    T_brake[3] =  U(4);    
    
                  
    
//   x(0) : x position (local) [m]
//   x(1) : vehicle longitudinal speed [m/sec]
//   x(2) : y position (local) [m]
//   x(3) : vehicle lateral speed [m/sec]
//   x(4) : yaw rate [rad/sec]
//   x(5) : FL wheel speed [rad/sec]
//   x(6) : FR wheel speed [rad/sec]
//   x(7) : RL wheel speed [rad/sec]
//   x(8) : RR wheel speed [rad/sec]
//   x(9) : yaw angle [rad]
//   x(10): x position (global coordinate) [m]
//   x(11): y position (global coordinate) [m]
   
   
    
    
    dx[0] = x[1];
    dx[1] = ( 
              cos(U(0)) * ( F_tire[0][0]+F_tire[1][0] ) 
              - sin(U(0)) * ( F_tire[0][1]+F_tire[1][1] ) 
              + F_tire[2][0] + F_tire[3][0] 
             ) / M_total + x[3]*x[4];
//		if(x[1]<min_vehicle_vel)
//    {
//    	if(dx[1]<0.) dx[1] = 0.;
//   }
    Ax = dx[1] - x[3]*x[4];
    dx[2] = x[3];
    dx[3] = ( 
             ( F_tire[0][1]+F_tire[1][1] ) * cos(U(0)) 
             + ( F_tire[0][0]+F_tire[1][0] ) * sin(U(0)) 
             + F_tire[2][1] + F_tire[3][1] 
             ) / M_total - x[1]*x[4];
    Ay = dx[3] + x[1]*x[4];
    dx[4] = ( 
              lf*( cos(U(0))*F_tire[0][1] + sin(U(0))*F_tire[0][0] 
                   + cos(U(0))*F_tire[1][1] + sin(U(0))*F_tire[1][0] )
              + 0.5*ltf*( -F_tire[0][0]*cos(U(0)) + F_tire[1][0]*cos(U(0))
                          + F_tire[0][1]*sin(U(0)) - F_tire[1][1]*sin(U(0)) )
              - 0.5*ltr*( F_tire[2][0] - F_tire[3][0] )
              - lr*( F_tire[2][1] + F_tire[3][1] )
             ) / I_cz;

    dx[5] = (-RTIRE[0]*F_tire[0][0] - T_brake[0]) / I_tire;
  //  if(x[5]<=(min_vehicle_vel/RTIRE[0]))
  //  {
  //  	if(dx[5]<=0.) dx[5] = 0.;
  //  }

    dx[6] = (-RTIRE[1]*F_tire[1][0] - T_brake[1]) / I_tire;  
   // if(x[6]<=(min_vehicle_vel/RTIRE[1]))
   // {
   // 	if(dx[6]<=0.) dx[6] = 0.;
   // } 
       
    dx[7] = (-RTIRE[2]*F_tire[2][0] - T_brake[2]) / I_tire;
    //if(x[7]<=(min_vehicle_vel/RTIRE[2]))
   // {
    //	if(dx[7]<=0.) dx[7] = 0.;
   // }
    
    dx[8] = (-RTIRE[3]*F_tire[3][0] - T_brake[3]) / I_tire;
   // if(x[8]<=(min_vehicle_vel/RTIRE[3]))
   // {
   // 	if(dx[8]<=0.) dx[8] = 0.;
   // }
    dx[9] = x[4];
    dx[10] = x[1] * cos(x[9]) - x[3]*sin(x[9]);
    dx[11] = x[1] * sin(x[9]) + x[3]*cos(x[9]);
    
}

/* function fn_slipangle()
 beta <= atan2(x[3], x[1]) : atan2(vy / vx)
 lf/x[1]*x[4] <= effect of yaw rate, x[1] : Vx[m/sec], x[4] : yaw rate [rad/sec]
 delta <== effect of steering angle [rad]
 slip_angle = beta + lf*yaw rate/Vx - steering angle
*/
/*---------------------------------------------------------------------------*/
void fn_slipangle(real_T *x, double delta)
{
    Slip_Angle[0] = atan2( (x[3]+x[4]*lf), (x[1]-0.5*x[4]*ltf) ) - delta;
    Slip_Angle[1] = atan2( (x[3]+x[4]*lf), (x[1]+0.5*x[4]*ltf) ) - delta;
    Slip_Angle[2] = atan2( (x[3]-x[4]*lr), (x[1]-0.5*x[4]*ltr) );
    Slip_Angle[3] = atan2( (x[3]-x[4]*lr), (x[1]+0.5*x[4]*ltr) ); 
}
/*---------------------------------------------------------------------------*/


/* function fn_slipratio()
  if(wheel velocity < vehicle velocity)
      braking situation
      slip ratio = ( vehicle velocity - wheel velocity ) / vehicle velocity
  else
      driving situaltion 
      slip ratio = ( vehicle velocity - wheel velocity ) / wheel velocity

  x[1] : vehicle velocity
  x[5] : FL wheel velocity
  x[6] : FR wheel velocity
  x[7] : RL wheel velocity
  x[8] : RR wheel velocity
*/
/*---------------------------------------------------------------------------*/
void fn_slipratio(real_T *x)
{
    int i = 0;
    double vw[4];
    double uw[4];
    double ws[4];
    for(i = 0; i<4; i++)
    {
    	ws[i] = x[5+i];
    //	if(ws[i]<0.) ws[i] = 0.;
    }
    vw[0] = sqrt( (x[1]-x[4]*ltf*0.5)*(x[1]-x[4]*ltf*0.5) + 
                  (x[3]+x[4]*lf)*(x[3]+x[4]*lf) );
    vw[1] = sqrt( (x[1]+x[4]*ltf*0.5)*(x[1]+x[4]*ltf*0.5) + 
                  (x[3]+x[4]*lf)*(x[3]+x[4]*lf) );
    vw[2] = sqrt( (x[1]-x[4]*ltr*0.5)*(x[1]-x[4]*ltr*0.5) + 
                  (x[3]-x[4]*lr)*(x[3]-x[4]*lr) );
    vw[3] = sqrt( (x[1]+x[4]*ltr*0.5)*(x[1]+x[4]*ltr*0.5) + 
                  (x[3]-x[4]*lr)*(x[3]-x[4]*lr) );

    for(i = 0; i<4; i++)
    {
      uw[i] = vw[i]*cos(Slip_Angle[i]);
    	/* if vehicle velocity is faster than the wheel speed. 
    	that is braking situation. */
    	if((ws[i]*RTIRE[i])<uw[i])
    		Slip_Ratio[i] = (uw[i]-ws[i]*RTIRE[i]) / uw[i];
      else                     
        Slip_Ratio[i] = (uw[i]-ws[i]*RTIRE[i]) / (ws[i]*RTIRE[i]);
    }
}
/*---------------------------------------------------------------------------*/

/* function fn_tireforce2()
    this function get 4 wheel's tire force from each wheel's slip angle,
                                                             slip ratio,
                                                             normal force
    by using brush tire model.
    
    global variables 
    Ftire[0][0]:FL longitudinal tire force, Ftire[0][1]:FL lateral tire force
    Ftire[1][0]:FR longitudinal tire force, Ftire[1][1]:FR lateral tire force
    Ftire[2][0]:RL longitudinal tire force, Ftire[2][1]:RL lateral tire force
    Ftire[3][0]:RR longitudinal tire force, Ftire[3][1]:RR lateral tire force
*/
/*---------------------------------------------------------------------------*/
void fn_tireforce2()
{
    double ftire_z[4] = {0., 0., 0., 0.};
    double lambda[4] = {0., 0., 0., 0.};
    double alpha[4] = {0., 0., 0., 0.};
    double power[4] = {0., 0., 0., 0.};
    double gamma[4] = {0., 0., 0., 0.};
    double epsilon[4] = {0., 0., 0., 0.};
    double sintheta[4] = {0., 0., 0., 0.};
    double costheta[4] = {0., 0., 0., 0.};
    double fepsilon[4] = {0., 0., 0., 0.};
    int k;
    
    
    // 1. get the each wheel's normal force
    ftire_z[0] =0.5*M_total*g*lr/(lf+lr);  /* front wheel */
    ftire_z[1] =0.5*M_total*g*lr/(lf+lr); 
    ftire_z[2] =0.5*M_total*g*lf/(lf+lr);  /* rear wheel  */
    ftire_z[3] =0.5*M_total*g*lf/(lf+lr);
 
    // using term lambda instead of slip ratio.
    lambda[0] = Slip_Ratio[0];
    lambda[1] = Slip_Ratio[1];
    lambda[2] = Slip_Ratio[2];
    lambda[3] = Slip_Ratio[3];
 
    // using alpha lambda instead of slip angle.
    alpha[0] = Slip_Angle[0];
    alpha[1] = Slip_Angle[1];
    alpha[2] = Slip_Angle[2];
    alpha[3] = Slip_Angle[3];

    // brush tire model rotune
    for(k=0; k<4; k++)
    {
    	if(lambda[k]> 0.99) lambda[k] = 0.99;
    	/* braking condition */
    	if(lambda[k] >0.)
      {
      	alpha[k] = alpha[k] + 1e-10;
	   		lambda[k]= lambda[k] + 1e-10;
		 		power[k] = tan(alpha[k])*tan(alpha[k]);
	   		gamma[k] = sqrt( (lambda[k]*lambda[k]) + 
	   		           (KALPHA/KLAMBDA)*(KALPHA/KLAMBDA) * power[k] );
	   		epsilon[k]= 1. - (KLAMBDA/(3.*MU[k]*ftire_z[k])) * 
	   		           ( gamma[k]/(1.-lambda[k]) );
	   		sintheta[k]=KALPHA * tan(alpha[k])  / 
	   		           (KLAMBDA * gamma[k]);
        costheta[k]=lambda[k] / gamma[k];
        fepsilon[k] = (1./6.) - ((1./2.)*epsilon[k]*epsilon[k]) + 
                   ((1./3.)*epsilon[k]*epsilon[k]*epsilon[k]);
        	if( epsilon[k] >= 0 )
          {
          	F_tire[k][0] = -(KLAMBDA * lambda[k] * epsilon[k] * epsilon[k])/ 
          	(1.-lambda[k])-
          	(6. * MU[k] * ftire_z[k]* costheta[k] * fepsilon[k]);
            F_tire[k][1] = -(KALPHA * (epsilon[k]*epsilon[k]) * tan(alpha[k]) / 
                          ( 1. - lambda[k] ))-
                          (6. * MU[k] * ftire_z[k] * sintheta[k] * fepsilon[k]);
          }
          else
          { 
          	F_tire[k][0] = -MU[k] * ftire_z[k] * costheta[k];
            F_tire[k][1] = -MU[k] * ftire_z[k] * sintheta[k];
          }
        }
        /* driving condition  */
        else
        {
        	alpha[k] = alpha[k] + 1e-10;
	    		lambda[k] = lambda[k] + 1e-10;
          power[k] = (1. + lambda[k]) * tan(alpha[k]);
          gamma[k] = sqrt( (lambda[k]*lambda[k]) + 
          					 (KALPHA/KLAMBDA) * (power[k]*power[k]) );
          epsilon[k] = 1. - ( ((KLAMBDA) / (3.*MU[k]*ftire_z[k])) * gamma[k] );
          sintheta[k] = ( KALPHA * tan(alpha[k] ) * (1. + lambda[k]) ) / 
          				      (KLAMBDA * gamma[k]);
          fepsilon[k] = (1./6.) - ((1./2.)*epsilon[k]*epsilon[k]) + 
                        ((1./3.)*epsilon[k]*epsilon[k]*epsilon[k]);
          costheta[k] = lambda[k]/gamma[k];
          	if (epsilon[k] > 0.)
          	{
          		F_tire[k][0] = -( KLAMBDA * lambda[k] * epsilon[k]* epsilon[k] )
              	             -( (6. * MU[k] * ftire_z[k] ) * 
              	             costheta[k] * fepsilon[k] );
            	F_tire[k][1] = -( KALPHA * ( 1. + lambda[k] ) * epsilon[k] * 
              	             epsilon[k] * tan(alpha[k]) ) - ( (6. * MU[k] * 
                	           ftire_z[k] * sintheta[k] ) * fepsilon[k] );
          	}
          	else
          	{
          		F_tire[k][0] = -MU[k] * ftire_z[k] * costheta[k];
            	F_tire[k][1] = -MU[k] * ftire_z[k] * sintheta[k];
          	}
        	}
    }
}
/*---------------------------------------------------------------------------*/


/* function fn_dugoff_tireforce2
    this function get 4 wheel's tire force from each wheel's slip angle,
                                                             slip ratio,
                                                             normal force
    by using dugoff tire model.
    
    global variables 
    Ftire[0][0]:FL longitudinal tire force, Ftire[0][1]:FL lateral tire force
    Ftire[1][0]:FR longitudinal tire force, Ftire[1][1]:FR lateral tire force
    Ftire[2][0]:RL longitudinal tire force, Ftire[2][1]:RL lateral tire force
    Ftire[3][0]:RR longitudinal tire force, Ftire[3][1]:RR lateral tire force
*/
/*---------------------------------------------------------------------------*/
void fn_dugoff_tireforce2(double vx, double ax, double ay)
{
    double ftire_z[4] = {0., 0., 0., 0.};
    double lambda[4] = {0., 0., 0., 0.};
    double alpha[4] = {0., 0., 0., 0.};
    double numk[4] = {0., 0., 0., 0.};
    double denk[4] = {0., 0., 0., 0.};
    double k[4] = {0., 0., 0., 0.};
    double fk[4] = {0., 0., 0., 0.};
    double mf, mr;
    double l;
    
    double er = 0.020;

    int i;
    
/*    
    // 1. get the each wheel's normal force
    l = lf+lr;
    mf = (lr*g/l + h*ax/l)*M_total/g;
    mr = (lf*g/l - h*ax/l)*M_total/g;
    ftire_z[0] =(g+2.*h*ay/ltf)*mf/2.0;  // front wheel
    ftire_z[1] =(g-2.*h*ay/ltf)*mf/2.0;
    ftire_z[2] =(g+2.*h*ay/ltr)*mr/2.0;  // rear wheel
    ftire_z[3] =(g-2.*h*ay/ltr)*mr/2.0;
*/

    ftire_z[0] =0.5*M_total*g*lr/(lf+lr);  // front wheel 
    ftire_z[1] =0.5*M_total*g*lr/(lf+lr); 
    ftire_z[2] =0.5*M_total*g*lf/(lf+lr);  // rear wheel  
    ftire_z[3] =0.5*M_total*g*lf/(lf+lr);
    
    
    // rolling radius of tire according to normal force,stiffness
    RTIRE[0] = R_tire - ftire_z[0]/KZ;
    RTIRE[1] = R_tire - ftire_z[1]/KZ;
    RTIRE[2] = R_tire - ftire_z[2]/KZ;
    RTIRE[3] = R_tire - ftire_z[3]/KZ;

    // using term lambda instead of slip ratio.
    lambda[0] = Slip_Ratio[0];
    lambda[1] = Slip_Ratio[1];
    lambda[2] = Slip_Ratio[2];
    lambda[3] = Slip_Ratio[3];
 
    // using alpha lambda instead of slip angle.
    alpha[0] = Slip_Angle[0];
    alpha[1] = Slip_Angle[1];
    alpha[2] = Slip_Angle[2];
    alpha[3] = Slip_Angle[3];

    // dugoff tire model rotune
    for(i=0; i<4; i++)
    {
      if (lambda[i]> 0.99) lambda[i]=0.99;
      if (lambda[i] < -0.99) lambda[i] = -0.99;
      
      if ( (lambda[i]<1.0e-10) && (lambda[i]>0.0) ) lambda[i] = 1.0e-10;
      if ( (lambda[i]>-1.0e-10) && (lambda[i]<0.0) ) lambda[i] = -1.0e-10;
      
      numk[i] = MU[i]*ftire_z[i]*(1.0 - er*vx*sqrt(lambda[i]*lambda[i] + 
                tan(alpha[i])*tan(alpha[i])))*(1.0-lambda[i]);
      denk[i] = 2.0*sqrt( KLAMBDA*KLAMBDA*lambda[i]*lambda[i] + 
                KALPHA*KALPHA*tan(alpha[i])*tan(alpha[i]) );
      k[i] = numk[i]/denk[i];

     	if(k[i]<1.0)
     	{
     		fk[i] = k[i]*(2.0-k[i]);
     	}
     	else
     	{
     		fk[i] = 1.0;
     	}
     	// Fx = (Ks*slipratio / 1-slipratio) * F(k)
     	// Fy = (Kalpha*tan(alpha) / 1-slipratio) * F(k)   
     	F_tire[i][0] = -(KLAMBDA*lambda[i]/(1.0-lambda[i]))*fk[i];
     	F_tire[i][1] = -(KALPHA*tan(alpha[i])/(1.0-lambda[i]))*fk[i];
   }
}

/* Function: mdlTerminate =====================================================
 * Abstract:
 *    No termination needed, but we are required to have this routine.
 */
static void mdlTerminate(SimStruct *S)
{
    UNUSED_ARG(S); /* unused input argument */
}

#ifdef  MATLAB_MEX_FILE    /* Is this file being compiled as a MEX-file? */
#include "simulink.c"      /* MEX-file interface mechanism */
#else
#include "cg_sfun.h"       /* Code generation registration function */
#endif
