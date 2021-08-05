#include "math.h"

/* this model has 12 states */
#define NORDER  12 
#define NUMOFINPUT 5
#define NUMOFOUTPUT 32

/* parameter valus here */
#define PI        3.141592
#define KALPHA		36500.0	  // tire lateral stiffness
#define KLAMBDA		63000.0	  // tire longitudinal stiffness
#define KZ        247212.0  // tire normal stiffness
#define g		  9.81	  // the acceleration of gravity
#define lf		0.998   // front wheel base[m] 
#define lr		1.572   // rear wheel base[m]
#define lt    1.5280  // mean wheel tread [m] 
#define ltf   1.5400  // front wheel tread [m] 
#define ltr   1.5160  // rear wheel tread [m] 
#define h     0.53    // CG Height [m] 
#define M_chassis	   1395.   // body mass [kg]
#define M_tire		   60.		 // tire mass	[kg]
#define I_cz		     2600. 	 // rotational inertia of body about z axis [kg*m^2]
#define R_tire		   0.3135  // radius of tires with normal status [m]
#define I_tire		   0.95    // tire inertia [kg*m^2]
#define vehicle_vel	22.2  // initial vehicle velocity : 80 km/hour
#define min_vehicle_vel	0.5556   // minimum vehicle velocity : 2 km/hour
#define M_total		   1635.   // M_chassis + 4*M_tire 

// double P_brake[5] = {0., 0., 0., 0., 0.};
double T_brake[4] = {0., 0., 0., 0.};

// set the slip angle and slip ratio as a global variables 
double Sideslip_angle = 0;
double Slip_Angle[4] = {0., 0., 0., 0.};
double Slip_Ratio[4] = {0., 0., 0., 0.};
double RTIRE[4] = {0.2937, 0.2937, 0.3009, 0.3009};
double F_tire[4][2] = { {0., 0.}, {0., 0.}, {0., 0.}, {0., 0.} };
double Ax = 0.;				// Longitudinal acceleration [m/s^2]
double Ay = 0.;				// Laterall accleration [m/s^2]

///////////////////////////////////////////////////////////////////////////////
// FL, FR, RL, RR Road frictional coefficient
//double MU[4] = {0.8, 0.8, 0.8, 0.8};	
double MU[4] = {0.3, 0.3, 0.3, 0.3};	
///////////////////////////////////////////////////////////////////////////////

// user function prototypes 
void InitialDataSet();
void fn_slipangle(real_T *x, double delta);
void fn_slipratio(real_T *x);
void fn_tireforce2();
void fn_dugoff_tireforce2(double vx, double ax, double ay);

