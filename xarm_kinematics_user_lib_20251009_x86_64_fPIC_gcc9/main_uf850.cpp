#include "xarm_kinematics_interface.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Test UF850 FK and IK at the same time, IK can return all solutions */
/* input: joint angles (dimension: DOF) in degree */

/* *************************************************/
/* execution example: ./main 10 0 0 -20 6 -52  */
/* *************************************************/

int main(int argc, char **argv)
{
	double deg2rad = M_PI/180.0; // convert from degree to radian

	if(argc!=7)
		exit(-1);

	double theta[6]={0}, theta_sol[6]={0};

	for(int j=0;j<6;j++)
		theta[j] = std::atof(argv[j+1])*deg2rad;

	/* Joint limits have to be passed, just use the proper Macros defined in 'xarm_kinematics_interface.h' */
	double q_min[6] = UF850_ANGLE_MIN; 
	double q_max[6] = UF850_ANGLE_MAX;
	
	/* q_pre: starting joint angle, reference joint angle */
	double q_pre[6] = {0, 0, 0, 0, 0, 0};

	double T[4][4] = {0}, pose_rpy[6] = {0};
	double tcp_offset[6] = {0,0,0, 0,0,0}; // change if there is any TCP offset, unit: mm, rad
	
	// * Attention! MUST BE Configured before calling FK or IK !
	uf_850_config(q_max, q_min, tcp_offset); 
	

	/*** Method 1: Use Transformation Matrix as TCP expression */
	int ret = uf_850_forward_kinematics(theta, T); // result is in transformation matrix

	fprintf(stderr, "UF850 FK (Matrix) result:\n" );
	for(int i=0; i<4; i++)
	{
		for(int j=0; j<4; j++)
			fprintf(stderr, "%lf,\t", T[i][j]);
		fprintf(stderr, "\n");
	}
	
	/* T: target Transformation matrix, theta_sol: solution angles */
	/* theta_sol: return the solution closest to q_pre. */
	/* all_sol: all possible joint angle solutions. [optional, only available for Lite6 and 850 models]*/
	std::vector< std::vector<double> > all_sol;
	fprintf(stderr, "q_max: %lf, %lf, %lf, %lf, %lf, %lf\n", q_max[0]/deg2rad, q_max[1]/deg2rad, q_max[2]/deg2rad, q_max[3]/deg2rad, q_max[4]/deg2rad, q_max[5]/deg2rad);
	fprintf(stderr, "q_min: %lf, %lf, %lf, %lf, %lf, %lf\n", q_min[0]/deg2rad, q_min[1]/deg2rad, q_min[2]/deg2rad, q_min[3]/deg2rad, q_min[4]/deg2rad, q_min[5]/deg2rad);
	ret = uf_850_inverse_kinematics(T, q_pre, theta_sol, q_max, q_min, &all_sol);

	if(ret)
	{	
		fprintf(stderr, "\nIK (Matrix) returns: %d, Solution Fail!\n", ret);
	}
	else
	{
		fprintf(stderr, "\nIK (Matrix) returns: %d, closest solution:\t", ret);
		for(int i=0; i<6; i++)
			fprintf(stderr, "%lf,\t", theta_sol[i]/deg2rad);
		fprintf(stderr, "\n" );

		for(int i=0; i<all_sol.size(); i++)
		{
			fprintf(stderr, "solution %d: \n", i+1);
			for(int j=0; j<6; j++)
				fprintf(stderr, "%lf,\t", all_sol[i][j]/deg2rad);
			fprintf(stderr, "\n" );
		}
	}


	/*** Method 2: Use [x,y,z,roll,pitch,yaw] as TCP expression */
	ret = uf_850_forward_kinematics(theta, pose_rpy);	
	fprintf(stderr, "\nUF850 FK (Pose) result:\n" );
	for(int i=0; i<6; i++)
	{	
		fprintf(stderr, "%lf,\t", pose_rpy[i]);
	}
	fprintf(stderr, "\n");

	ret = uf_850_inverse_kinematics(pose_rpy, q_pre, theta_sol);

	if(ret)
	{	
		fprintf(stderr, "\nIK (Pose) returns: %d, Solution Fail!\n", ret);
		exit(-1);
	}
	else
	{
		fprintf(stderr, "\nIK (Pose) returns: %d, closest solution:\t", ret);
		for(int i=0; i<6; i++)
			fprintf(stderr, "%lf,\t", theta_sol[i]/deg2rad);
		fprintf(stderr, "\n" );
	}

	return 0;
}
