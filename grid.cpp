#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

// Variables

int iterations = 1000;
int side = 100;
double lenght = 0.6;
double mu = -0.2;
double ni_x = 0.25;
double ni_y = 0.61803;
double boundary = 10.0;

double henon_map(double x, double px, double y, double py)
{
	double omega_x = ni_x * 2 * M_PI;
	double omega_y = ni_y * 2 * M_PI;

	double cosx = cos(omega_x);
	double sinx = sin(omega_x);
	double cosy = cos(omega_y);
	double siny = sin(omega_y);

	double L[4][4] = 
	{
		cosx, sinx, 0, 0,
		-sinx, cosx,0, 0,
		0, 0, cosy, siny,
		0, 0, -siny, cosy
	};
	double v[4] = 
	{
		x,
		px + x*x - y*y + mu*()
	};
}