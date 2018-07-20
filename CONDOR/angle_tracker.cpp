#include <iostream>
#include <fstream>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <string>

// Variables

double epsilon_k[7] =
{
	1.000e-4,
	0.218e-4, 
	0.708e-4, 
	0.254e-4, 
	0.100e-4, 
	0.078e-4, 
	0.218e-4
};

double Omega_k[7] =
{
	1  *(2 * M_PI / 868.12),
	2  *(2 * M_PI / 868.12),
	3  *(2 * M_PI / 868.12),
	6  *(2 * M_PI / 868.12),
	7  *(2 * M_PI / 868.12),
	10 *(2 * M_PI / 868.12),
	12 *(2 * M_PI / 868.12),
};

double omega_x0 = 0.168 * 2 * M_PI;
double omega_y0 = 0.201 * 2 * M_PI;

// 2 test
// 0.28
// 0.31

// 3 test
// 0.31
// 0.32

double * modulated_hennon_map(double * v0, double epsilon, unsigned int n)
{
	double sum = 0;
	for (int i = 0; i < 7; ++i)
		sum += epsilon_k[i] * cos(Omega_k[i] * n);
	double omega_x = omega_x0 * (1 + epsilon * sum);
	double omega_y = omega_y0 * (1 + epsilon * sum);

	double cosx = cos(omega_x);
	double sinx = sin(omega_x);
	double cosy = cos(omega_y);
	double siny = sin(omega_y);

	double v[4];
	// clockwise!
	v[0] = cosx * v0[0] + sinx * (v0[1] + v0[0] * v0[0] - v0[2] * v0[2]);
	v[1] = -sinx * v0[0] + cosx * (v0[1] + v0[0] * v0[0] - v0[2] * v0[2]);
	v[2] = cosy * v0[2] + siny * (v0[3] - 2 * v0[0] * v0[2]);
	v[3] = -siny * v0[2] + cosy * (v0[3] - 2 * v0[0] * v0[2]);

	v0[0] = v[0];
	v0[1] = v[1];
	v0[2] = v[2];
	v0[3] = v[3];

	return v0;
}

int modulated_particle(double x0, double y0, unsigned int T, double epsilon)
{
	double * v = new double [4];
	v[0] = x0;
	v[1] = 0;
	v[2] = y0;
	v[3] = 0;

	for (unsigned int i = 0; i < T; ++i)
	{
		v = modulated_hennon_map(v, epsilon, i);
		if (v[0]*v[0] + v[2]*v[2] > 1000000)
		{
			// Particle lost!
			delete v;
			return i;
		}
	}
	// Particle not lost!
	delete v;
	return -1;
}

std::vector<double> * modulated_radius_scan(double theta, double dx, double epsilon, unsigned int max_turns = 10000000, unsigned int min_turns = 1000)
{
	std::vector<double> * v = new std::vector<double>();
	
	unsigned int actual_turns = max_turns;
	int temp;
	int i = -1;
	
	while(actual_turns >= min_turns)
	{
		i++;
		temp = modulated_particle(i * dx * cos(theta), i * dx * sin(theta), actual_turns, epsilon);
		if (temp != -1)
		{
			actual_turns = temp;
			v->push_back(actual_turns);
		}
		else
		{
			v->push_back(actual_turns);
		}
	}
	return v;
}

std::vector<double> * modulated_linear_scan(double y, double dx, double epsilon, unsigned int max_turns = 10000000, unsigned int n_scans = 80)
{
	std::vector<double> * v = new std::vector<double>();

	for(unsigned i = 0; i < n_scans; i++)
	{
		double temp = modulated_particle(i * dx, y, max_turns, epsilon);
		if (temp != -1)
		{
			v->push_back(temp);
		}
		else
		{
			v->push_back(max_turns);
		}
	}
	return v;
}

int main(int argc, const char * argv[])
{

	epsilon_k[0]			= atof(argv[1]);
	epsilon_k[1]			= atof(argv[2]);
	epsilon_k[2]			= atof(argv[3]);
	epsilon_k[3]			= atof(argv[4]);
	epsilon_k[4]			= atof(argv[5]);
	epsilon_k[5]			= atof(argv[6]);
	epsilon_k[6]			= atof(argv[7]);
	omega_x0				= atof(argv[8]) * 2 * M_PI;
	omega_y0				= atof(argv[9]) * 2 * M_PI;
	double dx 				= atof(argv[10]);
	double n_theta 			= atoi(argv[11]);
	double epsilon 			= atof(argv[12]);
	unsigned int max_turns 	= atoi(argv[13]);
	unsigned int from 		= atoi(argv[14]);
	unsigned int to 		= atoi(argv[15]);

	double d_theta = M_PI / (2 * n_theta);

	// Variables for angular scan

	std::cout << "epsilon_k[0] "<< epsilon_k[0]	<< std::endl;
	std::cout << "epsilon_k[1] "<< epsilon_k[1]	<< std::endl;
	std::cout << "epsilon_k[2] "<< epsilon_k[2]	<< std::endl;
	std::cout << "epsilon_k[3] "<< epsilon_k[3]	<< std::endl;
	std::cout << "epsilon_k[4] "<< epsilon_k[4]	<< std::endl;
	std::cout << "epsilon_k[5] "<< epsilon_k[5]	<< std::endl;
	std::cout << "epsilon_k[6] "<< epsilon_k[6]	<< std::endl;
	std::cout << "omega_x0 "	<< atof(argv[8])<< std::endl;
	std::cout << "omega_y0 "	<< atof(argv[9])<< std::endl;
	std::cout << "dx " 			<< dx 			<< std::endl;
	std::cout << "n_theta " 	<< n_theta 		<< std::endl;
	std::cout << "dtheta " 		<< d_theta 		<< std::endl;
	std::cout << "epsilon " 	<< epsilon 		<< std::endl;
	std::cout << "max_turns " 	<< max_turns	<< std::endl;
	std::cout << "from_angle " 	<< from 		<< std::endl;
	std::cout << "to_angle " 	<< to 			<< std::endl;

	for (double angle = from * d_theta; angle < to * d_theta; angle += d_theta)
	{
		//std::cout << "Scanning angle: " << angle << "/" << M_PI / 4 << std::endl;
		std::cout << angle << " ";
		std::vector<double> * v = modulated_radius_scan(angle, dx, epsilon, max_turns);
		for (unsigned int i = 0; i < v->size(); ++i)
		{
			std::cout << v->at(i) << " ";
		}
		std::cout << std::endl;
		delete v;
	}
/**/
/*
	// Variables for linear scan
	epsilon_k[0]			= atof(argv[1]);
	epsilon_k[1]			= atof(argv[2]);
	epsilon_k[2]			= atof(argv[3]);
	epsilon_k[3]			= atof(argv[4]);
	epsilon_k[4]			= atof(argv[5]);
	epsilon_k[5]			= atof(argv[6]);
	epsilon_k[6]			= atof(argv[7]);
	omega_x0				= atof(argv[8]) * 2 * M_PI;
	omega_y0				= atof(argv[9]) * 2 * M_PI;
	double dx 				= atof(argv[10]);
	double epsilon 			= atof(argv[11]);
	unsigned int max_turns 	= atoi(argv[12]);
	unsigned int from 		= atoi(argv[13]);
	unsigned int to 		= atoi(argv[14]);

	// Variables for angular scan

	std::cout << "epsilon_k[0] "<< epsilon_k[0]	<< std::endl;
	std::cout << "epsilon_k[1] "<< epsilon_k[1]	<< std::endl;
	std::cout << "epsilon_k[2] "<< epsilon_k[2]	<< std::endl;
	std::cout << "epsilon_k[3] "<< epsilon_k[3]	<< std::endl;
	std::cout << "epsilon_k[4] "<< epsilon_k[4]	<< std::endl;
	std::cout << "epsilon_k[5] "<< epsilon_k[5]	<< std::endl;
	std::cout << "epsilon_k[6] "<< epsilon_k[6]	<< std::endl;
	std::cout << "omega_x0 "	<< atof(argv[8])<< std::endl;
	std::cout << "omega_y0 "	<< atof(argv[9])<< std::endl;
	std::cout << "dx " 			<< dx 			<< std::endl;
	std::cout << "epsilon " 	<< epsilon 		<< std::endl;
	std::cout << "max_turns " 	<< max_turns	<< std::endl;
	std::cout << "from_y " 	<< from 		<< std::endl;
	std::cout << "to_y " 	<< to 			<< std::endl;

	for (double y = from * dx; y < to * dx; y += dx)
	{
		std::cout << y << " ";
		std::vector<double> * v = modulated_linear_scan(y, dx, epsilon, max_turns);
		for (unsigned int i = 0; i < v->size(); ++i)
		{
			std::cout << v->at(i) << " ";
		}
		std::cout << std::endl;
		delete v;
	}
	/**/
	return 0;
}