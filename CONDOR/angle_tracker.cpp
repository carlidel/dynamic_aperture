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
	// orario o antiorario?
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

int main(int argc, const char * argv[])
{
	double dx = atof(argv[1]);
	double n_theta = atoi(argv[2]);
	double epsilon = atof(argv[3]);
	
	double d_theta = M_PI / (4 * n_theta);

	//std::string filename = "radscan_dx" + std::to_string(dx) + "_nthet" + std::to_string(n_theta) + "_epsilon" + std::to_string(epsilon) + ".txt";

	//std::ofstream out (filename, std::ofstream::out);

	std::cout << "dx " << dx << std::endl;
	std::cout << "n_theta " << n_theta << std::endl;
	std::cout << "epsilon " << epsilon << std::endl; 

	for (double angle = 0; angle <= M_PI / 4; angle += d_theta)
	{
		std::cout << "Scanning angle: " << angle << "/" << M_PI / 4 << std::endl;
		out << angle << " ";
		std::vector<double> * v = modulated_radius_scan(angle, dx, epsilon);
		for (unsigned int i = 0; i < v->size(); ++i)
		{
			std::cout << v->at(i) << " ";
			//out << v->at(i) << " ";
		}
		std::cout << std::endl;
		//out << std::endl;
		delete v;
	}
	return 0;
}