#ifndef __func__
#define __func__

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <ctime>
#include <numeric>
#include <cassert>
#include <iomanip>
#include <filesystem>
#include <math.h>
#include <cmath>
#include <cstdio>

#include <gsl/gsl_randist.h>


using namespace std;

enum edge_type{ EXTERNAL, PROTUSION, INNER };
enum actions{ RANDOM, DEGREE, DEGREE_NEIGH, TRIADIC_CLOSURE };

void load_info(const string&, int&, double&, vector<double>&, vector<double>&);

void load_vol_ratios(const string&, vector<vector<double>>&);

void compute_id(vector<double>&, const vector<size_t>&,
	vector<size_t>&, const vector<vector<double>>&);

void update_dist(const vector<vector<size_t>>&, vector<vector<size_t>>&,
		vector<size_t>&, const size_t&, const size_t&, const edge_type&);

double compute_id_dist(const vector<double>&, const vector<double>&, const size_t&, const bool&);

void save_IDs(const string&, const vector<size_t>&, const vector<double>&, const vector<vector<double>>&);

void save_output(const string&, const vector<int>&, const vector<int>&, const vector<size_t>&, 
				 const size_t&, const vector<double>&, const vector<vector<double>>&, const size_t&);

void save_edges(const string&, const vector<int>&, const vector<int>&, const size_t&);

void print_mat(const vector<vector<size_t>>&, size_t);

// actions

size_t extract_action(const vector<double>&);

tuple<int,int> degree_powerlaw(const vector<size_t>&, vector<size_t>&, vector<size_t>&, double);

tuple<int,int> degree_neighbour(const vector<size_t>&, vector<size_t>&,
    const vector<int>&, const vector<int>&, const size_t&);

tuple<int,int> triadic_closure(const vector<vector<size_t>>&, const size_t&);

#endif
