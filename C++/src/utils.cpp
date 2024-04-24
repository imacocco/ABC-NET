#ifndef __utils__
#define __utils__

#include "func.hpp"

using namespace std;

// load info
void load_info(const string& path, int& seed, double& beta, vector<double>& ps, vector<double>& ID_ref){

	double appo{0};
	string id_path;
	std::ifstream input(path+"input.dat", ios::in);
	if (input.is_open()) {
		input >> id_path;
		input >> seed;
		input >> beta;
		while(input>>appo)
			ps.push_back(appo);
		input.close();	
	}
	else {
		std::cerr << "Unable to open input file. Aborting\n";
		abort();
	}

	// load ref id -------------------------------------------------------------------------------------------------------------------
	std::ifstream input1(id_path, ios::in);
	if (input1.is_open()) {
		while(input1>>appo)
			ID_ref.push_back(appo);
		input1.close();
	}
	else{
		std::cerr << "Unable to open id_ref file.Aborting\n";
		abort();
	}
	return;
}

// compute distance between two ids, according to L1 or L_inf metrics
double compute_id_dist(const vector<double>& id_ref, const vector<double>& id_temp, const size_t& upper_dist, const bool& L_inf){

	double dist{0};

	if (L_inf){
		double dist_i{0};
		for (size_t i = 1; i < upper_dist; i++){ // possibly excluding ID at 0, as it is related to the degree
			dist_i = abs(id_ref[i] - id_temp[i]);
			if ( dist_i > dist )
				dist = dist_i;
		}
	}
	else{
		for (size_t i = 0; i < upper_dist; i++)
			dist += abs(id_ref[i] - id_temp[i]);
	}
	return dist;
}

//------------------------------------------------------------------------------
// load ratio of volumes
void load_vol_ratios(const string& filename, vector<vector<double>>& v){
	double appo{0};
	ifstream input(filename, ios::in);
	if (input.is_open()) {
		for (size_t i = 0; i < v.size(); i++) {
			for (size_t j = 0; j < v[0].size(); j++) {
				input >> appo;
				v[i][j] = appo;
			}
		}
	}
	else {
		std::cerr << "Unable to open volumes ratio file. Aborting\n";
		abort();
	}
	// std::cout << "Loaded " << v.size() << " lines and " << v[0].size() << " columns\n";
	return;
}

//------------------------------------------------------------------------------
// compute the ids at different radii by comparting the theoretical vol_ratio and the empirical one
void compute_id(vector<double>& id_new, const vector<size_t>& n_tot,
	vector<size_t>& cumulative, const vector<vector<double>>& vol_ratios){

	size_t idx{0};
	double nk_ratio{0};
	double m{0};
	partial_sum(n_tot.begin(),n_tot.end(),cumulative.begin());

	for (size_t i = 0; i < vol_ratios.size(); i++) {
		nk_ratio = static_cast<double>( cumulative[int(0.5*(i+1))] )/static_cast<double>( cumulative[i+1] );
		// cout << i+1 << ' ' << int(0.5*(i+1)) << ' ' << cumulative[int(0.5*(i+1))] << ' ' << cumulative[i+1] << ' ' << nk_ratio << endl;
		for (size_t j = 0; j < vol_ratios[0].size(); j++) {
			if (nk_ratio > vol_ratios[i][j]){
				idx = j;
				// linear interpolation, could even take 0.01*j, but fairly precise in this way
				m = (vol_ratios[i][idx]-vol_ratios[i][idx-1])/0.01;
				id_new[i] = idx*0.01 + (nk_ratio-vol_ratios[i][idx])/m; //(idx-1)*0.01 + (nk_ratio-vol_ratios[i][idx-1])/m;
				break;
			}
		}
		//cout << i << ' ' << id_new[i] << endl;
	}
}

//------------------------------------------------------------------------------
//find all k and n given l2 and l1
void update_dist(const vector<vector<size_t>>& dist_base, vector<vector<size_t>>& dist_upd,
	vector<size_t>& n_upd, const size_t& i, const size_t& j, const edge_type& et) {

	if (et==EXTERNAL){
		dist_upd[i][j] = 1;
		dist_upd[j][i] = 1;

		n_upd[0] += 2;
		n_upd[1] += 2;

		return;
	}
	
	size_t N{dist_base.size()};
	size_t dist_temp{0};

	if (et==PROTUSION){	// assuming i is source (degree!=0) and j is target (degree==0)
		for (size_t l = 0; l < N; l++){
			if (j==l || dist_base[i][l] == 10000)
				continue;
			dist_temp = dist_base[i][l]+1;
			dist_upd[j][l] = dist_temp;
			dist_upd[l][j] = dist_temp;
			n_upd[dist_temp] += 2;
			//cout << l<< ' '<< dist_base[i][l]<<' '<< dist_upd[j][l]<<endl;
		}
		n_upd[0] += 1;
	}
	else{	//(et==INNER)				// adding edge between vertices already connected to the giant component
	//std::fill(n_upd.begin()+1,n_upd.end(),0);
    	for (size_t l = 0; l < N; l++){
			if ( dist_base[i][l] == 10000 || dist_base[j][l] == 10000 )
				continue;
      		for (size_t m = 0; m < l; m++){
				if ( dist_base[l][m] == 10000 || dist_base[j][m] == 10000 || dist_base[i][m] == 10000 )
					continue;
				dist_temp = min( dist_base[l][m], dist_base[l][i]+1+dist_base[j][m] );
				dist_temp = min( dist_temp, dist_base[l][j]+1+dist_base[i][m] );
				//cout << l << ' ' << m<< ' ' << dist_base[l][m] << ' ' << dist_temp << endl;
				if (dist_temp != dist_base[l][m]){
					dist_upd[l][m] = dist_temp;
					dist_upd[m][l] = dist_temp;
					n_upd[dist_temp] += 2;
					n_upd[dist_base[l][m]] -= 2;
				}
			}
		}
	}
	return;
}

void save_IDs(const string& fileout, const vector<size_t>& step, const vector<double>& dist, const vector<vector<double>>& IDs){

	ofstream ofst(fileout, std::ios::out);
	if (ofst.is_open()) {
		ofst << setprecision(3) << fixed;
		ofst << "#step of update\tdistance\tID\n";
		for (size_t i = 0; i < IDs.size(); i++) {
			ofst << step[i] << ' ' << dist[i] << ' ';
			for (size_t j = 0; j < IDs[0].size(); j++)
				ofst << IDs[i][j] << ' ';
			ofst << endl;
		}
		ofst.close();
	}
}


void save_output(const string& fileout, const vector<int>& step, const vector<int>& actions,
				 const vector<size_t>& num_non_zero_deg, const size_t& starting_edges,
				 const vector<double>& dist, const vector<vector<double>>& IDs, const size_t& inserted_edges){

	const char* cstr = fileout.c_str();
	FILE *ofst;
	ofst = fopen(cstr, "w");
	if (ofst == nullptr){
		perror("Error opening file");
		return;
	}
	// header
	std::fprintf(ofst, "#step of update, action, num conected vertex, num edges, distance, ID\n");

	for (size_t i = 0; i < inserted_edges+1; i++) {
		std::fprintf(ofst, "%d %d %ld %ld %.3f ",step[i], actions[i], num_non_zero_deg[i], starting_edges+i, dist[i]);
		for (size_t j = 0; j < IDs[0].size(); j++)
			std::fprintf(ofst,"%.3f ",IDs[i][j]);
		std::fprintf(ofst,"\n");
	}
	fclose(ofst);
}

void save_edges(const string& fileout, const vector<int>& sources, const vector<int>& targets, const size_t& tot_edges){

	ofstream ofst(fileout, std::ios::out);
	if (ofst.is_open()) {
		for (size_t i = 0; i < tot_edges; i++)
			ofst << sources[i] << ' ' << targets[i] << endl;
		ofst.close();
	}
}

void print_mat(const vector<vector<size_t>>& m, size_t maxx){
	maxx = min(maxx,m.size());
	for (size_t i = 0; i < maxx; i++) {
		for (size_t j = 0; j < maxx; j++) {
			cout << m[i][j] << ' ';
		}
		cout<< endl;
	}
	cout << endl;
}
#endif
