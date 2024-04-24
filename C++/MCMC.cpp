#include "func.hpp"

#define MAX_DIST 30
#define UPPER_DIST 15
#define STARTING_EDGES 10
#define EDGE_TARGET 1036
#define NUM_VERTEX 500
#define MAX_ITER 150000

using namespace std;

// initialise static random generator
// in this way i don't have to pass it explicitly to every function
static gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);

int main(int argc, char const *argv[]) {

	// deal with input
	if (argc!=2) {
		std::cerr << "path to simulation directory needed" << '\n';
		return -1;
	}

	string dir{argv[1]};
	int seed{0};    		// set the random seed
	double beta{0};			// beta
	vector<double> ps;		// probabilities associated with each action
	vector<double> ID_ref;	// id ref

	load_info(argv[1], seed, beta, ps, ID_ref);
	gsl_rng_set(r, seed);

	assert(ID_ref.size()>UPPER_DIST);

	vector<double> ps_cum(ps.size(),0);	// cumulative of the probabilities	
	partial_sum(ps.begin(), ps.end(), ps_cum.begin());

	// initialize containers ---------------------------------------------------------------------------------------------------------
	vector<int> sources(EDGE_TARGET,-1);						// sources
	vector<int> targets(EDGE_TARGET,-1);						// and targets of edges, in order to eventually reconstruct the graph
	int source{0};
	int target{0};
	int appo{0};
	bool already_present{false};
	vector<size_t> degree(NUM_VERTEX,0);						// ...
	vector<size_t> cumulative(NUM_VERTEX,0);					// helper cumulative vector
	vector<size_t> cumulative1(NUM_VERTEX,0);					// helper cumulative vector
	vector<size_t> vertex_with_non_zero_degree(NUM_VERTEX,0);	// store which vertex have degree different from 0
	size_t num_non_zero_deg{0};									// number of vertex with degree different from 0
	vector<size_t> num_non_zero_deg_evo(EDGE_TARGET-STARTING_EDGES+1,0);
	size_t edge_index{0};										// index to update vectors
	size_t actual_edges{STARTING_EDGES};						// number of edges present in the network

	vector<vector<size_t>> distances_0(NUM_VERTEX, vector<size_t>(NUM_VERTEX,10000));
	vector<vector<size_t>> distances_1(NUM_VERTEX, vector<size_t>(NUM_VERTEX,10000));

	vector<size_t> n_tot_0(MAX_DIST+1,0);						// total of points AT distance given by the index of the array
	n_tot_0[0] = STARTING_EDGES+1;
	vector<size_t> n_tot_1(MAX_DIST+1,0);						// for temporary updates in MC steps
	n_tot_1[0] = STARTING_EDGES+1;
	vector<size_t> n_cum(MAX_DIST+1,0);							// vector to store the cumulatives of n_tot

	// intrinsic dimension
	// vector<double> ID_ref;									// target ID
	vector<double> ID(MAX_DIST,0.0);							// computed ID
	vector<vector<double>> ID_list(EDGE_TARGET-STARTING_EDGES+1, vector<double>(MAX_DIST,0.0)); // ID list
	// distances between IDs
	double id_dist{0};
	vector<double> id_dist_list(EDGE_TARGET-STARTING_EDGES+1,0.0);	// distances from ID_ref

	edge_type et{EXTERNAL};											// type of edge
	actions my_act{RANDOM};											// selected action
	vector<int> actions_list(EDGE_TARGET-STARTING_EDGES+1,0);		// action list
	actions_list[0] = -1;
	vector<int> step_of_update(EDGE_TARGET-STARTING_EDGES+1,0);		// list of MC-steps updates
	step_of_update[0] = -1;

	// load volumes ratios -----------------------------------------------------------------------------------------------------------
	vector<vector<double>> vol_ratios(MAX_DIST, vector<double>(1001,0.0));
	//load_vol_ratios("/home/iuri/Dropbox/Accademia/Progetti/NET/codes/C++/src/volume_ratios_r0.5_d0.01_R1_70.dat",vol_ratios);
	load_vol_ratios("src/volume_ratios_r0.5_d0.01_R1_70.dat",vol_ratios);

	// initialize the graph as a STARTING_EDGES-vertex chain -------------------------------------------------------------------------
	for (size_t i = 0; i < NUM_VERTEX; i++){
		if (i < STARTING_EDGES + 1){
			sources[i] = i;
			targets[i] = i+1;
			degree[i] = 2;
			vertex_with_non_zero_degree[i] = i;
			for (size_t j = 0; j < i; j++){
				distances_0[i][j] = i-j;
				distances_0[j][i] = i-j;
				distances_1[i][j] = i-j;
				distances_1[j][i] = i-j;
				n_tot_0[i-j] += 2;
				n_tot_1[i-j] += 2;
			}
		}
		distances_0[i][i] = 0;
		distances_1[i][i] = 0;
	}
	// correction for boundaries------------------------------------------------------------------------------------------------------
	degree[0] = 1; degree[STARTING_EDGES] = 1; sources[STARTING_EDGES] = -1; targets[STARTING_EDGES] = -1;
	num_non_zero_deg = STARTING_EDGES + 1;
	num_non_zero_deg_evo[0] = num_non_zero_deg;

	compute_id(ID, n_tot_0, n_cum, vol_ratios);
	ID_list[0] = ID;
	id_dist_list[0] = compute_id_dist(ID, ID_ref, UPPER_DIST, true);

	// ON THE FLY UPDATE, might be slower if more simulations are running in parallel
	// string fileout{dir+"output.dat"}; //ids_"+to_string(static_cast<int>(beta))+"
	// const char* cstr = fileout.c_str();
	// FILE *ofst;
	// ofst = fopen(cstr, "w");
	// if (ofst == nullptr){
	// 	perror("Error opening file");
	// 	return 1;
	// }
	// std::fprintf(ofst, "#step of update, action, num conected vertex, num edges, distance, ID\n");
	// std::fprintf(ofst, "-1 -1 %ld %ld %.3f ", num_non_zero_deg, actual_edges, id_dist_list[0] );
	// for (size_t j = 0; j < ID.size(); j++)
	// 	std::fprintf(ofst, "%.3f ",ID[j]);
	// std::fprintf(ofst,"\n");

	// update_dist(distances_0, distances_1, n_tot_1, 7, 8, EXTERNAL);
	// print_mat(distances_1, 10);
	// vector<size_t> counter(15,0);

	
	// for (size_t i = 0; i < 12; i++){
	// 	for (size_t j = 0; j < 12; j++)
	// 	{
	// 		counter[ distances_1[i][j] ]+=1;
	// 	}
	// }
	// for (size_t i = 0; i < 15; i++)
	// {
	// 	cout << n_tot_1[i] << ' ' <<counter[i]<<endl;
	// }
	

	// possible strategy: instead of copying the accepted move into che reference one, I use one object as a base and one as
	// update, changing their role when an MCMC move is accepted. This way no copying is ever made. THIS HAS CAVEATS!
	// in order to properly work you need to update the matrix nonetheless...
	// MCMC cycles -------------------------------------------------------------------------------------------------------------------
	for (size_t i = 0; i < MAX_ITER; i++){
		// if(i%1000 == 0)
		// 	printf("iter %ld\tedges %ld\tvertices %ld\t dist from trgt %.3f\n",i,actual_edges,num_non_zero_deg,id_dist_list[edge_index]);
		// extract 2 vertices, according to given moves and realtive probabilities: choose action
		do{
			already_present = false;
			// switch (gsl_rng_uniform_int(r,4)){
			switch (extract_action(ps_cum)){
			case 0:
				source = vertex_with_non_zero_degree[ gsl_rng_uniform_int(r, num_non_zero_deg) ];
				//source = gsl_rng_uniform_int(r, NUM_VERTEX);
				target = gsl_rng_uniform_int(r, NUM_VERTEX);
				my_act = RANDOM;					
				break;
			case 1:
				tie(source, target) = degree_powerlaw(degree, cumulative, cumulative1, 1.);
				my_act = DEGREE;
				break;
			case 2:
				tie(source, target) = degree_neighbour(degree, cumulative, sources, targets, actual_edges);
				my_act = DEGREE_NEIGH;
				break;
			case 3:
				tie(source, target) = triadic_closure(distances_0, n_tot_0[2]);
				my_act = TRIADIC_CLOSURE;
			default:
				break;
			}

			if (source == target){
				already_present = true;
				continue;
			}
			for (size_t j = 0; j < actual_edges; j++){
				if ( ( sources[j] == source && targets[j] == target) || ( sources[j] == target && targets[j] == source ) ){
					already_present = true;
					break;
				}
			}
		}
    	while(already_present);	// repeat till the selected edge is not present and source!=target
		// decide which kind of connection it is going to be
		if (degree[source]!=0){
			if (degree[target]==0)
				et = PROTUSION;
			else
				et = INNER;
		}
		else{
			if (degree[target]==0)
				et = EXTERNAL;
			else{
				appo = target;
				target = source;
				source = appo;
				et = PROTUSION;
			}
		}
		// printf("source %d\tdeg source %ld\ttarget %d\tdeg tar %ld\t edgtype %d\t action%d\n",source, degree[source], target, degree[target],et,my_act);
		
		update_dist(distances_0, distances_1, n_tot_1, source, target, et);
		compute_id(ID, n_tot_1, n_cum, vol_ratios);
		id_dist = compute_id_dist(ID, ID_ref, MAX_DIST, true);

		// accept/reject
    	if ( gsl_rng_uniform(r) < exp(beta*(-id_dist + id_dist_list[edge_index])) ){
			// UPDATE 
			distances_0 = distances_1;
			n_tot_0 = n_tot_1;
			
			degree[source] += 1;
			degree[target] += 1;

			sources[actual_edges] = source;
			targets[actual_edges] = target;

			if (et==PROTUSION){
				vertex_with_non_zero_degree[num_non_zero_deg] = target;
				num_non_zero_deg += 1;
			}
			if(et==EXTERNAL){
				vertex_with_non_zero_degree[num_non_zero_deg] = source;
				num_non_zero_deg += 1;
				vertex_with_non_zero_degree[num_non_zero_deg] = target;
				num_non_zero_deg += 1;
			}

			edge_index += 1;
			actual_edges += 1;

			step_of_update[edge_index] = static_cast<int>(i);
			id_dist_list[edge_index] = id_dist;
			ID_list[edge_index] = ID;
			actions_list[edge_index] = my_act;
			num_non_zero_deg_evo[edge_index] = num_non_zero_deg;	 

			// ON THE FLY save to output
			// std::fprintf(ofst, "%ld %d %ld %ld %.3f ",i, my_act, num_non_zero_deg, actual_edges, id_dist);
			// for (size_t j = 0; j < ID.size(); j++)
			// 	std::fprintf(ofst,"%.3f ",ID[j]);
			// std::fprintf(ofst,"\n");
		}
		else{
			// if (external){
			// 	for (size_t j = 0; j < NUM_VERTEX; j++){
			// 		distances_1[j][source] = distances_0[j][source];
			// 		distances_1[source][j] = distances_0[j][source];
			// 	}
			// }
			// else
			distances_1 = distances_0;
			n_tot_1 = n_tot_0;
		}
		// std::cout <<num_non_zero_deg << ' ' << n_tot_0[0]<< ' ' << n_tot_1[0] << ' ' << n_tot_0[1]<< ' ' << n_tot_1[1] << ' '
		// << actual_edges << ' '  << ID[0] << ' ' << 1.0*actual_edges/num_non_zero_deg << endl<<endl;

		
		if (actual_edges == EDGE_TARGET)
			break;
	}

	// fclose(ofst);
	// std::cout << "edges inserted " << actual_edges << endl;
	// std::cout << "vertices connected " << num_non_zero_deg << ' ' << n_tot_0[0]<< endl;
	// std::cout << "ID[0] " << ID[0] << " E/N " << 1.0*actual_edges/num_non_zero_deg << endl;

	// vector<size_t> counter(30,0);
	
	// for (size_t i = 0; i < NUM_VERTEX; i++){
	// 	for (size_t j = 0; j < NUM_VERTEX; j++)
	// 	{
	// 		counter[ distances_0[i][j] ]+=1;
	// 	}
	// }
	// for (size_t i = 0; i < 10; i++){
	// 	cout << n_tot_1[i] << ' ' <<counter[i]<<endl;
	// }
	save_edges(dir+"edges.dat",sources,targets,actual_edges); //_"+to_string(static_cast<int>(beta))+"
	save_output(dir+"output.dat", step_of_update, actions_list, num_non_zero_deg_evo, STARTING_EDGES, id_dist_list, ID_list, edge_index);
	return 0;
}
#include "src/actions.cpp"

// test update of number of points at given distance works
/*
vector<size_t> counter(8,0);

update_dist(distances_0,n_tot_0,distances_1, n_tot_1, 4,6,true);
print_mat(distances_0,8);
print_mat(distances_1,8);

for (size_t i = 0; i < 12; i++){
	for (size_t j = 0; j < 12; j++)
	{
		counter[ distances_1[i][j] ]+=1;
	}
}
for (size_t i = 0; i < 8; i++)
	std::cout << n_tot_0[i] << ' ' << n_tot_1[i] << ' ' << counter[i] << endl;
update_dist(distances_1,n_tot_1,distances_1, n_tot_1, 4,6,true);
update_dist(distances_1, n_tot_1, distances_0, n_tot_0, 3,6,false);
print_mat(distances_0, 8);
print_mat(distances_1, 8);

vector<size_t> counter1(8,0);
for (size_t i = 0; i < 8; i++){
	for (size_t j = 0; j < 8; j++)
	{
		counter1[ distances_0[i][j] ]+=1;
	}
}
for (size_t i = 0; i < 8; i++)
	std::cout << n_tot_1[i] << ' ' << n_tot_0[i] << ' ' << counter[i] << endl;
*/

// testers
/*
update_dist(distances_0, distances_1, n_tot_0, 10,11,true);
compute_id(ID_0, n_tot_0, n_cum, vol_ratios);
ID_list[0] = ID_0;
id_dist_list[0] = compute_id_dist(ID_ref,ID_0, MAX_DIST, false);

for (size_t i = 0; i < 12; i++){
	for (size_t j = 0; j < 12; j++)
	{
		std::cout << distances_1[i][j] << ' ';
	}
	std::cout << endl;
}

update_dist(distances_1, distances_0, n_tot_0, 2,9,false);
compute_id(ID_0, n_tot_0, n_cum, vol_ratios);
ID_list[1] = ID_0;
id_dist_list[1] = compute_id_dist(ID_ref,ID_0, MAX_DIST, false);

for (size_t i = 0; i < 12; i++){
	for (size_t j = 0; j < 12; j++)
	{
		std::cout << distances_0[i][j] << ' ';
	}
	std::cout << endl;
}


print_mat(distances_0,50);
vector<size_t> counter(70,0);
for (size_t k = 0; k < distances_0.size(); k++){
	for (size_t j = 0; j < k; j++)
	{
		counter[ distances_0[k][j] ]+=2;
	}
}
for (size_t k = 0; k < 25; k++){
	std::cout << n_tot_0[k] << ' ' << counter[k] << endl;
	counter[k]=0;
}

*/
