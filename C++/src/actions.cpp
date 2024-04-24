#ifndef __actions__
#define __actions__

#include "func.hpp"

using namespace std;

tuple<int,int> degree_powerlaw(const vector<size_t>& degree, vector<size_t>& cumulative, vector<size_t>& cumulative1, double alpha=1.){

    // size_t appo{0};

    // source cumulative if we want the source to have at least deg=1
    partial_sum(degree.begin(), degree.end(), cumulative.begin());
    // target cumulative, even if degree=0
    //if (alpha != 1.){
    // appo = (degree[0]==0) ? 1 : degree[0];//;pow(static_cast<double>(degree[0]),alpha);
    // cumulative1[0] = appo;
    // for (size_t i = 1; i < degree.size(); i++){
    //     appo = (degree[i]==0) ? 1 : degree[i];//pow(static_cast<double>(degree[i]),alpha);
    //     cumulative1[i] = cumulative1[i-1] + appo;
    // }
   
    double appo0{gsl_rng_uniform(r)*cumulative[cumulative.size()-1]};
    // double appo1{gsl_rng_uniform(r)*cumulative1[cumulative1.size()-1]};
   	// bool found{false};
	// bool found1{false};
    // int source{-1};
    // int target{-1};

    // for (int aa = 0; aa < static_cast<int>(cumulative1.size()); aa++){
    // 	if (appo0 <= cumulative1[aa] && !found){
    // 		source = aa;
    // 		found = true;
    // 	}
    // 	if (appo1 <= cumulative1[aa] && !found1){
    // 		target = aa;
    // 		found1 = true;
    // 	}
    // 	if (found && found1)
    // 		break;					
    // }
    // return make_tuple(source, target);

    for (size_t aa = 0; aa < cumulative.size(); aa++){
        if(appo0 <= cumulative[aa])
            return make_tuple( static_cast<int>(aa),  gsl_rng_uniform_int(r, NUM_VERTEX) );
    }
    return make_tuple(-1,-1);
}

tuple<int,int> degree_neighbour(const vector<size_t>& degree, vector<size_t>& cumulative,
    const vector<int>& sources, const vector<int>& targets, const size_t& num_edges){

    fill(cumulative.begin(), cumulative.end(),0);
    for (size_t i = 0; i < num_edges; i++){
        cumulative[ sources[i] ] += degree[targets[i]];
    //    cumulative[ targets[i] ] += degree[sources[i]]; 
    }
    
    partial_sum(cumulative.begin(), cumulative.end(), cumulative.begin());

    // bool found{false};
	// bool found1{false};
    // int source{-1};
    // int target{-1};
    double appo0{gsl_rng_uniform(r)*cumulative[cumulative.size()-1]};
    // double appo1{gsl_rng_uniform(r)*cumulative[cumulative.size()-1]};

    // for (int aa = 0; aa < static_cast<int>(cumulative.size()); aa++){
    // 	if (appo0 <= cumulative[aa] && !found){
    // 		source = aa;
    // 		found = true;
    // 	}
    // 	if (appo1 <= cumulative[aa] && !found1){
    // 		target = aa;
    // 		found1 = true;
    // 	}
    // 	if (found && found1)
    // 		break;					
    // }

    // return make_tuple(source, target);

    for (size_t aa = 0; aa < cumulative.size(); aa++){
        if(appo0 <= cumulative[aa])
            return make_tuple( static_cast<int>(aa),  gsl_rng_uniform_int(r, NUM_VERTEX) );
    }
    return make_tuple(-1,-1);
}

tuple<int,int> triadic_closure(const vector<vector<size_t>>& distances, const size_t& n){

    size_t idx{gsl_rng_uniform_int(r,n/2)};
    size_t counter{0};
 
    for (size_t i = 0; i < distances.size(); i++){
        for (size_t j = 0; j < i; j++){
            if (distances[i][j] == 2){
                if (idx == counter)
                    return make_tuple(static_cast<int>(i),static_cast<int>(j));    
                counter++;
            }
        }
    }
    //cout <<"smtg went wrong\n";
    return make_tuple(-1,-1);
}

size_t extract_action(const vector<double>& ps_cum){
    double appo{gsl_rng_uniform(r)};
    for (size_t i = 0; i < ps_cum.size(); i++){
        if(appo <= ps_cum[i])
            return i;
    }
}

#endif