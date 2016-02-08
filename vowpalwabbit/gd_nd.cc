#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <float.h>
#include <string.h>
#include <vector>
#ifdef _WIN32
#include <winsock2.h>
#else
#include <netdb.h>
#endif
#include <algorithm>

using namespace std;
// using namespace LEARNER;

struct val_id{
	double val;
	size_t id;
};

typedef val_id vl_pair;

// bool comp_vl_pair(const void *a,const void *b){
// 	if(((vl_pair*)a)->val > ((vl_pair*)b)->val)
// 		return true;
// 	else
// 		return false;
// }

bool comp_vl_pair(const vl_pair &a,const vl_pair &b){
	if(a.val > b.val)
		return true;
	else
		return false;
}

int non_decomposable_gradient_descent( vector<double>& stream ,vector<double>& labels, vector<double>& w_init , vector<double>& w_bar , size_t num_data , size_t d,size_t e, double k, size_t tic_spacing, size_t num_tics, size_t pass_counter, size_t num_epochs){   
	int t_counter,t_counter_now,d_counter,update_counter;
	double C_t;
	vector<double> w(d,0),y_bar_x(d,0),y_ast_x(d,0),w_inter(d,0);
	vector<vl_pair> score_loc_pairs(e);

	int x_offset,x_temp_offset;
	int tic_counter, e_counter, inner_counter;
	int this_epoch_length, num_pos_this_epoch, num_to_set_pos;
	double time_elapsed;
	clock_t tic, toc;


	update_counter = num_epochs*(pass_counter - 1);	// Used to update the final weight vector after each update
	tic_counter=0;
	time_elapsed=0;
	t_counter=0;
	e_counter=0;

	C_t=0;

	//score_loc_pairs : Store the scores assigned to the positive and negative points along with their location
	//w : Intermediate weight vector
	//w_inter : Running Average weight vector
	//y_ast_x :  Store \sum y_i*x_i vector
	//y_bar_x : Store \sum \bar(y)_i*x_i vector

	// Initialize the weight vectors

	for (d_counter = 0; d_counter < d ; ++d_counter)
	{
		w[d_counter]=w_init[d_counter];
		w_inter[d_counter]=w_init[d_counter];
	}

	//Go over the data stream
	while(1){
		C_t=C_t/sqrt((pass_counter-1)*num_epochs+e_counter+1);

		if(t_counter+e>=num_data)
			this_epoch_length=num_data- t_counter;
		else
			this_epoch_length=e;

		//No more points left in the data stream
		if(this_epoch_length==0)
			break;
		//Update is necessary in this epoch
		update_counter++;

		//The no of positives in this epoch has to be recorded
		num_pos_this_epoch=0;

		//Start the clock
		tic=clock();

		for (d_counter = 0; d_counter < d ; ++d_counter)
		{
			y_ast_x[d_counter]=0;
			y_bar_x[d_counter]=0;
		}

		// 2 passes are required over the data stream hence store the current location
		t_counter_now=t_counter;

		for (inner_counter = 0; inner_counter < this_epoch_length ; ++inner_counter)
		{
			if(labels[t_counter]>0)
				num_pos_this_epoch++;
			t_counter++;
		}

		if(num_pos_this_epoch!=0){
			//Rewind to the beginning of the epoch
			t_counter=t_counter_now;

			//Collect data points in this epoch buffer and compute scores
			for (inner_counter = 0; inner_counter < this_epoch_length; ++inner_counter)
			{
				x_offset=d*t_counter;

				//Store the value of y_t*x_t
				for (d_counter = 0; d_counter < d; ++d_counter)
				{
					y_ast_x[d_counter]+=labels[t_counter]* stream[x_offset+d_counter];
				}

				score_loc_pairs[inner_counter].id=t_counter;
				score_loc_pairs[inner_counter].val=0;
				for (d_counter = 0;d_counter < d ; ++d_counter)
				{
					score_loc_pairs[inner_counter].val+=w[d_counter]*stream[x_offset+d_counter];
				}

				if (k>0)
				{ 
					//For prec@k
					score_loc_pairs[inner_counter].val-=labels[t_counter]*this_epoch_length/(4*ceil(k*num_pos_this_epoch));
				}
				else{
					//For PRBEP
					if(num_pos_this_epoch==0)
						score_loc_pairs[inner_counter].val=0;
					else
						score_loc_pairs[inner_counter].val-=labels[t_counter]*this_epoch_length/(4*num_pos_this_epoch);
				}
				t_counter++;
			}
			sort(score_loc_pairs.begin(), score_loc_pairs.end(),comp_vl_pair);

			if (k>0)
				num_to_set_pos=ceil(k*num_pos_this_epoch);
			else
				num_to_set_pos=num_pos_this_epoch;

			for (inner_counter =this_epoch_length-1; inner_counter >= this_epoch_length - num_to_set_pos; inner_counter--)
			{
				x_temp_offset=d*score_loc_pairs[inner_counter].id;
				for (d_counter = 0; d_counter < d ; ++d_counter)
					y_bar_x[d_counter] += stream[x_temp_offset+d_counter];
			}

			for (; inner_counter >=0 ; --inner_counter)
			{
				x_temp_offset=d*score_loc_pairs[inner_counter].id;
				for (d_counter = 0; d_counter < d ; ++d_counter)
					y_bar_x[d_counter] -= stream[x_temp_offset+d_counter];
			}

			for (d_counter = 0; d_counter < d ; ++d_counter)
			{
				w[d_counter] -=C_t/this_epoch_length*(y_bar_x[d_counter]-y_ast_x[d_counter]);
			}

		}

		for (d_counter = 0; d_counter < d ; ++d_counter)
		{
			w_inter[d_counter]=(w_inter[d_counter]*(update_counter-1)+w[d_counter])/update_counter;		
		}

		toc=clock();

		//Time in doing useful work
		time_elapsed += (((double)toc-(double)tic)/((double)(CLOCKS_PER_SEC)));
		// If the time tic is appropriate, make a record of this vector

		if(e_counter%tic_spacing == 0){
			for(d_counter = 0; d_counter < d; d_counter++){
				w_bar[(d+1)*tic_counter+d_counter]=w_inter[d_counter];
			}
			w_bar[(d+1)*tic_counter+d]=time_elapsed;		
			tic_counter++;
		}
		e_counter++;
	}
	for (d_counter = 0; d_counter < d; ++d_counter)
	{
		w_bar[(d+1)*tic_counter+d_counter]=w_inter[d_counter];
	}
	w_bar[(d+1)*tic_counter+d]=time_elapsed;

	return 0;
}
