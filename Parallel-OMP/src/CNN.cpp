#include "CNN.h"


void CNN::add_conv(vector<int>& image_dim, vector<int>& kernels, int padding, int stride, double bias, double eta){

	int* dim_ptr = &image_dim[0];
	int* ker_ptr = &kernels[0];

	Convolutional element(dim_ptr, ker_ptr, padding, stride, bias, eta);	
	_convs.push_back( element );
	_layers.push_back('C');
	_tot_layers++;
}


void CNN::add_dense(int input, vector<int>& hidden, int num_classes, double bias, bool adam, double eta){

	vector<int> layers(hidden); // Obtain a copy (since hidden can change outside)
	layers.insert(layers.begin(),input);
	layers.push_back(num_classes);

	MultiLayerPerceptron element(layers, bias, adam, eta);

	_dense.push_back(element);
	_num_classes=num_classes;
	_layers.push_back('D');
	_tot_layers++;

}



void CNN::load_dataset(string data_name ){
	
	if(strcmp(data_name.c_str(),"MNIST")==0){

		MNIST mnist;
		
		mnist.get_mnist(Train_DS, Train_L, Test_DS, Test_L, Valid_DS, Valid_L);
		_image_shape[2]=Train_DS.get_shape(3);
		_image_shape[1]=Train_DS.get_shape(2);
		_image_shape[0]=Train_DS.get_shape(1);
	}
	

	
}



void CNN::_forward(volume& image){

	volume img_out;

	for(int i=0; i<_tot_layers; i++){
		
		if(_layers[i]=='C'){
			_convs[_conv_index].fwd(image,img_out);

			_conv_index++;
			image=img_out;			
		} else if(_layers[i]=='P') {
		} else if(_layers[i]=='D') {

			if(_dense_input_shape[0]==0){
				for(int i=0; i<3; i++) 
					_dense_input_shape[i]=image.get_shape(i);
			}

			_result=_dense[0].run(image.get_vector());

		}


	}

}



void CNN::_backward(vector<double>& gradient){
	
	volume img_out, img_in;  

	for(int i=_tot_layers-1; i>=0; i--){
		
		
		if(_layers[i]=='C'){
			_conv_index--;
			_convs[_conv_index].bp(img_in, img_out);	
			img_in=img_out;	
		}
		else if(_layers[i]=='P'){}
		else if(_layers[i]=='D'){	
			gradient=_dense[0].bp(gradient);	
			
			img_in.init(_dense_input_shape,3);
			img_in.get_vector()=gradient;
		}
	}
}


void CNN::_get_image(volume& image, volume& dataset, int index){

	double val;

	image.rebuild(_image_shape, 3);

	for(int d=0; d<_image_shape[0]; ++d)
	{
		for(int c=0;c<_image_shape[1];++c)
		{   
			for(int r=0;r<_image_shape[2];++r){

				int index_ds[4]  = {index,d,r,c};
				int index_im[3]  = {d,r,c};
				val=dataset.get_value(index_ds,4);
				image.assign(val,index_im,3);
			}
		}
	}

}


void CNN::_iterate(volume& dataset, vector<int>& labels, int batch_size, vector<double>& loss_list, vector<double>& acc_list, int preview_period, bool b_training ){

		int label = 0; 
		double accuracy = 0, loss = 0, correctAnswer = 0;
		volume image;
		
		double total = 0.0;

		int DS_len = dataset.get_shape(0);

		vector<double> total_error(_num_classes,0);
		double batch_count = 0;

		for(int sample=0; sample<DS_len; sample++ ){
			auto sample_start = chrono::high_resolution_clock::now();	
			_get_image(image, dataset, sample);	
			label = labels[sample];

			//feed the sample into the network 
			//The result is stored in _result
			_conv_index=0;
			auto fp_start = chrono::high_resolution_clock::now();
			_forward(image);	//--> _result
			auto fp_end = chrono::high_resolution_clock::now();
			long fp_time = chrono::duration_cast<chrono::microseconds>(fp_end - fp_start).count(); 

			//Error evaluation:
			vector<double> y(_num_classes,0), error(_num_classes,0);
			y[label]=1;

			batch_count++;
			for(int i=0; i<_num_classes; i++) {
				error[i] = y[i] - _result[i];
				total_error[i] += error[i];
			}
			
			// Cross entropy loss	
			loss = -log(_result[label]);

			loss_list.push_back( loss );
			
			int prediction=0;
			for(int i=0; i<_num_classes; i++){
				if(_result[i]>_result[prediction]) prediction=i;
			}
			
			if ( (int) prediction == label) correctAnswer++;

			// update accuracy
			accuracy = correctAnswer * 100 / ( sample + 1 );
			acc_list.push_back( accuracy );

			//adjust the weight
			long bp_time = 0;
			if (b_training && ((sample+1)%batch_size==0 || sample+1==DS_len)) {
				for(int i=0; i<_num_classes; i++)
					total_error[i] = total_error[i]/batch_count;

				auto bp_start = chrono::high_resolution_clock::now();
				_backward(total_error);
				auto bp_end = chrono::high_resolution_clock::now();
				bp_time = chrono::duration_cast<chrono::microseconds>(bp_end - bp_start).count();

				for(int i=0; i<_num_classes; i++)
					total_error[i] = 0;
				batch_count = 0;
			}

			auto sample_end = chrono::high_resolution_clock::now();
			long sample_time = chrono::duration_cast<chrono::microseconds>(sample_end - sample_start).count();

			total += sample_time * 1e-6;

			if((sample+1)%preview_period==0 || sample+1==DS_len) {
				printf("\t  [%s] Accuracy: %02.2f - Loss: %02.6f - Sample %04d  ||  Lable: %d - Prediction: %d  || Total time: %02.2f s FP: %ld us BP: %ld us Sample: %ld us \t\r", b_training ? "Train" : "Valid", accuracy, loss, sample+1, label, (int)prediction, total, fp_time, bp_time, sample_time);
			}
		}

}


		
//prew_freq is used to print every preview_period iterations the evolution of performances
void CNN::training( int epochs, int preview_period, int batch_size, bool validation){
		
	if(_tot_layers==0) cerr<<"Error: the network has no layers."<<endl;

	else{

		cout<<"\n\n\no Training: "<<endl;

		for(int epoch=0; epoch<epochs; epoch++){
			
			cout<< "\n\to Epoch "<<epoch+1 <<endl;
			//Batch Size = 1 => stochastic gradient descent learning algorithm
			_iterate(Train_DS, Train_L, batch_size, train_loss, train_acc, preview_period, true);
			cout << endl;
			/*
			cout<<("\nValidating:\n")<<endl;
			//the model evaluation is performed on the validation set after every epoch	
			_iterate(valid, valid_loss, valid_acc, false);
			*/
			if(validation)
				_iterate(Test_DS, Test_L, batch_size, valid_loss, valid_acc, preview_period, false);
			cout << endl;
		}
	}
}


//prew_freq is used to print every preview_period iterations the evolution of performances
void CNN::testing(int preview_period ){
		
	if(_tot_layers==0) cerr<<"Error: the network has no layers."<<endl;

	else{
		cout<<("\n\no Testing:")<<endl;
		//evaluate the performances on the test dataset
		_iterate(Test_DS, Test_L, 1, test_loss, test_acc, 100000,false);
	}
}



void CNN::sanity_check(int set_size ,int epochs ){

	if (_tot_layers==0)	cerr<<"Error: the network has no layers."<<endl;
	else{
		
		cout<<("\no Performing sanity check:\n")<<endl;

		vector<int> check_L;
		vector<double> check_loss, check_acc;

		volume check_DS(set_size, _image_shape[0], _image_shape[1], _image_shape[2]);

		for(int sample=0; sample<set_size; sample++ ){
			double val;
			for(int d=0; d<_image_shape[0]; ++d)
				for(int c=0;c<_image_shape[1];++c)
					for(int r=0;r<_image_shape[2];++r){
						int index[4]  = {sample,d,r,c};
						val=Test_DS.get_value(index,4);
						check_DS.assign(val,index,4);
					}
			check_L.push_back(Test_L[sample]);
		}

		for (int epoch=0;epoch<epochs;epoch++) {
			check_loss.clear();
			check_acc.clear();
			printf("\r\to Epoch %d  ||", (epoch+1));
			_iterate(check_DS, check_L, 1, check_loss, check_acc, (set_size-1), true);
		}

		double loss_avg = 0.0;
		for(int i=0; i<(int)check_loss.size();i++) loss_avg+=check_loss[i]/check_loss.size();

		printf("\n\n\tFinal losses: %02.6f", loss_avg );
	}
}



CNN::~CNN(){
	cout<<"\n\n\no Done"<<endl;
}