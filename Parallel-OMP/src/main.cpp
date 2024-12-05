
#include <iostream>
#include <vector>
#include <omp.h>
#include "CNN.h"

using namespace std;


bool        adam;
double      bias, eta;
vector<int> image_1{1,28,28}, kernels_1{8,3,3,1};
vector<int> image_2{8,13,13}, kernels_2{2,3,3,8},  hidden{72};
int         input_layer, num_classes, epochs, padding, stride;


int main(int argc, char ** argv){
    // Default values 
    int num_epochs = 1;           
    bool sanity_check = false;
    int preview_period = 10;
    int batch_size = 1;

    const char USAGE_MESSAGE[] = 
    "Usage: ./CNN [--num_epochs <int>] [--sanity_check <bool>] [--preview_period <int>] [--batch_size <int>]\n"
    "Default values:\n"
    "  --num_epochs: 1\n"
    "  --sanity_check: false\n"
    "  --preview_period: 10\n"
    "  --batch_size: 1\n";

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--num_epochs" && i + 1 < argc) {
            num_epochs = std::stoi(argv[i + 1]);
            i++; // Skip the next argument as it's part of --num_epochs
        } else if (arg == "--sanity_check" && i + 1 < argc) {
            std::string value = argv[i + 1];
            sanity_check = (value == "true" || value == "1"); // Allow "true"/"1" as true values
            i++; // Skip the next argument as it's part of --sanity_check
        } else if (arg == "--preview_period" && i + 1 < argc) {
            std::string value = argv[i + 1];
            preview_period = std::stoi(argv[i + 1]);
            i++; // Skip the next argument as it's part of --preview_period
        } else if (arg == "--batch_size" && i + 1 < argc) {
            std::string value = argv[i + 1];
            batch_size = std::stoi(argv[i + 1]);
            i++; // Skip the next argument as it's part of --batch_size
        }
        else if (arg == "--nthreads" && i + 1 < argc) {
            int value = atoi(argv[i + 1]);
            omp_set_num_threads(value);
            i++; // Skip the next argument as it's part of --batch_size
        } else {
            std::cerr 
                << "Unknown argument: " << arg << std::endl 
                << USAGE_MESSAGE << std::endl;
            exit(1); 
        }
    }

    // Display the parsed values
    std::cout << "Number of epochs: " << num_epochs << std::endl;
    std::cout << "Sanity check: " << (sanity_check ? "enabled" : "disabled") << std::endl;
    std::cout << "Preview period: " << preview_period << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    #pragma omp parallel
    {
        if(omp_get_thread_num() == 0){
            int nthreads = omp_get_num_threads();
            std::cout << "Thread count: " << nthreads << std::endl;
        }
    }
    //network istantiation

    CNN network;

    //build the network 

    network.add_conv(image_1, kernels_1, padding= 0, stride= 2, bias= 0.1, eta= 0.01 );
    network.add_conv(image_2 , kernels_2 , padding= 0, stride= 2, bias= 0.1, eta= 0.01);
    network.add_dense(input_layer=2*6*6, hidden, num_classes=10, bias=1.0,  adam=false, eta=0.5);

    //load the wanted dataset

    network.load_dataset("MNIST");

    //sanity check
    if (sanity_check)
        network.sanity_check();

    //train the network (Batch Size = 1)

    network.training(num_epochs, preview_period, batch_size);

    //evaluate new samples 

    network.testing(10);


    return 0;

}
