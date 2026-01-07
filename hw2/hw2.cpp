#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
#include <mpi.h>

#include "image.hpp"
#include "sift.hpp"


int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Only rank 0 handles I/O and prints messages
    if (world_rank == 0) {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(NULL);
    }

    if (argc != 4) {
        if (world_rank == 0) {
            std::cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_img = argv[1];
    std::string output_img = argv[2];
    std::string output_txt = argv[3];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // rank 0 讀取圖片，並且幫圖片轉灰階
    Image img;
    if (world_rank == 0) {
        img = Image(input_img);
        img = img.channels == 1 ? img : rgb_to_grayscale(img);
    }
    
    // rank 0 廣播圖片到所有rank
    mpi_broadcast_image(img, 0, MPI_COMM_WORLD);
    
    // Compute octave partition
    const int total_octaves = N_OCT;
    std::vector<int> octave_starts, octave_counts;
    compute_octave_partition(total_octaves, world_size, octave_starts, octave_counts);
    
    int my_start_octave = octave_starts[world_rank]; //取出當前這個 rank 的起始 octave」。
    int my_num_octaves = octave_counts[world_rank]; //取出當前這個 rank 要做幾個 octave
    
    // Each rank processes its assigned octaves
    std::vector<Keypoint> local_kps;
    
    // On-demand pyramid construction: each rank only builds up to its maximum octave
    // This avoids redundant construction while maintaining octave dependencies
    int octaves_to_build = (world_size == 1) ? total_octaves : (my_start_octave + my_num_octaves);
    
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(
        img, SIGMA_MIN, octaves_to_build, N_SPO);
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    
    if (world_size == 1) {
        // Single process: process all octaves
        local_kps = find_keypoints_range(dog_pyramid, grad_pyramid, 
                                        0, total_octaves,
                                        C_DOG, C_EDGE, LAMBDA_ORI, LAMBDA_DESC);
    } else if (my_num_octaves > 0) {
        // Multi-process: each processes only assigned octaves
        local_kps = find_keypoints_range(dog_pyramid, grad_pyramid, 
                                        my_start_octave, my_num_octaves,
                                        C_DOG, C_EDGE, LAMBDA_ORI, LAMBDA_DESC);
    }
    
    // Gather all keypoints to rank 0
    std::vector<Keypoint> kps = mpi_gather_keypoints(local_kps, 0, MPI_COMM_WORLD);


    // 只讓 rank0 寫檔/存圖/印時間
    if (world_rank == 0) {
        /////////////////////////////////////////////////////////////
        // The following code is for the validation
        // You can not change the logic of the following code, because it is used for judge system
        std::ofstream ofs(output_txt);
        if (!ofs) {
            std::cerr << "Failed to open " << output_txt << " for writing.\n";
        } else {
            ofs << kps.size() << "\n";
            for (const auto& kp : kps) {
                ofs << kp.i << " " << kp.j << " " << kp.octave << " " << kp.scale << " ";
                for (size_t i = 0; i < kp.descriptor.size(); ++i) {
                    ofs << " " << static_cast<int>(kp.descriptor[i]);
                }
                ofs << "\n";
            }
            ofs.close();
        }

        Image result = draw_keypoints(img, kps);
        result.save(output_img);
        /////////////////////////////////////////////////////////////

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Execution time: " << duration.count() << " ms\n";
        
        std::cout << "Found " << kps.size() << " keypoints.\n";
    }
    
    // Ensure all processes finish together
    MPI_Finalize();
    return 0;
}