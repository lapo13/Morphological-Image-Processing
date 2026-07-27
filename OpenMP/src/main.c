#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "matrix.h"
#include "image.h"
#include "morphologies.h"

#define N 4 // Number of threads for OpenMP parallelization

int main() {
    matrix input_image;
    matrix structuring_element;

    input_image.rows = 5;
    input_image.cols = 3;
    structuring_element.rows = 3;
    structuring_element.cols = 3;

    matrix eroded_image;
    matrix dilated_image;
    matrix opened_image;
    matrix closed_image;

    matrix scratch;

    #pragma omp parallel num_threads(N)
    {
        #pragma omp master
        {
            printf("Number of threads: %d\n", omp_get_num_threads());
        }
        allocate_matrix(&input_image, input_image.rows, input_image.cols);
        #pragma omp single
        {
            srand(time(NULL));
        }

        #pragma omp for collapse(2)
        for (int i = 0; i < input_image.rows; i++) {
            for (int j = 0; j < input_image.cols; j++) {
                input_image.data[i][j] = rand() % 10;
            }
        }

        #pragma omp master
        {
        printf("Input Image:\n");
        }
        print_matrix(&input_image);

        allocate_matrix(&structuring_element, structuring_element.rows, structuring_element.cols);
        #pragma omp for collapse(2)
        for (int i = 0; i < structuring_element.rows; i++) {
            for (int j = 0; j < structuring_element.cols; j++) {
                structuring_element.data[i][j] = 1;
            }
        }

        #pragma omp master
        {
        printf("Structuring Element:\n");
        }
        print_matrix(&structuring_element);

        #pragma omp master
        {
        printf("Performing binary image erosion...\n");
        }
        copy_matrix(&input_image, &eroded_image);
        #pragma omp master
        {
        printf("Before Erosion:\n");
        }
        print_matrix(&eroded_image);

        image_erosion(&eroded_image, &structuring_element, &scratch);
        #pragma omp master
        {
        printf("Eroded Image:\n");
        }
        print_matrix(&eroded_image);

        #pragma omp master
        {
        printf("Performing binary image dilation...\n");
        }

        copy_matrix(&input_image, &dilated_image);
        #pragma omp master
        {
        printf("Before Dilation:\n");
        }
        print_matrix(&dilated_image);
        image_dilation(&dilated_image, &structuring_element, &scratch);
        #pragma omp master
        {
        printf("Dilated Image:\n");
        }
        print_matrix(&dilated_image);

        #pragma omp master
        {
        printf("Performin an Opening operation (Erosion followed by Dilation)...\n");
        }
        copy_matrix(&input_image, &opened_image);
        image_opening(&opened_image, &structuring_element, &scratch);
        #pragma omp master
        {
        printf("Opened Image:\n");
        }
        print_matrix(&opened_image);

        #pragma omp master
        {
        printf("Performin a Closing operation (Dilation followed by Erosion)...\n");
        }
        copy_matrix(&input_image, &closed_image);
        image_closing(&closed_image, &structuring_element, &scratch);
        #pragma omp master
        {
        printf("Closed Image:\n");
        }
        print_matrix(&closed_image);

        #pragma omp master
        {
        printf("Freeing allocated memory...\n");
        }

        free_matrix(&input_image);
        free_matrix(&structuring_element);

        free_matrix(&eroded_image);
        free_matrix(&dilated_image);
        free_matrix(&opened_image);
        free_matrix(&closed_image);
    }
    return 0;
}
