#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#include "matrix.h"
#include "image.h"
#include "morphologies.h"

int main() {
    matrix input_image;
    matrix structuring_element;

    input_image.rows = 5;
    input_image.cols = 3;
    structuring_element.rows = 3;
    structuring_element.cols = 3;

    allocate_matrix(&input_image, input_image.rows, input_image.cols);
    srand(time(NULL));

    for (int i = 0; i < input_image.rows; i++) {
        for (int j = 0; j < input_image.cols; j++) {
            input_image.data[i][j] = rand() % 10;
        }
    }

    printf("Input Image:\n");
    print_matrix(&input_image);

    allocate_matrix(&structuring_element, structuring_element.rows, structuring_element.cols);
    for (int i = 0; i < structuring_element.rows; i++) {
        for (int j = 0; j < structuring_element.cols; j++) {
            structuring_element.data[i][j] = 1;
        }
    }

    printf("Structuring Element:\n");
    print_matrix(&structuring_element);

    printf("Performing binary image erosion...\n");
    matrix eroded_image;
    copy_matrix(&input_image, &eroded_image);
    printf("Before Erosion:\n");
    print_matrix(&eroded_image);

    image_erosion(&eroded_image, &structuring_element);
    printf("Eroded Image:\n");
    print_matrix(&eroded_image);

    printf("Performing binary image dilation...\n");
    matrix dilated_image;
    copy_matrix(&input_image, &dilated_image);
    printf("Before Dilation:\n");
    print_matrix(&dilated_image);

    image_dilation(&dilated_image, &structuring_element);
    printf("Dilated Image:\n");
    print_matrix(&dilated_image);

    printf("Performin an Opening operation (Erosion followed by Dilation)...\n");
    matrix opened_image;
    copy_matrix(&input_image, &opened_image);
    image_opening(&opened_image, &structuring_element);
    printf("Opened Image:\n");
    print_matrix(&opened_image);

    printf("Performin a Closing operation (Dilation followed by Erosion)...\n");
    matrix closed_image;
    copy_matrix(&input_image, &closed_image);
    image_closing(&closed_image, &structuring_element);
    printf("Closed Image:\n");
    print_matrix(&closed_image); 

    printf("Freeing allocated memory...\n");

    free_matrix(&input_image);
    free_matrix(&structuring_element);
    
    free_matrix(&eroded_image);
    free_matrix(&dilated_image);
    free_matrix(&opened_image);
    free_matrix(&closed_image);

    return 0;
}