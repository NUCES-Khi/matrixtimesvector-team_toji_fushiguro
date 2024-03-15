#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <time.h> 

#define MAX_SIZE 32768
#define TILE_SIZE 16

double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
        matrix[i] = (double*)malloc(cols * sizeof(double));
    return matrix;
}

double* allocate_vector(int size) {
    return (double*)malloc(size * sizeof(double));
}

void fill_random(double** matrix, double* vector, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
    for (int i = 0; i < cols; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

void sequential_mvm(double** matrix, double* vector, double* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void openmp_mvm(double** matrix, double* vector, double* result, int rows, int cols) {
#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void mpi_mvm(double** matrix, double* vector, double* result, int rows, int cols, int rank, int size) {
    int local_rows = rows / size;
    double* local_matrix = (double*)malloc(local_rows * cols * sizeof(double));
    double* local_result = (double*)malloc(local_rows * sizeof(double));
    MPI_Scatter(matrix[0], local_rows * cols, MPI_DOUBLE, local_matrix, local_rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(vector, local_rows, MPI_DOUBLE, local_result, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < local_rows; i++) {
        result[i + rank * local_rows] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i + rank * local_rows] += local_matrix[i * cols + j] * local_result[j];
        }
    }
    free(local_matrix);
    free(local_result);
    MPI_Gather(MPI_IN_PLACE, local_rows, MPI_DOUBLE, result, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void openmp_tiled_mvm(double** matrix, double* vector, double* result, int rows, int cols, int tile_size) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i += tile_size) {
        for (int j = 0; j < cols; j += tile_size) {
            for (int ii = i; ii < i + tile_size; ii++) {
                for (int jj = j; jj < j + tile_size; jj++) {
                    result[ii] += matrix[ii][jj] * vector[jj];
                }
            }
        }
    }
}

void mpi_tiled_mvm(double** matrix, double* vector, double* result, int rows, int cols, int rank, int size, int tile_size) {
    int local_rows = rows / size;
    double* local_matrix = (double*)malloc(local_rows * cols * sizeof(double));
    double* local_result = (double*)malloc(local_rows * sizeof(double));
    MPI_Scatter(matrix[0], local_rows * cols, MPI_DOUBLE, local_matrix, local_rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(vector, local_rows, MPI_DOUBLE, local_result, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < local_rows; i += tile_size) {
        for (int j = 0; j < cols; j += tile_size) {
            for (int ii = i; ii < i + tile_size && ii < local_rows; ii++) {
                for (int jj = j; jj < j + tile_size && jj < cols; jj++) {
                    result[i + rank * local_rows] += local_matrix[ii * cols + jj] * local_result[jj];
                }
            }
        }
    }
    free(local_matrix);
    free(local_result);
    MPI_Gather(MPI_IN_PLACE, local_rows, MPI_DOUBLE, result, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL)); 

    for (int N = 64; N <= MAX_SIZE; N *= 2) {
        double** matrix = allocate_matrix(N, N);
        double* vector = allocate_vector(N);
        double* result = allocate_vector(N);

        fill_random(matrix, vector, N, N);

        double start_time, end_time;

        if (rank == 0) {
            start_time = MPI_Wtime();
        }
        sequential_mvm(matrix, vector, result, N, N);
        if (rank == 0) {
            end_time = MPI_Wtime();
            printf("Sequential, %d, %.6f\n", N, end_time - start_time);
        }

        if (rank == 0) {
            start_time = MPI_Wtime();
        }
        openmp_mvm(matrix, vector, result, N, N);
        if (rank == 0) {
            end_time = MPI_Wtime();
            printf("OpenMP, %d, %.6f\n", N, end_time - start_time);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            start_time = MPI_Wtime();
        }
        mpi_mvm(matrix, vector, result, N, N, rank, size);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            end_time = MPI_Wtime();
            printf("MPI, %d, %.6f\n", N, end_time - start_time);
        }

        if (rank == 0) {
            start_time = MPI_Wtime();
        }
        openmp_tiled_mvm(matrix, vector, result, N, N, TILE_SIZE);
        if (rank == 0) {
            end_time = MPI_Wtime();
            printf("OpenMP Tiled, %d, %.6f\n", N, end_time - start_time);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            start_time = MPI_Wtime();
        }
        mpi_tiled_mvm(matrix, vector, result, N, N, rank, size, TILE_SIZE);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            end_time = MPI_Wtime();
            printf("MPI Tiled, %d, %.6f\n", N, end_time - start_time);
        }

        for (int i = 0; i < N; i++) {
            free(matrix[i]);
        }
        free(matrix);
        free(vector);
        free(result);
    }

    MPI_Finalize();
    return 0;
}

