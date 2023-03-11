#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi/mpi.h"
# define M_PI		3.14159265358979323846	/* pi */

#define SIZE 1000

void MultiplyScalarAndVector(double *vector, double scalar, double *resVector)
{
    for (int i = 0; i < SIZE; ++i)
    {
        resVector[i] = vector[i] * scalar;
    }
}

void PlusVectors(double *vector1, double *vector2, double *resVector)
{
    for (int i = 0; i < SIZE; ++i)
    {
        resVector[i] = vector1[i] + vector2[i];
    }
}

void MinusVectors(double *vector1, double *vector2, double *resVector)
{
    for (int i = 0; i < SIZE; ++i)
    {
        resVector[i] = vector1[i] - vector2[i];
    }
}

void MultiplyMatrixAndVector(double *matrix, double *vector, double *resVector)
{

    for (int i = 0; i < SIZE; ++i)
    {
        double sum = 0;
        for (int j = 0; j < SIZE; ++j)
        {
            sum += matrix[i * SIZE + j] * vector[j];
        }
        resVector[i] = sum;
    }
}

double GetNorm(double *vector)
{
    double sum = 0;
    for (int i = 0; i < SIZE; ++i)
    {
        sum += vector[i] * vector[i];
    }
    sum = sqrt(sum);
    return sum;
}

double MultiplyVectors(double *vector1, double *vector2)
{
    double res = 0;
    for (int i = 0; i < SIZE; ++i)
    {
        res += vector1[i] * vector2[i];
    }
    return res;
}

int Min(int a, int b)
{
    return a < b ? a : b;
}

// Return 1 if succses, 0 else
int CheckCriteria(double *vector1, double *vector2, double eps)
{
    if ((GetNorm(vector1) / GetNorm(vector2)) < eps)
        return 1;
    return 0;
}

void PrintVector(double *vector, int count)
{
    for (int i = 0; i < count; ++i)
    {
        printf("%lf ", vector[i]);
    }
    printf("\n");
}

void SetValues(double *A, double *u, double *x, double *b)
{
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            if (i == j)
                A[i * SIZE + j] = 2.0;
            else
                A[i * SIZE + j] = 1.0;
        }
    }

    for (int i = 0; i < SIZE; ++i)
    {
        u[i] = sin(2 * M_PI * i / SIZE);
        x[i] = 0;
    }

    MultiplyMatrixAndVector(A, u, b);
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Status status;
    double startTime, endTime;

    double eps = 1e-4;
    double tao = 1e-3;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int capacity = SIZE / size;
    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size - 1; ++i) {
        sendcounts[i] = capacity * SIZE;
        displs[i] = i * capacity * SIZE;
    }
    sendcounts[size - 1] = (capacity + SIZE % size) * SIZE;
    displs[size - 1] = (size - 1) * capacity * SIZE;

    double* A = NULL;
    double* u = NULL;
    double* x = (double*)malloc(SIZE * sizeof(double));
    double* b = (double*)malloc(SIZE * sizeof(double));
    double* Ax = NULL;
    double* tempVector = NULL;

    double* partA = (double*)malloc(sendcounts[rank] * sizeof(double));
    if (rank == 0) {
        A = (double*)malloc(SIZE * SIZE * sizeof(double));
        u = (double*)malloc(SIZE * sizeof(double));
        Ax = (double*)malloc(SIZE * sizeof(double));
        tempVector = (double*)malloc(SIZE * sizeof(double));

        SetValues(A, u, x, b);
        startTime = MPI_Wtime();
    }

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, partA, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < size; ++i) {
        sendcounts[i] /= SIZE;
        displs[i] /= SIZE;
    }

    int state = 1;
    while(state) {
        MPI_Bcast(x, SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int i = 0; i < sendcounts[rank]; i++) {
            if (rank == 0) Ax[i] = MultiplyVectors(partA + i * SIZE, x);
            else b[i] = MultiplyVectors(partA + i * SIZE, x);
        }
        if (rank == 0) {
            for (int i = 1; i < size; ++i) {
                MPI_Recv(Ax + i * capacity, sendcounts[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            }
        }
        else {
            MPI_Send(b, sendcounts[rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            MinusVectors(Ax, b, tempVector);
            MultiplyScalarAndVector(tempVector, tao, tempVector);
            MinusVectors(x, tempVector, x);
        
            MinusVectors(x, b, tempVector);
            if (CheckCriteria(tempVector, x, eps))
            {
                state = 0;
            }
        }
        MPI_Bcast(&state, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        endTime = MPI_Wtime();
        printf("%lf seconds have passed.\n", endTime - startTime);

        ///////
        // printf("\n==========RESULT============\n");
        // PrintVector(u, Min(15, SIZE));
        // PrintVector(x, Min(15, SIZE));
        // printf("============================\n");
        ///////

        free(A);
        free(u);
        free(Ax);
        free(tempVector);
    }

    free(x);
    free(b);
    free(partA);
    free(sendcounts);
    free(displs);

    MPI_Finalize();

    return 0;
}