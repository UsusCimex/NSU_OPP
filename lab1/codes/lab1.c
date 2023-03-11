#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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
    double *x = (double *)malloc(SIZE * sizeof(double));
    double *b = (double *)malloc(SIZE * sizeof(double));
    double *A = (double *)malloc(SIZE * SIZE * sizeof(double));
    double *Ax = (double *)malloc(SIZE * sizeof(double));
    double *u = (double *)malloc(SIZE * sizeof(double));
    double *tempVector = (double *)malloc(SIZE * sizeof(double));

    double eps = 1e-4;
    double tao = 1e-3;

    SetValues(A, u, x, b);

    clock_t startTime, endTime;
    startTime = clock();

    int state = 1;
    while (state)
    {
        MultiplyMatrixAndVector(A, x, Ax);
        MinusVectors(Ax, b, tempVector);
        MultiplyScalarAndVector(tempVector, tao, tempVector);
        MinusVectors(x, tempVector, x);

        MinusVectors(x, b, tempVector);
        if (CheckCriteria(tempVector, x, eps))
        {
            state = 0;
        }
    }

    endTime = clock();

    printf("%lf seconds have passed.\n", (double)(endTime - startTime) / (double)CLOCKS_PER_SEC);

    ///////
    printf("==========RESULT============\n");
    PrintVector(u, Min(15, SIZE));
    PrintVector(x, Min(15, SIZE));
    printf("============================\n");
    ///////

    free(u);
    free(A);
    free(Ax);
    free(tempVector);
    free(x);
    free(b);

    return 0;
}