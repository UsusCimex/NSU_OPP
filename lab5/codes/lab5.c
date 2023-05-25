#include <stdio.h>
#include <pthread.h>
#include <mpi/mpi.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#define N_TASKS 320
#define REQUEST_TAG 0
#define TASK_TAG 1

typedef struct Task {
    int taskNumber;
    int difficulty;
    char completed;
} Task;

pthread_mutex_t mutex;
Task* task_list;

void calculate_task(Task* task, int rank) {
    pthread_mutex_lock(&mutex);
    if (task->completed == 0) {
        // printf("Worker %d: Start task %d(%d)\n", rank, task->taskNumber, task->difficulty);
        task->completed = 1;
        pthread_mutex_unlock(&mutex);
        usleep(task->difficulty); //time in mcs
    }
}

int request_task(int request_rank) {
    int task_id = -1;
    int request = 1;
    MPI_Send(&request, 1, MPI_INT, request_rank, REQUEST_TAG, MPI_COMM_WORLD);
    MPI_Recv(&task_id, 1, MPI_INT, request_rank, TASK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return task_id;
}

void* worker_thread(void* args) {
    int rank = *((int*)args);
    int size = *((int*)args + 1);
    
    while (1) {
        Task* task_to_do = NULL;
        for (int i = rank * N_TASKS / size; i <  rank * N_TASKS / size + N_TASKS / size; ++i) {
            pthread_mutex_lock(&mutex);
            if (!task_list[i].completed) {
                task_to_do = &task_list[i];
                pthread_mutex_unlock(&mutex);
                break;
            }
            pthread_mutex_unlock(&mutex);
        }
        
        if (task_to_do == NULL) {
            // printf("Worker %d: local tasks complete, start request...\n", rank);
            for (int i = 0; i < size; ++i) {
                if (i != rank) {
                    // printf("Worker %d: connect to %d\n", rank, i);
                    int task_id = request_task(i);
                    if (task_id != -1) {
                        task_to_do = &task_list[task_id];
                        break;
                    }
                }
            }

            if (task_to_do == NULL) {
                // printf("Worker %d: BARRIER\n", rank);
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Send(NULL, 0, MPI_INT, rank, REQUEST_TAG, MPI_COMM_WORLD);
                break;
            }
        }

        calculate_task(task_to_do, rank);
    }
    return NULL;
}

void* server_thread(void* args) {
    int rank = *((int*)args);
    int size = *((int*)args + 1);
    int request;
    MPI_Status status;
    
    while (1) {
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);
        // printf("Server %d: %d connected\n", rank, status.MPI_SOURCE);
        if (status.MPI_SOURCE == rank) {
            break;
        }

        int task_id = -1;
        for (int i = rank * N_TASKS / size; i <  rank * N_TASKS / size + N_TASKS / size; ++i) {
            pthread_mutex_lock(&mutex);
            if (task_list[i].completed == 0) {
                task_id = i;
                task_list[i].completed = 1;
                pthread_mutex_unlock(&mutex);
                break;
            }
            pthread_mutex_unlock(&mutex);
        }
        // printf("Server %d: send %d to %d\n", rank, task_id, status.MPI_SOURCE);
        MPI_Send(&task_id, 1, MPI_INT, status.MPI_SOURCE, TASK_TAG, MPI_COMM_WORLD);
    }
    return NULL;
}

int main(int argc, char* argv[]) {
    int rank, size;
    int provided;
    pthread_attr_t attrs;
    pthread_t threads[2];
    double start_time, end_time;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided != MPI_THREAD_MULTIPLE)
    {
        printf("Can't init thread\n");
        MPI_Finalize();
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int check = 0;
    check = pthread_mutex_init(&mutex, NULL);
    if (check != 0)
    {
        printf("Can't init mutex\n");
        MPI_Finalize();
        return -1;
    }

    check = pthread_attr_init(&attrs);
    if (check != 0)
    {
        printf("Can't init attrs\n");
        MPI_Finalize();
        return -1;
    }

    check = pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);
    if (check != 0)
    {
        printf("Can't set attrs\n");
        MPI_Finalize();
        return -1;
    }

    task_list = (Task*)malloc(sizeof(Task) * N_TASKS);
    if (rank == 0) {
        srand(21212);
        for (int i = 0; i < N_TASKS; ++i) {
            task_list[i].taskNumber = i;
            task_list[i].difficulty = (i + 1) * 1000;
            task_list[i].completed = 0;
        }
    }
    MPI_Bcast(task_list, N_TASKS*sizeof(Task), MPI_BYTE, 0, MPI_COMM_WORLD);

    int* threadArgs = (int*)malloc(sizeof(int) * 2);
    threadArgs[0] = rank;
    threadArgs[1] = size;
    pthread_create(&threads[0], &attrs, worker_thread, (void*)threadArgs);
    pthread_create(&threads[1], &attrs, server_thread, (void*)threadArgs);

    start_time = MPI_Wtime();

    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);

    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Tasks completed! Time left: %lf\n", end_time - start_time);
        int sumDifficulties = 0;
        for (int i = 0; i < N_TASKS; ++i) {
            sumDifficulties += task_list[i].difficulty;
        }
        printf("All tasks difficulty: %lf\n", sumDifficulties / 1000000.0);
        printf("All difficulty / N: %lf\n", sumDifficulties / 1000000.0 / size);
        printf("Difference: %lf\n", fabs(sumDifficulties / 1000000.0 / size - end_time + start_time));
    }

    free(threadArgs);
    free(task_list);

    pthread_mutex_destroy(&mutex);
    pthread_attr_destroy(&attrs);

    MPI_Finalize();
    return 0;
}
