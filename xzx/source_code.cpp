#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>

#include <omp.h>
#include <immintrin.h>
#include <xmmintrin.h>

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

struct Edge
{
    float weight;
    int columnIndices;
};

struct CSRGraph
{
    vector<Edge> mergedArray;
    vector<int> rowPointers;
};

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

CSRGraph csrgraph;
vector<vector<int>> edge_index;
vector<int> degree;
vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

void readGraph(const char *fname)
{
    ifstream infile(fname);
    int source;
    int end;
    infile >> v_num >> e_num;
    while (!infile.eof())
    {
        infile >> source >> end;
        if (infile.peek() == EOF)
            break;
        raw_graph.push_back(source);
        raw_graph.push_back(end);
    }
}

void readFloat(char *fname, float *&dst, int num)
{
    dst = (float *)malloc(num * sizeof(float));
    FILE *fp = fopen(fname, "rb");
    fread(dst, num * sizeof(float), 1, fp);
    fclose(fp);
}

void initFloat(float *&dst, int num)
{
    dst = (float *)malloc(num * sizeof(float));
    memset(dst, 0, num * sizeof(float));
}

void Raw_Graph_To_CSR()
{
    vector<int> counts(v_num + 1, 0);
    vector<float> tmp_degree(v_num, 0);
    csrgraph.mergedArray.resize(e_num);
    csrgraph.rowPointers.resize(v_num + 1, 0);
    degree.resize(v_num, 0);

    int src, dst, i, pointer;

#pragma omp parallel for private(i, dst, src)
    for (i = 0; i < e_num; i++)
    {
        src = raw_graph[i * 2];
        dst = raw_graph[i * 2 + 1];
#pragma omp atomic
        csrgraph.rowPointers[src + 1]++;
    }

#pragma omp parallel for private(i)
    for (i = 0; i < v_num; i++)
    {
        degree[i] = csrgraph.rowPointers[i + 1];
        tmp_degree[i] = 1.0f / sqrt(degree[i]);
    }
    for (i = 1; i <= v_num; i++)
    {
        csrgraph.rowPointers[i] = csrgraph.rowPointers[i - 1] + csrgraph.rowPointers[i];
    }
#pragma omp for private(i)
    for (i = 1; i <= v_num; i++)
    {
        counts[i] = csrgraph.rowPointers[i];
    }

#pragma omp parallel for private(i, dst, src, pointer)
    for (i = 0; i < e_num; i++)
    {
        src = raw_graph[i * 2];
        dst = raw_graph[i * 2 + 1];
        pointer = counts[src];
#pragma omp atomic
        counts[src]++;
        csrgraph.mergedArray[pointer].columnIndices = dst;
        csrgraph.mergedArray[pointer].weight = tmp_degree[src] * tmp_degree[dst];
    }
}

void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{
    float(*tmp_in_X)[in_dim] = (float(*)[in_dim])in_X;
    float(*tmp_out_X)[out_dim] = (float(*)[out_dim])out_X;
    float(*tmp_W)[out_dim] = (float(*)[out_dim])W;
    int i, j, k;
#pragma omp parallel for simd private(i, j, k)
    for (i = 0; i < v_num; i++)
    {
        for (k = 0; k < in_dim; k++)
        {
            for (j = 0; j < out_dim; j++)
            {
                tmp_out_X[i][j] += tmp_in_X[i][k] * tmp_W[k][j];
            }
        }
    }
}

void AX(int dim, float *in_X, float *out_X)
{
    float(*tmp_in_X)[dim] = (float(*)[dim])in_X;
    float(*tmp_out_X)[dim] = (float(*)[dim])out_X;
    int i, start, k;
    Edge nbr;
#pragma omp parallel for simd private(i, k, start, nbr)
    for (i = 0; i < v_num; i++)
    {
        for (start = csrgraph.rowPointers[i]; start < csrgraph.rowPointers[i + 1]; start++)
        {
            nbr = csrgraph.mergedArray[start];
            for (k = 0; k < dim; k++)
            {
                tmp_out_X[i][k] += tmp_in_X[nbr.columnIndices][k] * nbr.weight;
            }
        }
    }
}

void ReLU(int dim, float *X)
{
#pragma omp parallel for simd
    for (int i = 0; i < v_num * dim; i++)
        if (X[i] < 0)
            X[i] = 0;
}

void LogSoftmax(int dim, float *X)
{
    float(*tmp_X)[dim] = (float(*)[dim])X;
#pragma omp parallel for
    for (int i = 0; i < v_num; i++)
    {
        float max = tmp_X[i][0];
        for (int j = 1; j < dim; j++)
        {
            if (tmp_X[i][j] > max)
                max = tmp_X[i][j];
        }

        float sum = 0;
        float exp_sum[dim];
        for (int j = 0; j < dim; j++)
        {
            exp_sum[j] = expf(tmp_X[i][j] - max);
            sum += exp_sum[j];
        }
        sum = logf(sum);

#pragma omp simd
        for (int j = 0; j < dim; j++)
        {
            tmp_X[i][j] = tmp_X[i][j] - max - sum;
        }
    }
}

float MaxRowSum(float *X, int dim)
{
    float(*tmp_X)[dim] = (float(*)[dim])X;
    float max = -__FLT_MAX__;
    for (int i = 0; i < v_num; i++)
    {
        float sum = 0;
        for (int j = 0; j < dim; j++)
        {
            sum += tmp_X[i][j];
        }
        if (sum > max)
            max = sum;
    }
    return max;
}

void freeFloats()
{
    free(X0);
    free(W1);
    free(W2);
    free(X1);
    free(X2);
    free(X1_inter);
    free(X2_inter);
}

void somePreprocessing()
{
    Raw_Graph_To_CSR();
}

int main(int argc, char **argv)
{
    F0 = atoi(argv[1]);
    F1 = atoi(argv[2]);
    F2 = atoi(argv[3]);

    readGraph(argv[4]);
    readFloat(argv[5], X0, v_num * F0);
    readFloat(argv[6], W1, F0 * F1);
    readFloat(argv[7], W2, F1 * F2);

    initFloat(X1, v_num * F1);
    initFloat(X1_inter, v_num * F1);
    initFloat(X2, v_num * F2);
    initFloat(X2_inter, v_num * F2);

    TimePoint start = chrono::steady_clock::now();

    somePreprocessing();

    XW(F0, F1, X0, X1_inter, W1);

    AX(F1, X1_inter, X1);

    ReLU(F1, X1);

    XW(F1, F2, X1, X2_inter, W2);

    AX(F2, X2_inter, X2);

    LogSoftmax(F2, X2);

    float max_sum = MaxRowSum(X2, F2);

    TimePoint end = chrono::steady_clock::now();
    chrono::duration<double> l_durationSec = end - start;
    double l_timeMs = l_durationSec.count() * 1e3;

    printf("%.8f\n", max_sum);
    printf("%.8lf\n", l_timeMs);

    freeFloats();
}
