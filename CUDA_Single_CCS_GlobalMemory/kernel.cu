#include "stdio.h"
#include "kernel.h"
#include "curand_kernel.h"

#define THREADS_PER_BLOCK 256
#define SIMILARITY_THRESHOLD 0.6

//Possible optimizations:
//-Coalesced Access
//-Remove branching
//-Use shared memory
//-Matrix tilling
//-Better control of out of bound

__global__ void
buildHashMatrix_kernel(int* d_hashMatrix, int numShingles, unsigned long seed, int primeForHashing)
{
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  curandState state;
  curand_init(seed, tid, 0, &state);
  int a = curand(&state);
  int b = curand(&state);
  if (tid < numShingles) {
    int hashValue = ((((a*tid)+b)%primeForHashing)%numShingles); //Universal Hashing - not efficient because of the modulo operation
    if (hashValue < 0) {
      hashValue = hashValue*(-1); //Find a way to return only positive values from cuRAND
    }
    d_hashMatrix[tid] = hashValue;
  }
}

__global__ void
buildSignatureMatrix_kernel(int* d_signatureMatrix, int smSize)
{
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < smSize) {
    d_signatureMatrix[tid] = INT_MAX;
  }
}

__global__ void
updateSignatureMatrix_kernel(int *d_signatureMatrix, int *d_hashMatrix, int *d_cmRowIdx, int* d_cmColPtr, int numShingles, int smSize, int numSets, int numBins, int binSize)
{
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int binIdx, offSetSM, shingleIdx;
  if (tid < numSets) {
    for (int i = d_cmColPtr[tid]; i < d_cmColPtr[tid+1]; i++ ) {
      shingleIdx = d_cmRowIdx[i];
      binIdx = d_hashMatrix[shingleIdx]/binSize;
      offSetSM = (binIdx*numSets)+tid;
      atomicMin(&d_signatureMatrix[offSetSM], d_hashMatrix[shingleIdx]);
    }
  }
}

__global__ void
nestedLoopJoin_kernel(int* d_signatureMatrix, int rSize, int sSize, float threshold, int *d_similarPairsCount, int numBins)
{
  int identicalMinhashes, emptyBins, numSets = rSize+sSize;
  float similarity;
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < rSize) {
    for (int i = rSize; i < numSets; i++) {
      identicalMinhashes = 0;
      emptyBins = 0;
      for (int j = 0; j < numBins; j++) {
	if (d_signatureMatrix[tid+(j*numSets)] == d_signatureMatrix[i+(j*numSets)]) {
	  if (d_signatureMatrix[tid+(j*numSets)] == INT_MAX) {
	    emptyBins++;
	  } else {
	    identicalMinhashes++;
	  }
	}
      }
      similarity = (identicalMinhashes*1.0)/((numBins*1.0) - (emptyBins*1.0));
      if (similarity > threshold) {
	//	printf("The similarity between the %dth record and the %dth record is %1.2f.\n", threshold, tid, i, similarity);
	atomicAdd(d_similarPairsCount, 1);
      }
    }
  }
}

void
kernelManager(std::vector<int> &h_signatureMatrix, ccsMatrix* h_characteristicMatrix, int numShingles, int primeForHashing, int sSize, int rSize, int numBins, int binSize)
{
  int numberOfThreads = THREADS_PER_BLOCK;
  int numberOfBlocks;
  int numSets = rSize + sSize;
  float threshold = SIMILARITY_THRESHOLD;
  int h_similarPairsCount = 0;

  //Device variables
  int *d_hashMatrix, *d_signatureMatrix, *d_cmRowIdx, *d_cmColPtr;
  int *d_similarPairsCount;

  //Size of data structures
  int cmRowIdxSize = h_characteristicMatrix -> row_ind.size();
  int cmColPtrSize = h_characteristicMatrix -> col_ptr.size();
  int smSize = h_signatureMatrix.size();
  int hmSize = numShingles;

  //CRS representation of the characteristic matrix
  std::vector<int> h_cmRowIdx = h_characteristicMatrix -> row_ind;
  std::vector<int> h_cmColPtr = h_characteristicMatrix -> col_ptr;

  //Memory allocation on GPU
  cudaMalloc(&d_signatureMatrix, sizeof(int) * smSize);
  cudaMalloc(&d_hashMatrix, sizeof(int) * hmSize);
  cudaMalloc(&d_cmRowIdx, sizeof(int) * cmRowIdxSize);
  cudaMalloc(&d_cmColPtr, sizeof(int) * cmColPtrSize);
  cudaMalloc(&d_similarPairsCount, sizeof(int));
  
  //Memory transfer CPU -> GPU
  cudaMemcpy(d_cmRowIdx, &h_cmRowIdx[0], sizeof(int) * cmRowIdxSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cmColPtr, &h_cmColPtr[0], sizeof(int) * cmColPtrSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_similarPairsCount, &h_similarPairsCount, sizeof(int), cudaMemcpyHostToDevice);

  //Build Hash Matrix
  numberOfBlocks = numShingles / THREADS_PER_BLOCK;
  if (numShingles % THREADS_PER_BLOCK) numberOfBlocks++;
  buildHashMatrix_kernel<<<numberOfBlocks, numberOfThreads>>>(d_hashMatrix, numShingles, time(NULL), primeForHashing);
  /*
  std::vector<int> h_hashMatrix(hmSize);
  cudaMemcpy(&h_hashMatrix[0], d_hashMatrix, sizeof(int)*hmSize, cudaMemcpyDeviceToHost);
  for (int i = 0; i < numShingles; i++) {
    std::cout << h_hashMatrix[i] << " ";
  }
  std::cout << "\n";
  */

  //Build Signature Matrix
  numberOfBlocks = smSize / THREADS_PER_BLOCK;
  if (smSize % THREADS_PER_BLOCK) numberOfBlocks++;
  buildSignatureMatrix_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrix, smSize);

  //Update Signature Matrix
  numberOfBlocks = numShingles / THREADS_PER_BLOCK;
  if (numShingles % THREADS_PER_BLOCK) numberOfBlocks++;
  updateSignatureMatrix_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrix, d_hashMatrix, d_cmRowIdx, d_cmColPtr, numShingles, smSize, numSets, numBins, binSize);
  /*
  cudaMemcpy(&h_signatureMatrix[0], d_signatureMatrix, sizeof(int)*smSize, cudaMemcpyDeviceToHost);
  for (int i = 0; i < numBins; i++) {
    for (int j = i*numBins; j < (i*numBins)+numSets; j++) {
      std::cout << h_signatureMatrix[j] << " ";      
    }
    std::cout << "\n";
  }
  */

  //Nested Loop Join
  numberOfBlocks = rSize / THREADS_PER_BLOCK;
  if (rSize % THREADS_PER_BLOCK) numberOfBlocks++;
  //  nestedLoopJoin_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrix, rSize, sSize, threshold, d_similarPairsCount, numBins);

  //Memory transfer GPU -> CPU
  //  cudaMemcpy(&h_similarPairsCount, d_similarPairsCount, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_signatureMatrix[0], d_signatureMatrix, sizeof(int)*smSize, cudaMemcpyDeviceToHost);

  std::cout << "Number of similar pairs: " << h_similarPairsCount << "\n";

  //Free GPU allocated memory
  cudaFree(d_signatureMatrix);
  cudaFree(d_hashMatrix);
  cudaFree(d_cmRowIdx);
  cudaFree(d_cmColPtr);
  cudaFree(d_similarPairsCount);
}