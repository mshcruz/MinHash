#include "stdio.h"
#include "kernel.h"
#include "curand_kernel.h"

#define THREADS_PER_BLOCK 256
#define SIMILARITY_THRESHOLD 0.6
#define SETS_PER_KERNEL 10000

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
    int hashValue = ((((a*tid)+b)%primeForHashing)%numShingles);
    if (hashValue < 0) {
      hashValue = hashValue*(-1);
    }
    d_hashMatrix[tid] = hashValue;
    //    printf("hash for tid=%d: %d\n", tid, d_hashMatrix[tid]);
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
updateSignatureMatrix_kernel(int *d_signatureMatrix, int *d_hashMatrix, int *d_cmColIdx, int* d_cmRowPtr, int numShingles, int smSize, int numSets, int numBins, int binSize)
{
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int binIdx, offSetSM, setIdx;
  if (tid < numShingles) {
    binIdx = d_hashMatrix[tid]/binSize;
    for (int i = d_cmRowPtr[tid]; i < d_cmRowPtr[tid+1]; i++ ) {
      setIdx = d_cmColIdx[i];
      offSetSM = (setIdx*numBins)+binIdx;
      atomicMin(&d_signatureMatrix[offSetSM], d_hashMatrix[tid]);
    }
  }
}

__global__ void
nestedLoopJoin_kernel(int* d_signatureMatrix, int rSize, int sSize, float threshold, int *d_similarPairsCount, int numBins)
{
  int identicalBins, emptyBins, numSets = rSize+sSize;
  float similarity;
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < rSize) {
    for (int i = rSize; i < numSets; i++) {
      identicalBins = 0;
      emptyBins = 0;
      for (int j = 0; j < numBins; j++) {
	if (d_signatureMatrix[(tid*numBins)+j] == d_signatureMatrix[(i*numBins)+j]) {
	  if (d_signatureMatrix[(tid*numBins)+j] == INT_MAX) {
	    emptyBins++;
	  } else {
	    identicalBins++;
	  }
	}
      }
      similarity = (identicalBins*1.0)/((numBins*1.0) - (emptyBins*1.0));
      if (similarity > threshold) {
	//	printf("The similarity between the %dth record and the %dth record is %1.2f.\n", threshold, tid, i, similarity);
	atomicAdd(d_similarPairsCount, 1);
      }
    }
  }
}

void
kernelManager(std::vector<int> &h_signatureMatrix, crsMatrix* h_characteristicMatrix, int numShingles, int primeForHashing, int sSize, int rSize, int numBins, int binSize)
{
  int numberOfThreads = THREADS_PER_BLOCK;
  int numberOfBlocks;
  int numSets = rSize + sSize;
  float threshold = SIMILARITY_THRESHOLD;
  int h_similarPairsCount = 0;

  //Device variables
  int *d_hashMatrix, *d_signatureMatrix, *d_cmColIdx, *d_cmRowPtr;
  int *d_similarPairsCount;

  //Size of data structures
  int cmColIdxSize = h_characteristicMatrix -> col_ind.size();
  int cmRowPtrSize = h_characteristicMatrix -> row_ptr.size();
  int smSize = h_signatureMatrix.size();
  int hmSize = numShingles;

  //CRS representation of the characteristic matrix
  std::vector<int> h_cmColIdx = h_characteristicMatrix -> col_ind;
  std::vector<int> h_cmRowPtr = h_characteristicMatrix -> row_ptr;

  //Memory allocation on GPU
  cudaMalloc(&d_signatureMatrix, sizeof(int) * SETS_PER_KERNEL * numBins);
  cudaMalloc(&d_hashMatrix, sizeof(int) * hmSize);
  cudaMalloc(&d_cmColIdx, sizeof(int) * cmColIdxSize);
  cudaMalloc(&d_cmRowPtr, sizeof(int) * cmRowPtrSize);
  cudaMalloc(&d_similarPairsCount, sizeof(int));
  
  //Memory transfer CPU -> GPU
  cudaMemcpy(d_cmColIdx, &h_cmColIdx[0], sizeof(int) * cmColIdxSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cmRowPtr, &h_cmRowPtr[0], sizeof(int) * cmRowPtrSize, cudaMemcpyHostToDevice);
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
  updateSignatureMatrix_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrix, d_hashMatrix, d_cmColIdx, d_cmRowPtr, numShingles, smSize, numSets, numBins, binSize);
  
  //Print Signature Matrix
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
  //  cudaMemcpy(&h_signatureMatrix[0], d_signatureMatrix, sizeof(int)*smSize, cudaMemcpyDeviceToHost);

  std::cout << "Number of similar pairs: " << h_similarPairsCount << "\n";

  //Free GPU allocated memory
  cudaFree(d_signatureMatrix);
  cudaFree(d_hashMatrix);
  cudaFree(d_cmColIdx);
  cudaFree(d_cmRowPtr);
  cudaFree(d_similarPairsCount);
}