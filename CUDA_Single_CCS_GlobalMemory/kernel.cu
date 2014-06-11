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
    int hashValue = ((((a*tid)+b)%primeForHashing)%numShingles); //Universal Hashing - not efficient because of the modulo operation
    if (hashValue < 0) {
      hashValue = hashValue*(-1); //Find a way to return only positive values from cuRAND
    }
    d_hashMatrix[tid] = hashValue;
  }
}

__global__ void
buildSignatureMatrixChunk_kernel(int* d_signatureMatrixChunk, int smChunkSize)
{
  //  printf("building sm...\n");
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < smChunkSize) {
    d_signatureMatrixChunk[tid] = INT_MAX;
  }
}

__global__ void
updateSignatureMatrix_kernel(int *d_signatureMatrixChunk, int *d_hashMatrix, int *d_cmRowIdx, int* d_cmColPtr, int numShingles, int smChunkSize, int numSetsChunk, int numBins, int binSize, int offSetCM)
{
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int binIdx, offSetSMChunk, shingleIdx;
  if (tid < numSetsChunk) {
    for (int i = d_cmColPtr[offSetCM+tid]; i < d_cmColPtr[offSetCM+tid+1]; i++ ) {
      shingleIdx = d_cmRowIdx[i];
      binIdx = d_hashMatrix[shingleIdx]/binSize;
      //      offSetSM = (binIdx*numSets)+tid;
      offSetSMChunk = binIdx + (tid*numBins);
      

      atomicMin(&d_signatureMatrixChunk[offSetSMChunk], d_hashMatrix[shingleIdx]);
      //  printf("tid: %d, shingleIdx: %d, hashedShingle: %d, binIdx: %d, smValue: %d\n", tid, shingleIdx, d_hashMatrix[shingleIdx], binIdx, d_signatureMatrixChunk[offSetSM]);      
//printf("smChunk new value: %d", &d_signatureMatrixChunk[offSetSM]);
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
	// if (d_signatureMatrix[tid+(j*numSets)] == d_signatureMatrix[i+(j*numSets)]) {
	//   if (d_signatureMatrix[tid+(j*numSets)] == INT_MAX) {
	//     emptyBins++;
	//   } else {
	//     identicalMinhashes++;
	//   }
	// }
	if (d_signatureMatrix[(tid*numBins)+j] == d_signatureMatrix[(i*numBins)+j]) {
	  if (d_signatureMatrix[(tid*numBins)+j] == INT_MAX) {
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
  int numberOfThreads = THREADS_PER_BLOCK, numberOfBlocks, numSets = rSize + sSize, h_similarPairsCount = 0;
  float threshold = SIMILARITY_THRESHOLD;

  //Device variables
  int *d_hashMatrix, *d_signatureMatrixChunk, *d_cmRowIdx, *d_cmColPtr, *d_similarPairsCount;

  //Size of data structures
  int cmRowIdxSize = h_characteristicMatrix -> row_ind.size();
  int cmColPtrSize = h_characteristicMatrix -> col_ptr.size();
  int smSize = h_signatureMatrix.size();

  int hmSize = numShingles;

  //Characteristic matrix
  std::vector<int> h_cmRowIdx = h_characteristicMatrix -> row_ind;
  std::vector<int> h_cmColPtr = h_characteristicMatrix -> col_ptr;

  //Memory allocation on GPU
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

  //Calculate parts of the signature matrix (each kernel call computes the bin values for SETS_PER_KERNEL number of sets)
  int smChunkSize = SETS_PER_KERNEL*numBins;
  cudaMalloc(&d_signatureMatrixChunk, sizeof(int) * smChunkSize);
  int numKernelCalls = numSets / SETS_PER_KERNEL;
  for (int i = 0; i < numKernelCalls; i++) {
    numberOfBlocks = smChunkSize / THREADS_PER_BLOCK;
    if (smChunkSize % THREADS_PER_BLOCK) numberOfBlocks++;
    buildSignatureMatrixChunk_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrixChunk, smChunkSize);

    numberOfBlocks = SETS_PER_KERNEL / THREADS_PER_BLOCK;
    if (SETS_PER_KERNEL % THREADS_PER_BLOCK) numberOfBlocks++;
    int offSetCM = i*SETS_PER_KERNEL;
    updateSignatureMatrix_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrixChunk, d_hashMatrix, d_cmRowIdx, d_cmColPtr, numShingles, smChunkSize, SETS_PER_KERNEL, numBins, binSize, offSetCM);
    cudaMemcpy(&h_signatureMatrix[i*smChunkSize], d_signatureMatrixChunk, sizeof(int)*smChunkSize, cudaMemcpyDeviceToHost);
  }
  cudaFree(d_signatureMatrixChunk);

  //Calculate the bin values for the rest of the sets (a number smaller than SETS_PER_KERNEL)
  int lastSetsSize = numSets%SETS_PER_KERNEL;  
  if (lastSetsSize) {
    smChunkSize = lastSetsSize * numBins;
    cudaMalloc(&d_signatureMatrixChunk, sizeof(int) * smChunkSize);

    numberOfBlocks = smChunkSize / THREADS_PER_BLOCK;
    if (smChunkSize % THREADS_PER_BLOCK) numberOfBlocks++;
    buildSignatureMatrixChunk_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrixChunk, smChunkSize);

    numberOfBlocks = lastSetsSize / THREADS_PER_BLOCK;
    if (lastSetsSize % THREADS_PER_BLOCK) numberOfBlocks++;
    int offSetCM = numKernelCalls*SETS_PER_KERNEL;
    updateSignatureMatrix_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrixChunk, d_hashMatrix, d_cmRowIdx, d_cmColPtr, numShingles, smChunkSize, lastSetsSize, numBins, binSize, offSetCM);

    cudaMemcpy(&h_signatureMatrix[numKernelCalls*SETS_PER_KERNEL*numBins], d_signatureMatrixChunk, sizeof(int)*smChunkSize, cudaMemcpyDeviceToHost);
    cudaFree(d_signatureMatrixChunk);
  }
  /*
  printf("printing...\n");
  for (int m = 0; m < h_signatureMatrix.size(); m++) {
    printf("%d ", h_signatureMatrix[m]);
  }
  */

  //Nested Loop Join
  /*
  int* d_signatureMatrix; 
  cudaMalloc(&d_signatureMatrix, sizeof(int) * smSize);
  cudaMemcpy(d_signatureMatrix, &h_signatureMatrix[0], sizeof(int) * smSize, cudaMemcpyHostToDevice);
  numberOfBlocks = rSize / THREADS_PER_BLOCK;
  if (rSize % THREADS_PER_BLOCK) numberOfBlocks++;
  nestedLoopJoin_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrix, rSize, sSize, threshold, d_similarPairsCount, numBins);
  */

  //Memory transfer GPU -> CPU
  //  cudaMemcpy(&h_similarPairsCount, d_similarPairsCount, sizeof(int), cudaMemcpyDeviceToHost);

  //  std::cout << "Number of similar pairs: " << h_similarPairsCount << "\n";

  //Free GPU allocated memory
  cudaFree(d_hashMatrix);
  cudaFree(d_cmRowIdx);
  cudaFree(d_cmColPtr);
  cudaFree(d_similarPairsCount);
}