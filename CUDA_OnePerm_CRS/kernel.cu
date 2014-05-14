#include "stdio.h"
#include "kernel.h"
#include "curand_kernel.h"
#define THREADS_PER_BLOCK 256
#define NUM_HASH_FUNCTIONS 500
#define SIMILARITY_THRESHOLD 0.6
//#define SM_SIZE 49152

//TODO
//-Coalesced Access
//-Remove branching
//-Use shared memory
//-Matrix tilling?

//Each thread initializes the vales for the set it is responsible to
__global__ void
buildSignatureMatrix_kernel(int* d_signatureMatrix, int numHashFunctions, int smSize)
{
  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int offSetSM;
  for (int i = 0; i < numHashFunctions; i++) {
    offSetSM = i * tid;
    if (offSetSM < smSize) {
      d_signatureMatrix[offSetSM] = INT_MAX;
    }
  }
  /* 
  //Not coalesced
  int offsetSM = tid*numHashFunctions;
  for (int i = offsetSM; i < offsetSM+numHashFunctions; i++) {
    if (i < smSize) {
      d_signatureMatrix[i] = INT_MAX;
    }
  }
  */
}

/*
//Each thread calculates the minhash for the set it is responsible to
__global__ void
updateSignatureMatrix_kernel(int *d_signatureMatrix, int *d_cmColIdx, int* d_cmRowPtr, int lwSize, int smSize, int numSets, int numHashFunctions, unsigned long seed)
{
  int i, j, k;
  int smRow[NUM_HASH_FUNCTIONS];

  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

  //Generate hashes for one unique word
  curandState state;
  curand_init(seed, tid, 0, &state);
  int hashesForWord[NUM_HASH_FUNCTIONS] = {0};
  for (i = 0; i < numHashFunctions; i++) {
    hashesForWord[i] = curand(&state);
  }
  
  //Compares the current values of in the signature matrix with the new hashes
  if (tid < lwSize - 1) {
    int offsetCM = d_cmRowPtr[tid];
    for (i = offsetCM; i < d_cmRowPtr[tid+1]; i++) {
      int setIdx = d_cmColIdx[i];
      int offsetSM = setIdx * numHashFunctions;
      for (j = 0; j < numHashFunctions; j++) {
	smRow[j] = d_signatureMatrix[offsetSM + j];
      }
      for (k = 0; k < numHashFunctions; k++) {
	if (smRow[k] > hashesForWord[k]) {
	  //Find a better way without using atomics
	  atomicExch(&d_signatureMatrix[offsetSM + k], hashesForWord[k]);
	}
      }
    }
  }
}
*/

//Each thread calculates the minhash for the set it is responsible to
__global__ void
updateSignatureMatrix_kernel(int *d_signatureMatrix, int *d_cmColIdx, int* d_cmRowPtr, int lwSize, int smSize, int numSets, int numHashFunctions, unsigned long seed)
{
  int i, j, k;

  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

  //Generate hashes for one unique word
  curandState state;
  curand_init(seed, tid, 0, &state);
  int hashesForWord[NUM_HASH_FUNCTIONS] = {0};
  for (i = 0; i < numHashFunctions; i++) {
    hashesForWord[i] = curand(&state);
  }
  
  //Compares the current values of in the signature matrix with the new hashes
  if (tid < lwSize - 1) {
    //Load imbalance: threads with many sets will work more
    int offsetCM = d_cmRowPtr[tid];
    for (i = offsetCM; i < d_cmRowPtr[tid+1]; i++) {
      int setIdx = d_cmColIdx[i];
      int offsetSM = setIdx * numHashFunctions;
      for (j = 0, k = offsetSM; ((j < numHashFunctions) && (k < offsetSM+numHashFunctions)); j++, k++) {
	if (k < smSize) {
	  if (d_signatureMatrix[k] > hashesForWord[j]) { //Incorrect: other threads can update the value after it is read
	    //Find a better way without using atomics
	    atomicExch(&d_signatureMatrix[k], hashesForWord[j]);
	  }
	}
      }
    }
  }
}

/*
__global__ void
nestedLoopJoin_kernel(int* d_signatureMatrix, int rSize, int sSize, int numHashFunctions, float threshold, int *d_similarPairsCount)
{
  int m,i,j,k,identicalMinhashes;
  float similarity;
  //Bring the hash values for this thread's set to the registers
  int smRow[NUM_HASH_FUNCTIONS];

  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < rSize) { //Find a better way to control out of bounds access
    int offsetSM = tid * numHashFunctions;
    for (m = 0; m < numHashFunctions; m++) {
      smRow[m] = d_signatureMatrix[offsetSM + m];
    }
    for (i = (rSize*numHashFunctions); i < ((rSize+sSize)*numHashFunctions); i=i+numHashFunctions) {
      identicalMinhashes = 0;
      for (j = 0, k = i; (j < numHashFunctions) && (k < (i+numHashFunctions)); j++, k++) {
	if (smRow[j] == d_signatureMatrix[k]) {
	  identicalMinhashes++;
	}
      }
      similarity = (identicalMinhashes*1.0)/(numHashFunctions*1.0);
      if (similarity >= threshold) {
	//Find a better way without using atomics - Maybe change to some kind of reduction
	atomicAdd(d_similarPairsCount,1);
	//How to retrieve the pairs? Write atomically in an array? What should be the size of the array?
      	printf("The similarity between the %dth record and the %dth record is %1.2f.\n", tid, (k/numHashFunctions)-rSize, similarity);
      }
    }
  }
}
*/

__global__ void
nestedLoopJoin_kernel(int* d_signatureMatrix, int rSize, int sSize, int numHashFunctions, float threshold, int *d_similarPairsCount)
{
  int m,n,i,j,identicalMinhashes;
  float similarity;
  //Variables to make use of registers for faster memory access
  int smFixedRow[NUM_HASH_FUNCTIONS];
  int smVariableRow[NUM_HASH_FUNCTIONS];

  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < rSize) { //Find better way to control out of bounds access
    int offsetSM = tid * numHashFunctions;
    //Bring the hash values for this thread's set to the registers 
    for (m = 0; m < numHashFunctions; m++) {
      smFixedRow[m] = d_signatureMatrix[offsetSM + m];
    }
    for (i = (rSize*numHashFunctions); i < ((rSize+sSize)*numHashFunctions); i=i+numHashFunctions) {
      identicalMinhashes = 0;
      //Brings the hashes of the set to be compared to the registers
      for (n = 0; n < numHashFunctions; n++) {
        smVariableRow[n] = d_signatureMatrix[i + n];
      }
      for (j = 0; j < numHashFunctions; j++) {
        if (smFixedRow[j] == smVariableRow[j]) {
          identicalMinhashes++;
        }
      }
      similarity = (identicalMinhashes*1.0)/(numHashFunctions*1.0);
      if (similarity >= threshold) {
        //Not very efficient - Maybe change to some kind of reduction
        //How to retrieve the pairs? Write atomically in an array? What should be the size of the array?        
	atomicAdd(d_similarPairsCount,1);
	//        printf("The similarity between the %dth record and the %dth record is %1.2f.\n", tid, ((i+j)/numHashFunctions)-rSize, similarity);
      }
    }
  }
}

/*
__global__ void
nestedBlockLoopJoin_kernel(int* d_signatureMatrix, int rSize, int sSize, int numHashFunctions, float threshold, int *d_similarPairsCount)
{
  int i,j,k,identicalMinhashes;
  float similarity;
  __shared__ int relationRBlock[SM_SIZE/2];
  __shared__ int relationSBlock[SM_SIZE/2];

  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < rSize) {
    relationRBlock[tid] = d_signatureMatrix[tid];

    int offsetSM = tid * numHashFunctions;
    for (i = (rSize*numHashFunctions); i < ((rSize+sSize)*numHashFunctions); i=i+numHashFunctions) {
      identicalMinhashes = 0;
      
      for (j = offsetSM, k = i; (j < (offsetSM+numHashFunctions)) && (k < (i+numHashFunctions)); j++, k++) {
	if (d_signatureMatrix[j] == d_signatureMatrix[k]) {
	  identicalMinhashes++;
	}
      }
      similarity = (identicalMinhashes*1.0)/(numHashFunctions*1.0);
      if (similarity >= threshold) {
	//Not very efficient - Maybe change to some kind of reduction
	//How to retrieve the pairs? Write atomically in an array? What should be the size of the array?
	atomicAdd(d_similarPairsCount,1);
      	printf("The similarity between the %dth record and the %dth record is %1.2f.\n", tid, (k/numHashFunctions)-rSize, similarity);
      }
    }
  }
}
*/

void
kernelManager(vector<int> &h_signatureMatrix, crsMatrix* h_characteristicMatrix, int lwSize, int numHashFunctions, int sSize, int rSize, vector<string> relationRSetsID, vector<string> relationSSetsID)
{
  int numberOfThreads = THREADS_PER_BLOCK;
  int numberOfBlocks;
  int numSets = rSize + sSize;
  float threshold = SIMILARITY_THRESHOLD;
  int h_similarPairsCount = 0;

  //Device variables
  int *d_signatureMatrix, *d_cmColIdx, *d_cmRowPtr, *d_similarPairsCount;

  //Size of data structures
  int cmColIdxSize = h_characteristicMatrix -> col_ind.size();
  int cmRowPtrSize = h_characteristicMatrix -> row_ptr.size();
  int smSize = h_signatureMatrix.size();

  //CRS representation of the characteristic matrix
  vector<int> h_cmColIdx = h_characteristicMatrix -> col_ind;
  vector<int> h_cmRowPtr = h_characteristicMatrix -> row_ptr;

  //Memory allocation on GPU
  cudaMalloc(&d_signatureMatrix, sizeof(int) * smSize);
  cudaMalloc(&d_cmColIdx, sizeof(int) * cmColIdxSize);
  cudaMalloc(&d_cmRowPtr, sizeof(int) * cmRowPtrSize);
  cudaMalloc(&d_similarPairsCount, sizeof(int));
  
  //Memory transfer CPU -> GPU
  cudaMemcpy(d_cmColIdx, &h_cmColIdx[0], sizeof(int) * cmColIdxSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cmRowPtr, &h_cmRowPtr[0], sizeof(int) * cmRowPtrSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_similarPairsCount, &h_similarPairsCount, sizeof(int), cudaMemcpyHostToDevice);

  //Build Signature Matrix
  numberOfBlocks = numSets / THREADS_PER_BLOCK;
  if (numSets % THREADS_PER_BLOCK) numberOfBlocks++;
  buildSignatureMatrix_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrix, numHashFunctions, smSize);

  //Update Signature Matrix
  numberOfBlocks = lwSize / THREADS_PER_BLOCK;
  if (lwSize % THREADS_PER_BLOCK) numberOfBlocks++;
  updateSignatureMatrix_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrix, d_cmColIdx, d_cmRowPtr, lwSize, smSize, numSets, numHashFunctions, time(NULL));

  //Nested Loop Join
  /*
  numberOfBlocks = rSize / THREADS_PER_BLOCK;
  if (rSize % THREADS_PER_BLOCK) numberOfBlocks++;
  nestedLoopJoin_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrix, rSize, sSize, numHashFunctions, threshold, d_similarPairsCount);
  */

  //Nested Block Loop Join
  /*
  numberOfBlocks = rSize / THREADS_PER_BLOCK;
  if (rSize % THREADS_PER_BLOCK) numberOfBlocks++;
  nestedBlockLoopJoin_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrix, rSize, sSize, numHashFunctions, threshold, d_similarPairsCount);
  */

  //Memory transfer GPU -> CPU
  //  cudaMemcpy(&h_similarPairsCount, d_similarPairsCount, sizeof(int), cudaMemcpyDeviceToHost);

  //printf("Number of similar pairs: %d\n", h_similarPairsCount);

  //Free GPU allocated memory
  cudaFree(d_signatureMatrix);
  cudaFree(d_cmColIdx);
  cudaFree(d_cmRowPtr);
  cudaFree(d_similarPairsCount);
}