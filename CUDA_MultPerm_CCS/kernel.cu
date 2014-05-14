#include "stdio.h"
#include "kernel.h"
#include "curand_kernel.h"
#define THREADS_PER_BLOCK 256
#define NUM_HASH_FUNCTIONS 500
#define SIMILARITY_THRESHOLD 0.6

//Possible optimizations:
//-Build the hash matrix in the GPU
//-Coalesced Access
//-Remove branching
//-Use shared memory
//-Matrix tilling
//-Better control of out of bound

//Each thread calculates the hash values for the word it is responsible to
__global__ void
buildHashMatrix_kernel(int* d_hashMatrix, int numHashFunctions, int lwSize, unsigned long seed)
{
  int offSetHM;
  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  curandState state;
  curand_init(seed, tid, 0, &state);

  if (tid < lwSize) {
    for (int i = 0; i < numHashFunctions; i++) {
      offSetHM = tid + (i*lwSize);
      d_hashMatrix[offSetHM] = curand(&state);
    }
  }
  /*
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
*/
}


//Each thread initializes the vales for the set it is responsible to
__global__ void
buildSignatureMatrix_kernel(int* d_signatureMatrix, int numHashFunctions, int smSize, int numSets)
{
  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int offSetSM;
  for (int i = 0; i < numHashFunctions; i++) {
    offSetSM = tid + (i * numSets); 
    if (offSetSM < smSize) {
      d_signatureMatrix[offSetSM] = INT_MAX;
    }
  }
}

//Each thread calculates the minhash for the set it is responsible to
__global__ void
updateSignatureMatrix_kernel(int *d_signatureMatrix, int *d_hashMatrix, int *d_cmRowIdx, int* d_cmColPtr, int lwSize, int smSize, int numSets, int numHashFunctions)
{
  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < numSets) { //OOB Control
    int offSetSM, offSetCM, offSetHM;
    for (int i = 0; i < numHashFunctions; i++) {
      offSetSM = tid + (i*numSets); //Coalesced
      offSetCM = d_cmColPtr[tid]; //Coalesced
      //Find out what words belong to the set
      for (int j = offSetCM; j < d_cmColPtr[tid+1]; j++) {
	offSetHM = d_cmRowIdx[j] + (i*lwSize); //Not coalesced
	//Update matrix if there is a smaller hash value
	if (d_signatureMatrix[offSetSM] > d_hashMatrix[offSetHM]) { //Branching
	  //	  printf("New value: %d - Old value: %d\n", d_hashMatrix[offSetHM], d_signatureMatrix[offSetSM]);
	  d_signatureMatrix[offSetSM] = d_hashMatrix[offSetHM];
	}
      }
    }
  }
}

__global__ void
nestedLoopJoin_kernel(int* d_signatureMatrix, int rSize, int sSize, int numHashFunctions, float threshold, int *d_similarPairsCount)
{
  float similarity;
  int identicalMinhashes, offSetSM, numSets = rSize+sSize;

  //Variables to make use of registers for faster memory access
  int smFixedRow[NUM_HASH_FUNCTIONS];
  int smVariableRow[NUM_HASH_FUNCTIONS];

  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < rSize) { //OOB Control
    //Bring the minhash values for this thread's set to the registers 
    for (int i = 0; i < numHashFunctions; i++) {
      offSetSM = tid + (i*numSets); //Serial/Coalesced
      smFixedRow[i] = d_signatureMatrix[offSetSM];
    }

    for (int j = rSize; j < numSets; j++) {
      identicalMinhashes = 0;
      //Bring the minhashes of the set to be compared to the registers
      for (int k = 0; k < numHashFunctions; k++) {
	offSetSM = j + (k*numSets); //Serial/Not Coalesced
        smVariableRow[k] = d_signatureMatrix[offSetSM];
      }

      //Calculate the number of identical minhashes between two sets
      for (int l = 0; l < numHashFunctions; l++) {
        if (smFixedRow[l] == smVariableRow[l]) { //Serial/Not coalesced
          identicalMinhashes++;
        }
      }

      similarity = (identicalMinhashes*1.0)/(numHashFunctions*1.0);
      //      printf("The similarity between the %dth record and the %dth record is %1.2f.\n", tid, j, similarity);
      //printf("Set %d and set %d have similarity %1.2f.\n", tid, j, similarity);
      if (similarity >= threshold) {
        //Not very efficient - Maybe change to some kind of reduction
        //How to retrieve the pairs? Write atomically in an array? What should be the size of the array?        
	//printf("Set %d and set %d have similarity %1.2f.\n", tid, j-rSize, similarity);
	atomicAdd(d_similarPairsCount,1);
      }
    }
  }
}

void
kernelManager(vector<int> &h_signatureMatrix, ccsMatrix* h_characteristicMatrix, int lwSize, int numHashFunctions, int sSize, int rSize, vector<string> relationRSetsID, vector<string> relationSSetsID)
{
  int numberOfThreads = THREADS_PER_BLOCK;
  int numberOfBlocks;
  int numSets = rSize + sSize;
  float threshold = SIMILARITY_THRESHOLD;
  int h_similarPairsCount = 0;

  //Device variables
  int *d_hashMatrix, *d_signatureMatrix, *d_cmRowIdx, *d_cmColPtr, *d_similarPairsCount;

  //Size of data structures
  int cmRowIdxSize = h_characteristicMatrix -> row_ind.size();
  int cmColPtrSize = h_characteristicMatrix -> col_ptr.size();
  int smSize = h_signatureMatrix.size();
  int hmSize = lwSize*numHashFunctions;

  //CRS representation of the characteristic matrix
  vector<int> h_cmRowIdx = h_characteristicMatrix -> row_ind;
  vector<int> h_cmColPtr = h_characteristicMatrix -> col_ptr;

  //Memory allocation on GPU
  cudaMalloc(&d_signatureMatrix, sizeof(int) * smSize);
  cudaMalloc(&d_hashMatrix, sizeof(int) * hmSize);
  cudaMalloc(&d_cmRowIdx, sizeof(int) * cmRowIdxSize);
  cudaMalloc(&d_cmColPtr, sizeof(int) * cmColPtrSize);
  cudaMalloc(&d_similarPairsCount, sizeof(int));
  
  //Memory transfer CPU -> GPU
  cudaMemcpy(d_cmRowIdx, &h_cmRowIdx[0], sizeof(int) * cmRowIdxSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cmColPtr, &h_cmColPtr[0], sizeof(int) * cmColPtrSize, cudaMemcpyHostToDevice);
  //  cudaMemcpy(d_hashMatrix, &h_hashMatrix[0], sizeof(int) * hmSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_similarPairsCount, &h_similarPairsCount, sizeof(int), cudaMemcpyHostToDevice);

  //Build Hash Matrix
  numberOfBlocks = lwSize / THREADS_PER_BLOCK;
  if (lwSize % THREADS_PER_BLOCK) numberOfBlocks++;
  buildHashMatrix_kernel<<<numberOfBlocks, numberOfThreads>>>(d_hashMatrix, numHashFunctions, lwSize, time(NULL));

  //Print Hash Matrix
  /*
  vector<int> h_hashMatrix(hmSize);
  cudaMemcpy(&h_hashMatrix[0], d_hashMatrix, sizeof(int)*hmSize, cudaMemcpyDeviceToHost);
  for (int i = 0; i < numHashFunctions; i++) {
    for (int j = i*numHashFunctions; j < (i*numHashFunctions)+lwSize; j++) {
      cout << h_hashMatrix[j] << " ";      
    }
    cout << "\n";
  }
  */

  //Build Signature Matrix
  numberOfBlocks = numSets / THREADS_PER_BLOCK;
  if (numSets % THREADS_PER_BLOCK) numberOfBlocks++;
  buildSignatureMatrix_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrix, numHashFunctions, smSize, numSets);

  //Print Signature Matrix
  /*
  cudaMemcpy(&h_signatureMatrix[0], d_signatureMatrix, sizeof(int)*smSize, cudaMemcpyDeviceToHost);
  for (int i = 0; i < numHashFunctions; i++) {
    for (int j = i*numHashFunctions; j < (i*numHashFunctions)+numSets; j++) {
      cout << h_signatureMatrix[j] << " ";      
    }
    cout << "\n";
  }
  */



  //Update Signature Matrix
  updateSignatureMatrix_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrix, d_hashMatrix, d_cmRowIdx, d_cmColPtr, lwSize, smSize, numSets, numHashFunctions);

  //Print Signature Matrix
  /*
  cudaMemcpy(&h_signatureMatrix[0], d_signatureMatrix, sizeof(int)*smSize, cudaMemcpyDeviceToHost);
  for (int i = 0; i < numHashFunctions; i++) {
    for (int j = i*numHashFunctions; j < (i*numHashFunctions)+numSets; j++) {
      cout << h_signatureMatrix[j] << " ";      
    }
    cout << "\n";
  }
  */

  //Nested Loop Join
  numberOfBlocks = rSize / THREADS_PER_BLOCK;
  if (rSize % THREADS_PER_BLOCK) numberOfBlocks++;
  nestedLoopJoin_kernel<<<numberOfBlocks, numberOfThreads>>>(d_signatureMatrix, rSize, sSize, numHashFunctions, threshold, d_similarPairsCount);

  //Memory transfer GPU -> CPU
  cudaMemcpy(&h_similarPairsCount, d_similarPairsCount, sizeof(int), cudaMemcpyDeviceToHost);
  //  cudaMemcpy(&h_signatureMatrix[0], d_signatureMatrix, sizeof(int)*smSize, cudaMemcpyDeviceToHost);

  printf("Number of similar pairs: %d\n", h_similarPairsCount);

  //Free GPU allocated memory
  cudaFree(d_signatureMatrix);
  cudaFree(d_hashMatrix);
  cudaFree(d_cmRowIdx);
  cudaFree(d_cmColPtr);
  cudaFree(d_similarPairsCount);
}