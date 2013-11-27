#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>

#define NUM_HASH_FUNCTIONS 10
#define NUM_SETS 2

typedef char* string;

int calculateHash(int *hash){
  srand(time(NULL));
  
  for (int i = 0; i < NUM_HASH_FUNCTIONS; i++){
    int a = (int)rand();
    int b = (int)rand();
    int c = (int)rand();

    hash[i] = (int)((a * ((a*b*c) >> 4) + b * (a*b*c) + c) & 131071);
  }
  
  return 0;
}

int initializeHashBuckets(int *minHashValues){
  for(int i=0; i < NUM_SETS; i++){
    for(int j=0; j < NUM_HASH_FUNCTIONS; j++){
      minHashValues[i][j] = INT_MAX;
    }
  }
}

int main(int argc, char *argv[]){
  int hash[NUM_HASH_FUNCTIONS] = {0};
  int minHashValues[NUM_SETS][NUM_HASH_FUNCTIONS] = {{0},{0}};

  string set1[4];
  set1[0] = "bola";
  set1[1] = "faca";
  set1[2] = "gota";
  set1[3] = "kadu";

  string set2[5];
  set2[0] = "aviao";
  set2[1] = "bola";
  set2[2] = "caule";
  set2[3] = "gota";
  set2[4] = "joao";

  buildBitMap();

  calculateHash(hash);
  
  calculateSimilarity();

  for (int i = 0; i < NUM_HASH_FUNCTIONS; i++){
    if(hash[i] < minhash){
      minhash = hash[i];
    }
  }
  printf("Minhash value: %d\n\n", minhash);    
  return 0;
}

