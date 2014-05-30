#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <limits.h>
#include <unordered_map>
#include "characteristicMatrix.h"

#define SIMILARITY_THRESHOLD 0.6
#define NUM_HASH_FUNCTIONS 500

using namespace std;

//Read files containing sentences and insert them to the list of sets
int
processInputRelation(unordered_multimap<int,int>& shingleSetMap, unordered_map<string,int>& shingles, vector<string>& setsIDs, string fileName, int relationOffset)
{
  ifstream relation (fileName);
  string word, value, line = "";
  int shingle,relationSize = 0;
  unordered_map<string,int>::iterator shingleIterator;
  if (relation.is_open()){
    while (getline(relation,line)){
      value = "";
      word = "";
      istringstream tuple(line); 
      getline(tuple,value,'\t');
      //Obtain the ID of the record
      setsIDs.push_back(value); 
      getline(tuple,value,'\t');      
      istringstream sentence(value);
      while(getline(sentence,word,' ')){ 
	//Each new word becomes a key of the unordered map and it has one or more sets associated to it
	shingleIterator = shingles.find(word);
	if (shingleIterator == shingles.end()) {
	  shingle = shingles.size();
	  shingles.emplace(word,shingles.size());
	} else {
	  shingle = shingleIterator -> second;
	}
	pair<int,int> shingleSetPair (relationOffset,shingle);
	shingleSetMap.insert(shingleSetPair);
      }
      relationSize++;
      relationOffset++;
    }    
    relation.close();
  } else {
    cout << "Error opening file.";
  }
  return relationSize;
}


int
generateHash()
{
  int a = (int)rand();
  int b = (int)rand();
  int c = (int)rand();
  int hashValue = ((a * ((a*b*c) >> 4) + b * (a*b*c) + c) & 131071);
  return hashValue;
}

vector<int>
buildHashMatrix(int numUniqueWords, int numHashFunctions)
{
  vector<int> hashMatrix;
  for (int i = 0; i < numHashFunctions; i++) {
    for (int j = 0; j < numUniqueWords; j++) {
      hashMatrix.push_back(generateHash());
    }
  }
  return hashMatrix;
}

vector<int>
buildSignatureMatrix(int numSets, int numHashFunctions)
{
  vector<int> signatureMatrix;
  for (int i = 0; i < numHashFunctions; i++) {
    for (int j = 0; j < numSets; j++) {
      signatureMatrix.push_back(INT_MAX);
    }
  }
  return signatureMatrix;
}

void
computeSignatureMatrix(vector<int> &signatureMatrix, vector<int> &hashMatrix, ccsMatrix* characteristicMatrix, int numSets, int numHashFunctions, int numUniqueWords)
{
  int offSetHM, offSetCM, offSetSM, minhashValue;
  for (int i = 0; i < numHashFunctions; i++) {
    for (int j = 0; j < numSets; j++) {
      offSetSM = j + (i*numSets);
      minhashValue = signatureMatrix[offSetSM];
      offSetCM = characteristicMatrix -> col_ptr[j];
      for (int k = offSetCM; k < characteristicMatrix -> col_ptr[j+1]; k++) {
	offSetHM = characteristicMatrix -> row_ind[k] + (i*numUniqueWords);
	if (minhashValue > hashMatrix[offSetHM]) {
	  minhashValue = hashMatrix[offSetHM];
	}
      }
      signatureMatrix[offSetSM] = minhashValue;      
    }
  }
}

void
computeSimilarities(vector<int> signatureMatrix, int rSize, int sSize, int numHashFunctions, vector<string> relationRSetsID, vector<string> relationSSetsID)
{
  cout << "Computing similarities...\n";
  int identicalMinhashes, similarPairs = 0, offSetR, offSetS, numSets = rSize+sSize;
  vector<int> tempRHashes, tempSHashes;
  float similarity;
  for (int i = 0; i < rSize; i++) {
    tempRHashes.clear();
    for (int j = 0; j < numHashFunctions; j++) {
      tempRHashes.push_back(signatureMatrix[i+(j*numSets)]);
    }
    for (int j = rSize; j < numSets; j++) {
      identicalMinhashes = 0;
      tempSHashes.clear();
      for (int k = 0; k < numHashFunctions; k++) {
	tempSHashes.push_back(signatureMatrix[j+(k*numSets)]);
      }
      for (int k = 0; k < numHashFunctions; k++) {
	//	if (signatureMatrix[i+(k*numSets)] == signatureMatrix[j+(k*numSets)]) {
	//	  identicalMinhashes++;
	//	}
	if (tempRHashes[k] == tempSHashes[k]) {
	  identicalMinhashes++;
	}
      }
      similarity = (identicalMinhashes*1.0)/(numHashFunctions*1.0);
      if (similarity >= SIMILARITY_THRESHOLD) {
	//cout << "The similarity between " << relationRSetsID[i] << " and " << relationSSetsID[j-rSize] << " is " << similarity << "\n";
	similarPairs++;
      }
    }
  }
  cout << "Number of similar pairs: " << similarPairs << "\n";
}

int
main(int argc, char *argv[])
{
  srand(time(NULL));

  if (argc < 3) {
    cout << "Usage: " << argv[0] << "pathToRelation/relationRFile.data pathToRelation/relationSFile.data\n";
    return 1;
  }

  int numHashFunctions = NUM_HASH_FUNCTIONS;
  unordered_multimap<int,int> shingleSetMap;
  unordered_map<string,int> shingles;
  vector<string> relationRSetsID;
  vector<string> relationSSetsID;  

  //Receive input relation files and creates a list of sets and lists of sets' IDs
  int rSize = processInputRelation(shingleSetMap, shingles, relationRSetsID, argv[1] ,0);
  int sSize = processInputRelation(shingleSetMap, shingles, relationSSetsID, argv[2] ,rSize);
  cout << "r Size: " << rSize << "\n";
  cout << "s Size: " << sSize << "\n";
  int numSets = rSize + sSize;

  //Create the CRS representation of the characteristic matrix
  ccsMatrix* characteristicMatrix = buildCharacteristicMatrix(shingleSetMap);
  cout << "characteristicMatrix Size: " << ccsMatrixSize(characteristicMatrix) << "\n";
  //  printCharacteristicMatrixVectors (characteristicMatrix);

  //Calculate the number of unique words
  int numberOfUniqueWords = shingles.size();
  cout << "Number of Unique Words: " << numberOfUniqueWords << "\n";

  //Create the hash matrix
  vector<int> hashMatrix = buildHashMatrix(numberOfUniqueWords, numHashFunctions);
  cout << "hmMatrix Size: " << hashMatrix.size() << "\n";

  vector<int> signatureMatrix = buildSignatureMatrix(rSize+sSize, numHashFunctions);
  cout << "signatureMatrixSize: " << signatureMatrix.size() << "\n";

  computeSignatureMatrix(signatureMatrix, hashMatrix, characteristicMatrix, numSets, numHashFunctions, numberOfUniqueWords);
  
  //computeSimilarities(signatureMatrix, rSize, sSize, numHashFunctions, relationRSetsID, relationSSetsID);

  delete(characteristicMatrix);
  
  return 0;
}

