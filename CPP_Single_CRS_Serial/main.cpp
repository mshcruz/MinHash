#include <chrono>
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

#define NUM_BINS 256

int
processInputRelation(std::unordered_multimap<std::string,int>& shingleSetMap, std::vector<std::string> &setsIDs, std::string fileName, int relationOffset)
{
  std::ifstream relation (fileName);
  std::string shingle, value, line = "";
  int relationSize = 0;
  if (relation.is_open()){
    while (getline(relation,line)){
      value = "";
      shingle = "";
      std::istringstream tuple(line); 
      getline(tuple,value,'\t');
      setsIDs.push_back(value); //Obtain the ID of the record
      getline(tuple,value,'\t');      
      std::istringstream sentence(value);
      while(getline(sentence,shingle,' ')){ //Each new word becomes a key of the unordered map and it has one or more sets associated to it
	std::pair<std::string,int> shingleSetPair (shingle,relationOffset);
	shingleSetMap.insert(shingleSetPair);
      }
      relationSize++;
      relationOffset++;
    }    
    relation.close();
  } else {
    std::cout << "Error opening file.";
  }
  return relationSize;
}

void
buildHashMatrix(std::vector<int> &hashMatrix, int primeForHashing)
{
  std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<int> distribution(0,INT_MAX);
  for (int i = 0; i < hashMatrix.size(); i++) {
    hashMatrix[i] = distribution(generator);
  }
  /*
  std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<int> distribution(1,primeForHashing);
  int a = distribution(generator);
  int b = distribution(generator);
  //  std::cout << "a: " << a << " b: " << b << "\n";
  for (int i = 0; i < hashMatrix.size(); i++) {
    hashMatrix[i] = ((((a*i)+b)%primeForHashing)%hashMatrix.size());
  }
  */
}

void
buildSignatureMatrix(std::vector<int> &signatureMatrix)
{
  for (int i = 0; i < signatureMatrix.size(); i++) {
    signatureMatrix[i] = INT_MAX;
  }
}

void
computeSignatureMatrix(std::vector<int> &signatureMatrix, std::vector<int> &hashMatrix, crsMatrix* characteristicMatrix, int numSets, int numBins, int binSize)
{
  int shingleIdx = 0, binIdx = 0, offSetCM, offSetSM, setIdx;
  
  for (int i = 0; i < hashMatrix.size(); i++) {
    offSetCM = characteristicMatrix -> row_ptr[shingleIdx];
    for (int j = offSetCM; j < characteristicMatrix -> row_ptr[shingleIdx+1]; j++) {
      setIdx = characteristicMatrix -> col_ind[j];
      offSetSM = (setIdx*numBins)+binIdx;
      if (signatureMatrix[offSetSM] > hashMatrix[shingleIdx]) {
	signatureMatrix[offSetSM] = hashMatrix[shingleIdx];
      }
    }
    shingleIdx++;
    if (shingleIdx%binSize == 0) {
      binIdx++;
    }
  }
}

void
computeSimilarities(std::vector<int> signatureMatrix, int rSize, int sSize, int numBins, std::vector<std::string> relationRSetsID, std::vector<std::string> relationSSetsID)
{
  int numSets = rSize+sSize, similarPairs = 0;
  for (int i = 0; i < rSize; i++) {
    for (int j = rSize; j < numSets; j++) {
      int emptyBins = 0;
      int identicalMinhashes = 0;
      for (int k = 0; k < numBins; k++) {
	if (signatureMatrix[k+(i*numBins)] == signatureMatrix[k+(j*numBins)]) {
	  if (signatureMatrix[k+(i*numBins)] == INT_MAX) {
	    emptyBins++;
	  } else {
	    identicalMinhashes++;
	  }
	}
      }
      float similarity = (identicalMinhashes*1.0)/((numBins*1.0)-(emptyBins*1.0));
      if (similarity >= 0.6) {
	std::cout << "The similarity between record " << relationRSetsID[i] << " and record " << relationSSetsID[j-rSize] << " is " << similarity << "\n";
	similarPairs++;
      }     
    }
  }
  std::cout << "Number of similar pairs: " << similarPairs << "\n";
}

int
shinglesNumber(std::unordered_multimap<std::string,int> shingleSetMap)
{
  int count = 0;
  for (auto it = shingleSetMap.begin(); it != shingleSetMap.end(); it = shingleSetMap.equal_range(it->first).second){
    count++;
  }
  return count;
}

int
main(int argc, char *argv[])
{
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << "pathToRelation/relationRFile.data pathToRelation/relationSFile.data\n";
    return 1;
  }
  
  std::unordered_multimap<std::string,int> shingleSetMap;
  std::vector<std::string> relationRSetsID;
  std::vector<std::string> relationSSetsID;
  std::vector<int> primeNumbersForHashing = {7993, 13729, 23399, 39551, 65371, 102437, 164617, 274007, 441937, 7108087};

  int rSize = processInputRelation(shingleSetMap, relationRSetsID, argv[1], 0);
  int sSize = processInputRelation(shingleSetMap, relationSSetsID, argv[2], rSize);
  int numSets = rSize+sSize;
  std::cout << "Number of sets (R + S = Total): " << rSize << " + " << sSize << " = " << numSets << "\n";

  int numShingles = shinglesNumber(shingleSetMap);
  std::cout << "Number of shingles: " << numShingles << "\n";

  int numBins = NUM_BINS;
  int binSize = numShingles/numBins;
  if (numShingles % numBins) binSize++;
  std::cout << "numBins: " << numBins << "\n";
  std::cout << "binSize: " << binSize << "\n";

  int primeForHashing;
  for (int i = 0; i < primeNumbersForHashing.size(); i++) {
    if (numShingles < primeNumbersForHashing[i]) {
      primeForHashing = primeNumbersForHashing[i];
      break;
    }
  }
  std::cout << "Prime used: " << primeForHashing << "\n";  

  crsMatrix *characteristicMatrix = buildCharacteristicMatrix(shingleSetMap);
  std::cout << "characteristicMatrixSize: " << crsMatrixSize(characteristicMatrix) << "\n";
  //  printCharacteristicMatrixVectors(characteristicMatrix);

  std::vector<int> hashMatrix(numShingles);
  buildHashMatrix(hashMatrix, primeForHashing);

  std::vector<int> signatureMatrix(numSets*numBins);
  std::cout << "signatureMatrixSize: " << signatureMatrix.size() << "\n";
  buildSignatureMatrix(signatureMatrix);
  computeSignatureMatrix(signatureMatrix, hashMatrix, characteristicMatrix, numSets, numBins, binSize);
  /*
  for (int i = 0; i<rSize+sSize; i++) {
    for (int j = i*numBins; j < (i*numBins)+numBins; j++) {
      cout << signatureMatrix[j] << " ";
    }
    cout << "\n";
  }
  */
  std::cout << "Computing similarities...\n";
  //computeSimilarities(signatureMatrix, rSize, sSize, numBins, relationRSetsID, relationSSetsID);

  delete(characteristicMatrix);
  
  return 0;
}

