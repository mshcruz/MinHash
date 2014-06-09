#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <climits>
#include <string>
#include <map>
#include <random>
#include "characteristicMatrix.h"

#define SIMILARITY_THRESHOLD 0.6
#define NUM_BINS 256

int
processInputRelation(std::multimap<int,int>& shingleSetMap, std::map<std::string,int>& shingles, std::vector<std::string>& setsIDs, std::string fileName, int relationOffset)
{
  std::ifstream relation (fileName);
  std::string word, value, line = "";
  int shingle, relationSize = 0;
  std::map<std::string,int>::iterator shingleIterator;
  if (relation.is_open()){
    while (getline(relation,line)){
      value = "";
      word = "";
      std::istringstream tuple(line); 
      getline(tuple,value,'\t');
      //Obtain the ID of the record
      setsIDs.push_back(value); 
      getline(tuple,value,'\t');      
      std::istringstream sentence(value);
      while(getline(sentence,word,' ')){ 
	//Each new word becomes a key of the ordered map and it has one or more sets associated to it
	shingleIterator = shingles.find(word);
	if (shingleIterator == shingles.end()) {
	  shingle = shingles.size();
	  shingles.emplace(word,shingles.size());
	} else {
	  shingle = shingleIterator -> second;
	}
	std::pair<int,int> shingleSetPair (relationOffset,shingle);
	//	std::cout << "shingleSetPair - set: " << relationOffset << " | shingle: " << shingle << "\n";
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
  std::uniform_int_distribution<int> distribution(1,primeForHashing);
  int a = distribution(generator);
  int b = distribution(generator);
  //  std::cout << "a: " << a << " b: " << b << "\n";
  for (int i = 0; i < hashMatrix.size(); i++) {
    hashMatrix[i] = ((((a*i)+b)%primeForHashing)%hashMatrix.size());
  }
}

void
buildSignatureMatrix(std::vector<int> &signatureMatrix)
{
  for (int i = 0; i < signatureMatrix.size(); i++) {
    signatureMatrix[i] = INT_MAX;
  }
}

void
computeSignatureMatrix(std::vector<int> &signatureMatrix, std::vector<int> &hashMatrix, ccsMatrix* characteristicMatrix, int numSets, int numShingles, int numBins)
{
  int binIdx, shingleIdx, offSetSM, offSetCM, shingleNewIdx;
  int binSize = numShingles/numBins;
  if (numShingles % numBins) binSize++;
  for (int i = 0; i < numSets; i++) {
    offSetCM = characteristicMatrix -> col_ptr[i];
    for (int j = offSetCM; j < characteristicMatrix -> col_ptr[i+1]; j++) {
      shingleIdx = characteristicMatrix -> row_ind[j];
      //      std::cout << "shingleidx: " << shingleIdx << " shingleNewIdx: " << hashMatrix[shingleIdx]  << "\n";
      binIdx = hashMatrix[shingleIdx]/binSize;
      offSetSM = i + (binIdx*numSets);
      if (signatureMatrix[offSetSM] > hashMatrix[shingleIdx]) {
	signatureMatrix[offSetSM] = hashMatrix[shingleIdx];
      }
    }
  }
}

void
computeSimilarities(std::vector<int> signatureMatrix, int rSize, int sSize, std::vector<std::string> relationRSetsID, std::vector<std::string> relationSSetsID, int numBins)
{
  int similarPairs = 0, numSets = rSize+sSize;

  for (int i = 0; i < rSize; i++) {
    for (int j = rSize; j < numSets; j++) {
      int identicalMinhashes = 0;
      int emptyBins = 0;
      for (int k = 0; k < numBins; k++) {
	if (signatureMatrix[i+(k*numSets)] == signatureMatrix[j+(k*numSets)]) {
	  if (signatureMatrix[i+(k*numSets)] == INT_MAX) {
	    emptyBins++;
	  } else {
	    identicalMinhashes++;
	  }
	}
      }
      float similarity = (identicalMinhashes*1.0)/((numBins*1.0) - (emptyBins*1.0));
      if (similarity >= SIMILARITY_THRESHOLD) {
	//	std::cout << "Identical minhashes: " << identicalMinhashes << " | Empty bins: " << emptyBins << "\n";
	similarPairs++;
      }
    }
  }
  std::cout << "Number of similar pairs: " << similarPairs << "\n";
}

int
main(int argc, char *argv[])
{
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << "pathToRelation/relationRFile.data pathToRelation/relationSFile.data\n";
    return 1;
  }

  std::multimap<int,int> shingleSetMap;
  std::map<std::string,int> shingles;
  std::vector<std::string> relationRSetsID;
  std::vector<std::string> relationSSetsID;  
  std::vector<int> primeNumbersForHashing = {7993, 13729, 23399, 39551, 65371, 102437, 164617, 274007, 441937, 7108087};

  int rSize = processInputRelation(shingleSetMap, shingles, relationRSetsID, argv[1] ,0);
  int sSize = processInputRelation(shingleSetMap, shingles, relationSSetsID, argv[2] ,rSize);
  int numSets = rSize + sSize;
  std::cout << "Number of sets (R + S = Total): " << rSize << " + " << sSize << " = " << numSets << "\n";

  int numShingles = shingles.size();
  std::cout << "Number of Shingles: " << numShingles << "\n";

  int numBins = NUM_BINS;
  std::cout << "numBins: " << numBins << "\n";

  int primeForHashing;
  for (int i = 0; i < primeNumbersForHashing.size(); i++) {
    if (numShingles < primeNumbersForHashing[i]) {
      primeForHashing = primeNumbersForHashing[i];
      break;
    }
  }
  std::cout << "Prime used: " << primeForHashing << "\n";

  ccsMatrix* characteristicMatrix = buildCharacteristicMatrix(shingleSetMap);
  std::cout << "characteristicMatrix Size: " << ccsMatrixSize(characteristicMatrix) << "\n";
  //  printCharacteristicMatrixVectors(characteristicMatrix);

  std::vector<int> hashMatrix (numShingles);
  buildHashMatrix(hashMatrix, primeForHashing);
  /*
  for (int i = 0; i < numShingles; i++) {
    std::cout << hashMatrix[i] << " ";
  }
  std::cout << "\n\n\n";
  */
  std::vector<int> signatureMatrix (numSets*numBins);
  std::cout << "signatureMatrixSize: " << signatureMatrix.size() << "\n";  
  buildSignatureMatrix(signatureMatrix);
  computeSignatureMatrix(signatureMatrix, hashMatrix, characteristicMatrix, numSets, numShingles, numBins);
  /*
  for (int i = 0; i < numBins; i++) {
    for (int j = i*numSets; j < (i*numSets)+numSets; j++) {
      std::cout << signatureMatrix[j] << " ";
    }
    std::cout << "\n";
  }
  */
  std::cout << "Computing similarities...\n";
  computeSimilarities(signatureMatrix, rSize, sSize, relationRSetsID, relationSSetsID, numBins);
  
  delete(characteristicMatrix);
  
  return 0;
}
