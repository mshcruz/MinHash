#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <limits.h>
#include <map>
#include "characteristicMatrix.h"
#include "kernel.h"

#define SIMILARITY_THRESHOLD 0.6
#define NUM_BINS 256

//Read files containing sentences and insert them to the list of sets
int
processInputRelation(std::multimap<int,int>& shingleSetMap, std::map<std::string,int>& shingles, std::vector<std::string>& setsIDs, std::string fileName, int relationOffset)
{
  std::ifstream relation (fileName);
  std::string word, value, line = "";
  int shingle, relationSize = 0;
  std::map<std::string, int>::iterator shingleIterator;

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
	//Each new word becomes a key of the unordered map and it has one or more sets associated to it
	shingleIterator = shingles.find(word);
	if (shingleIterator == shingles.end()) {
	  shingle = shingles.size();
	  shingles.emplace(word,shingles.size());
	} else {
	  shingle = shingleIterator -> second;
	}
	std::pair<int,int> shingleSetPair (relationOffset,shingle);
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
computeSimilarities(std::vector<int> signatureMatrix, int rSize, int sSize, std::vector<std::string> relationRSetsID, std::vector<std::string> relationSSetsID, int numBins)
{
  int identicalMinhashes, emptyBins, similarPairs = 0, numSets = rSize+sSize;
  float similarity;

  for (int i = 0; i < rSize; i++) {
    for (int j = rSize; j < numSets; j++) {
      identicalMinhashes = 0;
      emptyBins = 0;
      for (int k = 0; k < numBins; k++) {
	if (signatureMatrix[(i*numBins)+k] == signatureMatrix[(j*numBins)+k]) {
	  if (signatureMatrix[(i*numBins)+k] == INT_MAX) {
	    emptyBins++;
	  } else {
	    identicalMinhashes++;
	  }
	}
      }
      similarity = (identicalMinhashes*1.0)/((numBins*1.0) - (emptyBins*1.0));
      if (similarity >= SIMILARITY_THRESHOLD) {
	//	std::cout << "The similarity between " << relationRSetsID[i] << " and " << relationSSetsID[j]  << " is " << similarity << "\n";
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

  std::multimap<int,int> h_shingleSetMap;
  std::map<std::string,int> h_shingles;
  std::vector<std::string> relationRSetsID;
  std::vector<std::string> relationSSetsID;  
  std::vector<int> primeNumbersForHashing = {7993, 13729, 23399, 39551, 65371, 102437, 164617, 274007, 441937, 7108087};

  //Receive input relation files and creates a list of sets and lists of sets' IDs
  int rSize = processInputRelation(h_shingleSetMap, h_shingles, relationRSetsID, argv[1] ,0);
  int sSize = processInputRelation(h_shingleSetMap, h_shingles, relationSSetsID, argv[2] ,rSize);
  int numSets = rSize + sSize;
  std::cout << "Number of sets (R + S = Total): " << rSize << " + " << sSize << " = " << numSets << "\n";

  //Calculate the number of unique shingles
  int numShingles = h_shingles.size();
  std::cout << "Number of Shingles: " << numShingles << "\n";

  //Calculate the number of bins
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

  //Create the CCS representation of the characteristic matrix
  ccsMatrix* h_characteristicMatrix = buildCharacteristicMatrix(h_shingleSetMap);
  std::cout << "characteristicMatrix Size: " << ccsMatrixSize(h_characteristicMatrix) << "\n";
  //  printCharacteristicMatrixVectors (h_characteristicMatrix);

  //Initialize and update the signature matrix (on GPU)
  std::vector<int> h_signatureMatrix (numSets*numBins);
  std::cout << "smMatrix Size: " << h_signatureMatrix.size() << "\n";
  kernelManager(h_signatureMatrix, h_characteristicMatrix, numShingles, primeForHashing, rSize, sSize, numBins, binSize);
  /*
  printf("printing SM...\n");
  for (int m = 0; m < h_signatureMatrix.size(); m++) {
    printf("%d ", h_signatureMatrix[m]);
  }
  */
  //Perform a nested loop to check the similarities between sets
  computeSimilarities(h_signatureMatrix, rSize, sSize, relationRSetsID, relationSSetsID, numBins);  

  delete(h_characteristicMatrix);

  return 0;
}

