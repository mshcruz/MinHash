#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <limits.h>
#include <unordered_map>
#include "characteristicMatrix.h"
#include "kernel.h"

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

//Performs a nested loop to check the similarities between the sets and output pairs with similarity greater than the threshold
//*Change to a more efficient join algorithm
//*Run on parallel using GPU
 /*
void
computeSimilarities(vector<int> signatureMatrix, int rSize, int sSize, int numHashFunctions, float threshold, vector<string> relationRSetsID, vector<string> relationSSetsID)
{
  int i,j,k,l,identicalMinhashes,similarPairs=0;

  for (i = 0; i < (signatureMatrix.size()-(sSize*numHashFunctions)); i=i+numHashFunctions) {
    for (j = (rSize*numHashFunctions); j < signatureMatrix.size(); j=j+numHashFunctions) {
      identicalMinhashes = 0;
      if (i != j) {
	for (k = i, l = j; (k < i+numHashFunctions) && (l < j+numHashFunctions); k++, l++) {
	  if (signatureMatrix[k] == signatureMatrix[l]) {
	    identicalMinhashes++;
	  }
	}
	float similarity = (identicalMinhashes*1.0)/(numHashFunctions*1.0);
	if (similarity >= 0.6) {
	  //	  cout << "The similarity between record" << relationRSetsID[i/numHashFunctions] << " and record" << relationSSetsID[(j/numHashFunctions)-rSize] << " is " << similarity << "\n";
	  similarPairs++;
	}
      }
    }
  }
  cout << "Number of similar pairs: " << similarPairs << "\n";
}
 */

int
main(int argc, char *argv[])
{
  srand(time(NULL));

  if (argc < 3) {
    cout << "Usage: " << argv[0] << "pathToRelation/relationRFile.data pathToRelation/relationSFile.data\n";
    return 1;
  }

  int numHashFunctions = NUM_HASH_FUNCTIONS;
  unordered_multimap<int,int> h_shingleSetMap;
  unordered_map<string,int> h_shingles;
  vector<string> relationRSetsID;
  vector<string> relationSSetsID;  

  //Receive input relation files and creates a list of sets and lists of sets' IDs
  int rSize = processInputRelation(h_shingleSetMap, h_shingles, relationRSetsID, argv[1] ,0);
  int sSize = processInputRelation(h_shingleSetMap, h_shingles, relationSSetsID, argv[2] ,rSize);
  cout << "r Size: " << rSize << "\n";
  cout << "s Size: " << sSize << "\n";

  //Create the CRS representation of the characteristic matrix
  ccsMatrix* h_characteristicMatrix = buildCharacteristicMatrix(h_shingleSetMap);
  cout << "characteristicMatrix Size: " << ccsMatrixSize(h_characteristicMatrix) << "\n";
  //  printCharacteristicMatrixVectors (h_characteristicMatrix);

  //Calculate the number of unique words
  int numberOfUniqueWords = h_shingles.size();
  cout << "Number of Unique Words: " << numberOfUniqueWords << "\n";

  //Create the hash matrix
  //vector<int> h_hashMatrix = buildHashMatrix(numberOfUniqueWords, numHashFunctions);
  cout << "hmMatrix Size: " << numHashFunctions*numberOfUniqueWords << "\n";

  //Initialize and update the signature matrix (on GPU)
  vector<int> h_signatureMatrix ((rSize+sSize)*numHashFunctions);
  kernelManager(h_signatureMatrix, h_characteristicMatrix, numberOfUniqueWords, numHashFunctions, rSize, sSize, relationRSetsID, relationSSetsID);
  cout << "smMatrix Size: " << h_signatureMatrix.size() << "\n";

  //Perform a nested loop to check the similarities between sets
  
  //computeSimilarities(h_signatureMatrix, rSize, sSize, numHashFunctions, threshold, relationRSetsID, relationSSetsID);

  delete(h_characteristicMatrix);

  return 0;
}

