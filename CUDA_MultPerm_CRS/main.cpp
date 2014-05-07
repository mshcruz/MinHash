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
//*Is it possible to parallelize file reading? 
//*How to treat stop words?
int
processInputRelation(unordered_multimap<string, int>& setWordMap, vector<string> &setsIDs, string fileName, int relationOffset)
{
  ifstream relation (fileName);
  string word, value, line = "";
  int relationSize = 0;

  if (relation.is_open()){
    while (getline(relation,line)){
      value = "";
      word = "";
      istringstream tuple(line); 
      getline(tuple,value,'\t');
      setsIDs.push_back(value); //Obtain the ID of the record
      getline(tuple,value,'\t');      
      istringstream sentence(value);
      while(getline(sentence,word,' ')){ //Each new word becomes a key of the unordered map and it has one or more sets associated to it
	pair<string,int> wordSetPair (word,relationOffset);
	setWordMap.insert(wordSetPair);
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
numberOfUniqueWords(unordered_multimap<string,int> wordSetMap)
{
  int count = 0;
  for (auto it = wordSetMap.begin(); it != wordSetMap.end();){
    count++;
    it = wordSetMap.equal_range(it->first).second;
  }
  return count;
}

int
main(int argc, char *argv[])
{
  int numHashFunctions = NUM_HASH_FUNCTIONS;
  //  float threshold = SIMILARITY_THRESHOLD;
  unordered_multimap<string,int> h_wordSetMap;
  vector<string> relationRSetsID;
  vector<string> relationSSetsID;  

  //Receive input relation files and creates a list of sets and lists of sets' IDs
  int rSize = processInputRelation(h_wordSetMap, relationRSetsID, "newRelationR_ID.data",0);
  int sSize = processInputRelation(h_wordSetMap, relationSSetsID, "newRelationS_ID.data",rSize);
  cout << "r Size: " << rSize << "\n";
  cout << "s Size: " << sSize << "\n";

  //Create the CRS representation of the characteristic matrix
  crsMatrix* h_characteristicMatrix = buildCharacteristicMatrix(h_wordSetMap);
  cout << "characteristicMatrix Size: " << crsMatrixSize(h_characteristicMatrix) << "\n";

  //Initialize and update the signature matrix (on GPU)
  vector<int> h_signatureMatrix ((rSize+sSize)*numHashFunctions);
  kernelManager(h_signatureMatrix, h_characteristicMatrix, numberOfUniqueWords(h_wordSetMap), numHashFunctions, rSize, sSize, relationRSetsID, relationSSetsID);
  //  cout << "smMatrix Size: " << h_signatureMatrix.size() << "\n";

  //Perform a nested loop to check the similarities between sets
  //  computeSimilarities(h_signatureMatrix, rSize, sSize, numHashFunctions, threshold, relationRSetsID, relationSSetsID);

  delete(h_characteristicMatrix);

  return 0;
}

