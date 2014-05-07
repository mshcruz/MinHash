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

#define BIN_SIZE 15

using namespace std;

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

/*
int
processInputRelation(unordered_multimap<int, string> &setWordMap, vector<string> &setsIDs, string fileName)
{
  ifstream relation (fileName);
  string line = "";
  int relationSize = 0;

  if (relation.is_open()){
    while (getline(relation,line)){
      vector<string> wordsOfSet;
      istringstream tuple(line);
      string value;
      getline(tuple,value,'\t');
      setsIDs.push_back(value);
      getline(tuple,value,'\t');      
      istringstream sentence(value);
      string word;
      while(getline(sentence,word,' ')){
	wordsOfSet.push_back(word);
      }
      relationSize++;
      listOfSets.push_back(wordsOfSet);
    }    
    relation.close();
  } else {
    cout << "Error opening file.";
  }

  return relationSize;
}
*/
 /*
vector<string>
buildListOfUniqueWords(vector<vector<string> > listOfSets)
{
  unordered_map<string,double> uniqueWords;
  vector<string> listOfWords;
  int uniqueWordsCount = 0;
  
  for (vector<string> set : listOfSets) {
    for (string word : set) {
      if (uniqueWords.find(word) == uniqueWords.end()) {
	uniqueWords.emplace(word,uniqueWords.size());
      }
    }
  }
  for (auto kv : uniqueWords) {
    listOfWords.push_back(kv.first);
  }

  return listOfWords;
}
 */
/*
//Inefficient because it runs over the list of unique words many times
vector<string>
buildListOfUniqueWords(vector<vector<string> > listOfSets)
{
  vector<string> listOfWords;

  for (vector<string> set : listOfSets) {
    for (string word : set) {
      if (find(listOfWords.begin(), listOfWords.end(), word) == listOfWords.end()) {
	listOfWords.push_back(word);
      }
    }
  }

  return listOfWords;
}
*/

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
buildSignatureMatrix(int numSets, int numBins)
{
  vector<int> signatureMatrix;

  for (int i = 0; i < numSets; i++) {
    for (int j = 0; j < numBins; j++) {
      signatureMatrix.push_back(INT_MAX);
    }
  }

  return signatureMatrix;
}

void
computeSignatureMatrix(vector<int> &signatureMatrix, crsMatrix* characteristicMatrix, unordered_multimap<string,int> wordSetMap, int numSets, int numBins)
{
  int uniqueWordIdx = 0, j = 0, k = 0, l = 0, binIdx = 0;
  
  for (auto it = wordSetMap.begin(); it != wordSetMap.end(); it = wordSetMap.equal_range(it->first).second){
    //    cout << "Word: " << uniqueWordIdx << "-" << it->first << " | Bin idx: " << binIdx;
    int hashForWord = generateHash();
    //    cout << " | hash: " << hashForWord << "\n";
    int offSetCM = characteristicMatrix -> row_ptr[uniqueWordIdx];
    for (j = offSetCM; j < characteristicMatrix -> row_ptr[uniqueWordIdx+1]; j++) {
      int setIdx = characteristicMatrix -> col_ind[j];
      int offSetSM = (setIdx*numBins)+binIdx;
      if (signatureMatrix[offSetSM] > hashForWord) {
	signatureMatrix[offSetSM] = hashForWord;
      }
      /*
      for (int m = 0; m < numSets; m++) {
	for (int n = m*numBins; n < (m*numBins)+numBins; n++) {
	  cout << signatureMatrix[n] << "\t";
	}
	cout << "\n";
      }
      cout << "\n";
      */
    }
    uniqueWordIdx++;
    if (uniqueWordIdx%BIN_SIZE == 0) {
      binIdx++;
    }
  }


  //  for (int m = 0; m < signatureMatrix.size(); m++) {
  //    cout << signatureMatrix[m] << "\t";
  //  }
  //  cout << "\n\n";
  /*
  for (int m = 0; m < numSets; m++) {
    for (int n = m*numBins; n < (m*numBins)+numBins; n++) {
      cout << signatureMatrix[n] << "\t";
    }
    cout << "\n";
  }
  */
}

void
computeSimilarities(vector<int> signatureMatrix, int rSize, int sSize, int numBins, vector<string> relationRSetsID, vector<string> relationSSetsID)
{
  int i = 0, j = 0, k = 0, l = 0, identicalMinhashes = 0, similarPairs = 0, emptyBins = 0;

  for (i = 0; i < (signatureMatrix.size()-(sSize*numBins)); i=i+numBins) {
    for (j = (rSize*numBins); j < signatureMatrix.size(); j=j+numBins) {
      emptyBins = 0;
      identicalMinhashes = 0;
      if (i != j) {
	for (k = i, l = j; (k < i+numBins) && (l < j+numBins); k++, l++) {
	  if (signatureMatrix[k] == signatureMatrix[l]) {
	    if (signatureMatrix[k] == INT_MAX) {
	      emptyBins++;
	    } else {
	      identicalMinhashes++;
	    }
	  }
	}
	float similarity = (identicalMinhashes*1.0)/((numBins*1.0)-(emptyBins*1.0));
	if (similarity >= 0.6) {
	  //	  cout << "The similarity between record" << relationRSetsID[i/numBins] << " and record" << relationSSetsID[(j/numBins)-rSize] << " is " << similarity << "\n";
	  similarPairs++;
	}
      }
    }
  }
  cout << "Number of similar pairs: " << similarPairs << "\n";
}

int
uniqueWordsNumber(unordered_multimap<string,int> wordSetMap)
{
  int count = 0;
  for (auto it = wordSetMap.begin(); it != wordSetMap.end(); it = wordSetMap.equal_range(it->first).second){
    //    cout << "Set: " << (it -> second) << "\n";
    //    cout << "Word: " << it -> first << "\n";
    count++;
  }
  return count;
}

int
main(int argc, char *argv[])
{
  srand(time(NULL));

  unordered_multimap<string,int> wordSetMap;
  vector<string> relationRSetsID;
  vector<string> relationSSetsID;

  int rSize = processInputRelation(wordSetMap, relationRSetsID, "newRelationR_ID.data",0);
  int sSize = processInputRelation(wordSetMap, relationSSetsID, "newRelationS_ID.data",rSize);
  //int rSize = processInputRelation(wordSetMap, relationRSetsID, "relationR.csv",0);
  //int sSize = processInputRelation(wordSetMap, relationSSetsID, "relationS.csv",rSize);

  cout << "sSize: " << sSize << "\n";
  cout << "rSize: " << rSize << "\n";

  int numUniqueWords = uniqueWordsNumber(wordSetMap);
  cout << "Number of unique words: " << numUniqueWords << "\n";

  int numBins = numUniqueWords / BIN_SIZE;
  cout << "binSize: " << BIN_SIZE << "\n";
  cout << "numBins: " << numBins << "\n";

  crsMatrix *characteristicMatrix = buildCharacteristicMatrix(wordSetMap);
  cout << "characteristicMatrixSize: " << crsMatrixSize(characteristicMatrix) << "\n";

  vector<int> signatureMatrix = buildSignatureMatrix(rSize+sSize, numBins);
  cout << "signatureMatrixSize: " << signatureMatrix.size() << "\n";
  
  computeSignatureMatrix(signatureMatrix, characteristicMatrix, wordSetMap, rSize+sSize, numBins);
  
  computeSimilarities(signatureMatrix, rSize, sSize, numBins, relationRSetsID, relationSSetsID);

  delete(characteristicMatrix);
  
  return 0;
}

