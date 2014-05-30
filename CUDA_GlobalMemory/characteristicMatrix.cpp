#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "characteristicMatrix.h"

ccsMatrix* 
buildCharacteristicMatrix(unordered_multimap<int,int> shingleSetMap){
  /*
  for (auto it = shingleSetMap.begin(); it != shingleSetMap.end(); it++){
    cout << "\n";
    cout << "it->first: " << it->first << "\n";
    cout << "it->second: " << it->second << "\n";
  }
  */
  ccsMatrix *characteristicMatrix = new ccsMatrix();
  int newColumnFlag;
  int addedElements = 0;
  for (auto it = shingleSetMap.begin(); it != shingleSetMap.end();){
    newColumnFlag = 1;
    pair<unordered_multimap<int,int>::iterator, unordered_multimap<int,int>::iterator> ii = shingleSetMap.equal_range(it->first);
    unordered_multimap<int,int>::iterator i;
    for (i = ii.first; i != ii.second; ++i) { //Checks what words are contained in the current set
        characteristicMatrix -> row_ind.push_back(i -> second);
      if (newColumnFlag) {
	characteristicMatrix -> col_ptr.push_back(addedElements);
	newColumnFlag = 0;
      }
      addedElements++;
    }
    it = i; //Goes to the next set
  }
  characteristicMatrix -> col_ptr.push_back(addedElements);
  return characteristicMatrix;

  /*
  crsMatrix *characteristicMatrix = new crsMatrix();
  int newRowFlag;
  int addedElements = 0;
  for (auto it = wordSetMap.begin(); it != wordSetMap.end();){
    newRowFlag = 1;
    pair<unordered_multimap<string,int>::iterator, unordered_multimap<string,int>::iterator> ii = wordSetMap.equal_range(it->first);
    unordered_multimap<string,int>::iterator i;
    for (i = ii.first; i != ii.second; ++i) { //Checks what sets contain the current unique word
      characteristicMatrix -> val.push_back(1);
      characteristicMatrix -> col_ind.push_back(i -> second);
      if (newRowFlag) {
	characteristicMatrix -> row_ptr.push_back(addedElements);
	newRowFlag = 0;
      }
      addedElements++;
    }
    it = i; //Goes to the next unique word
  }
  characteristicMatrix -> row_ptr.push_back(addedElements);
  return characteristicMatrix;
   */
}

void 
printCharacteristicMatrixVectors (ccsMatrix *characteristicMatrix)
{
  cout << "row_ind vector - Size: " << characteristicMatrix -> row_ind.size() << "\n";
  for (int value : characteristicMatrix -> row_ind) {
    cout << value << " ";
  }
  cout << "\n\n";

  cout << "col_ptr vector - Size: " << characteristicMatrix -> col_ptr.size() << "\n";
  for (int value : characteristicMatrix -> col_ptr) {
    cout << value << " ";
  }
  cout << "\n\n";
}

int 
ccsMatrixSize (ccsMatrix *characteristicMatrix)
{
  return (characteristicMatrix -> row_ind.size()) + characteristicMatrix -> col_ptr.size(); 
}
