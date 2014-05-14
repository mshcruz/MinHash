#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "characteristicMatrix.h"

crsMatrix* 
buildCharacteristicMatrix(unordered_multimap<string,int> wordSetMap){
  crsMatrix *characteristicMatrix = new crsMatrix();
  int newRowFlag;
  int addedElements = 0;
  for (auto it = wordSetMap.begin(); it != wordSetMap.end();){
    newRowFlag = 1;
    pair<unordered_multimap<string,int>::iterator, unordered_multimap<string,int>::iterator> ii = wordSetMap.equal_range(it->first);
    unordered_multimap<string,int>::iterator i;
    for (i = ii.first; i != ii.second; ++i) { //Checks what sets contain the current unique word
      //      characteristicMatrix -> val.push_back(1);
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
}


void 
printCharacteristicMatrixVectors (crsMatrix *characteristicMatrix)
{
  /*
  cout << "val vector - Size: " << characteristicMatrix -> val.size() << "\n";
  for (int value : characteristicMatrix -> val) {
    cout << value << " ";
  }
  cout << "\n\n";
  */
  cout << "col_ind vector - Size: " << characteristicMatrix -> col_ind.size() << "\n";
  for (int value : characteristicMatrix -> col_ind) {
    cout << value << " ";
  }
  cout << "\n\n";

  cout << "row_ptr vector - Size: " << characteristicMatrix -> row_ptr.size() << "\n";
  for (int value : characteristicMatrix -> row_ptr) {
    cout << value << " ";
  }
  cout << "\n\n";
}

int 
crsMatrixSize (crsMatrix *characteristicMatrix)
{
  return (characteristicMatrix -> col_ind.size()) + characteristicMatrix -> row_ptr.size(); 
}
