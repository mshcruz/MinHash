#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "characteristicMatrix.h"

crsMatrix* 
buildCharacteristicMatrix(std::multimap<std::string,int> shingleSetMap){
  crsMatrix *characteristicMatrix = new crsMatrix();
  int newRowFlag;
  int addedElements = 0;
  for (auto it = shingleSetMap.begin(); it != shingleSetMap.end();){
    //std::cout << "*first: " << it-> first << "\n";
    newRowFlag = 1;
    std::pair<std::multimap<std::string,int>::iterator, std::multimap<std::string,int>::iterator> ii = shingleSetMap.equal_range(it->first);
    std::multimap<std::string,int>::iterator i;
    for (i = ii.first; i != ii.second; ++i) { //Checks what sets contain the current unique word
      //std::cout << "second: " << i->second << "\n";
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
  /*  cout << "val vector - Size: " << characteristicMatrix -> val.size() << "\n";
  for (int value : characteristicMatrix -> val) {
    cout << value << " ";
  }
  cout << "\n\n";
  */
  std::cout << "col_ind vector - Size: " << characteristicMatrix -> col_ind.size() << "\n";
  for (int value : characteristicMatrix -> col_ind) {
    std::cout << value << " ";
  }
  std::cout << "\n\n";

  std::cout << "row_ptr vector - Size: " << characteristicMatrix -> row_ptr.size() << "\n";
  for (int value : characteristicMatrix -> row_ptr) {
    std::cout << value << " ";
  }
  std::cout << "\n\n";
}

int 
crsMatrixSize (crsMatrix *characteristicMatrix)
{
  return (characteristicMatrix -> col_ind.size() * 2) + characteristicMatrix -> row_ptr.size(); 
}
