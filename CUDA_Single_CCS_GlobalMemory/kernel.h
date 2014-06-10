#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
//#include "characteristicMatrix.h"

typedef struct ccsMatrix ccsMatrix;
struct ccsMatrix{
  std::vector<int> row_ind;
  std::vector<int> col_ptr;
};

void kernelManager(std::vector<int> &h_signatureMatrix, ccsMatrix* h_characteristicMatrix, int numShingles, int primeForHashing, int sSize, int rSize, int numBins, int binSize);
