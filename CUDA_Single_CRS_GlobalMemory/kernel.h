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

typedef struct crsMatrix ccsMatrix;
struct crsMatrix{
  std::vector<int> col_ind;
  std::vector<int> row_ptr;
};

void kernelManager(std::vector<int> &h_signatureMatrix, crsMatrix* h_characteristicMatrix, int numShingles, int primeForHashing, int sSize, int rSize, int numBins, int binSize);
