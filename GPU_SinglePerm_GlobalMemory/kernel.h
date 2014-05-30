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

using namespace std;

typedef struct ccsMatrix ccsMatrix;
struct ccsMatrix{
  vector<int> row_ind;
  vector<int> col_ptr;
};

void kernelManager(vector<int> &h_signatureMatrix, ccsMatrix* h_characteristicMatrix, int lwSize, int numHashFunctions, int sSize, int rSize, vector<string> relationRSetsID, vector<string> relationSSetsID);
