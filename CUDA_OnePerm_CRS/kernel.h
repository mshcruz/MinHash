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

typedef struct crsMatrix crsMatrix;
struct crsMatrix{
  //  vector<int> val;
  vector<int> col_ind;
  vector<int> row_ptr;
};

void kernelManager(vector<int> &h_signatureMatrix, crsMatrix* h_characteristicMatrix, int lwSize, int numHashFunctions, int sSize, int rSize, vector<string> relationRSetsID, vector<string> relationSSetsID);
