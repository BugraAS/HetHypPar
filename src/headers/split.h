#pragma once

#include "common.h"

SPLIT_CSR* cleanSplit(CSR in, int* partvec);
SPLIT_CSR* sparseSplit(CSR big, int* partvec);