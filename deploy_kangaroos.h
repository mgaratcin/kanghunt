// deploy_kangaroos.h

#ifndef DEPLOY_KANGAROOS_H
#define DEPLOY_KANGAROOS_H

#include <vector>
#include <string>
#include "secp256k1/Int.h"

// Function declarations (no definitions here)
std::string hexToBinary(const std::string& hex);
void updateKangarooCounter(double power_of_two);
void deploy_kangaroos(const std::vector<Int>& kangaroo_batch);

#endif // DEPLOY_KANGAROOS_H
