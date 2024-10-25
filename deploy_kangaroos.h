//Copyright 2024 MGaratcin//  
//All rights reserved.//
//This code is proprietary and confidential. Unauthorized copying, distribution,//
//modification, or any other use of this code, in whole or in part, is strictly//
//prohibited. The use of this code without explicit written permission from the//
//copyright holder is not permitted under any circumstances.//

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
