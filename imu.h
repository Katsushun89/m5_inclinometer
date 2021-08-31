#pragma once

#include "utility/MPU9250.h"

void initIMU(MPU9250 *imu);
bool calcIMU(MPU9250 *imu);
