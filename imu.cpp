#include "imu.h"

#include <M5Stack.h>

#include "utility/MPU9250.h"
#include "utility/quaternionFilters.h"

#define SerialDebug (false)
//#define LCD

boolean resetMPU9250 = false;

void initIMU(MPU9250 *imu) {
    byte c = imu->readByte(MPU9250_ADDRESS, WHO_AM_I_MPU9250);
    Serial.print("MPU9250 ");
    Serial.print("I AM ");
    Serial.print(c, HEX);
    Serial.print(" I should be ");
    Serial.println(0x71, HEX);

    if (c == 0x71) {  // WHO_AM_I should always be 0x68
        Serial.println("MPU9250 is online...");

        // Start by performing self test and reporting values
        imu->MPU9250SelfTest(imu->SelfTest);
        Serial.print("x-axis self test: acceleration trim within : ");
        Serial.print(imu->SelfTest[0], 1);
        Serial.println("% of factory value");
        Serial.print("y-axis self test: acceleration trim within : ");
        Serial.print(imu->SelfTest[1], 1);
        Serial.println("% of factory value");
        Serial.print("z-axis self test: acceleration trim within : ");
        Serial.print(imu->SelfTest[2], 1);
        Serial.println("% of factory value");
        Serial.print("x-axis self test: gyration trim within : ");
        Serial.print(imu->SelfTest[3], 1);
        Serial.println("% of factory value");
        Serial.print("y-axis self test: gyration trim within : ");
        Serial.print(imu->SelfTest[4], 1);
        Serial.println("% of factory value");
        Serial.print("z-axis self test: gyration trim within : ");
        Serial.print(imu->SelfTest[5], 1);
        Serial.println("% of factory value");

        // Calibrate gyro and accelerometers, load biases in bias registers
        // imu->calibrateMPU9250(imu->gyroBias, imu->accelBias);
        delay(1000);

        imu->initMPU9250();

        byte d = imu->readByte(AK8963_ADDRESS, WHO_AM_I_AK8963);
        Serial.print("AK8963 ");
        Serial.print("I AM ");
        Serial.print(d, HEX);
        Serial.print(" I should be ");
        Serial.println(0x48, HEX);
        if (d == 0x48) {
            Serial.println("AK8963 is online...");

            imu->initAK8963(imu->magCalibration);
        }
    }
}

bool calcIMU(MPU9250 *imu) {
    if (imu->readByte(MPU9250_ADDRESS, INT_STATUS) & 0x01) {
        imu->readAccelData(imu->accelCount);  // Read the x/y/z adc values
        imu->getAres();

        // Now we'll calculate the accleration value into actual g's
        // This depends on scale being set
        imu->ax = (float)imu->accelCount[0] * imu->aRes;  // - accelBias[0];
        imu->ay = (float)imu->accelCount[1] * imu->aRes;  // - accelBias[1];
        imu->az = (float)imu->accelCount[2] * imu->aRes;  // - accelBias[2];

        imu->readGyroData(imu->gyroCount);  // Read the x/y/z adc values
        imu->getGres();

        // Calculate the gyro value into actual degrees per second
        // This depends on scale being set
        imu->gx = (float)imu->gyroCount[0] * imu->gRes;
        imu->gy = (float)imu->gyroCount[1] * imu->gRes;
        imu->gz = (float)imu->gyroCount[2] * imu->gRes;

#if 0
    imu->readMagData(imu->magCount);  // Read the x/y/z adc values
    imu->getMres();
    // User environmental x-axis correction in milliGauss, should be
    // automatically calculated
    imu->magbias[0] = +470.;
    // User environmental x-axis correction in milliGauss TODO axis??
    imu->magbias[1] = +120.;
    // User environmental x-axis correction in milliGauss
    imu->magbias[2] = +125.;

    // Calculate the magnetometer values in milliGauss
    // Include factory calibration per data sheet and user environmental
    // corrections
    // Get actual magnetometer value, this depends on scale being set
    imu->mx = (float)imu->magCount[0]*imu->mRes*imu->magCalibration[0] -
               imu->magbias[0];
    imu->my = (float)imu->magCount[1]*imu->mRes*imu->magCalibration[1] -
               imu->magbias[1];
    imu->mz = (float)imu->magCount[2]*imu->mRes*imu->magCalibration[2] -
               imu->magbias[2];
#endif
        // Must be called before updating quaternions!
        imu->updateTime();

#if 0
        // Sensors x (y)-axis of the accelerometer is aligned with the y
        // (x)-axis of the magnetometer; the magnetometer z-axis (+ down) is
        // opposite to z-axis
        // (+ up) of accelerometer and gyro! We have to make some allowance for
        // this orientationmismatch in feeding the output to the quaternion
        // filter. For the MPU-9250, we have chosen a magnetic rotation that
        // keeps the sensor forward along the x-axis just like in the LSM9DS0
        // sensor. This rotation can be modified to allow any convenient
        // orientation convention. This is ok by aircraft orientation standards!
        // Pass gyro rate as rad/s
        MahonyQuaternionUpdate(imu->ax, imu->ay, imu->az, imu->gx * DEG_TO_RAD,
                               imu->gy * DEG_TO_RAD, imu->gz * DEG_TO_RAD,
                               imu->mx, imu->my, imu->mz, imu->deltat);

        // Serial print and/or display at 0.5 s rate independent of data rates
        imu->delt_t = millis() - imu->count;

        // update LCD once per half-second independent of read rate
        // if (imu->delt_t > 500)
        if (imu->delt_t > 100) {
            if (SerialDebug) {
                Serial.print("ax = ");
                Serial.print((int)1000 * imu->ax);
                Serial.print(" ay = ");
                Serial.print((int)1000 * imu->ay);
                Serial.print(" az = ");
                Serial.print((int)1000 * imu->az);
                Serial.println(" mg");

                Serial.print("gx = ");
                Serial.print(imu->gx, 2);
                Serial.print(" gy = ");
                Serial.print(imu->gy, 2);
                Serial.print(" gz = ");
                Serial.print(imu->gz, 2);
                Serial.println(" deg/s");

                Serial.print("mx = ");
                Serial.print((int)imu->mx);
                Serial.print(" my = ");
                Serial.print((int)imu->my);
                Serial.print(" mz = ");
                Serial.print((int)imu->mz);
                Serial.println(" mG");

                Serial.print("q0 = ");
                Serial.print(*getQ());
                Serial.print(" qx = ");
                Serial.print(*(getQ() + 1));
                Serial.print(" qy = ");
                Serial.print(*(getQ() + 2));
                Serial.print(" qz = ");
                Serial.println(*(getQ() + 3));
            }

            // Define output variables from updated quaternion---these are
            // Tait-Bryan angles, commonly used in aircraft orientation. In this
            // coordinate system, the positive z-axis is down toward Earth. Yaw
            // is the angle between Sensor x-axis and Earth magnetic North (or
            // true North if corrected for local declination, looking down on
            // the sensor positive yaw is counterclockwise. Pitch is angle
            // between sensor x-axis and Earth ground plane, toward the Earth is
            // positive, up toward the sky is negative. Roll is angle between
            // sensor y-axis and Earth ground plane, y-axis up is positive roll.
            // These arise from the definition of the homogeneous rotation
            // matrix constructed from quaternions. Tait-Bryan angles as well as
            // Euler angles are non-commutative; that is, the get the correct
            // orientation the rotations must be applied in the correct order
            // which for this configuration is yaw, pitch, and then roll. For
            // more see
            // http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
            // which has additional links.
            imu->yaw = atan2(2.0f * (*(getQ() + 1) * *(getQ() + 2) +
                                     *getQ() * *(getQ() + 3)),
                             *getQ() * *getQ() + *(getQ() + 1) * *(getQ() + 1) -
                                 *(getQ() + 2) * *(getQ() + 2) -
                                 *(getQ() + 3) * *(getQ() + 3));
            imu->pitch = -asin(2.0f * (*(getQ() + 1) * *(getQ() + 3) -
                                       *getQ() * *(getQ() + 2)));
            imu->roll =
                atan2(2.0f * (*getQ() * *(getQ() + 1) +
                              *(getQ() + 2) * *(getQ() + 3)),
                      *getQ() * *getQ() - *(getQ() + 1) * *(getQ() + 1) -
                          *(getQ() + 2) * *(getQ() + 2) +
                          *(getQ() + 3) * *(getQ() + 3));
            imu->pitch *= RAD_TO_DEG;
            imu->yaw *= RAD_TO_DEG;
            // Declination of SparkFun Electronics (40°05'26.6"N 105°11'05.9"W)
            // is 	8° 30' E  ± 0° 21' (or 8.5°) on 2016-07-19
            // - http://www.ngdc.noaa.gov/geomag-web/#declination
            imu->yaw -= 8.5;
            imu->roll *= RAD_TO_DEG;

            if (SerialDebug) {
                Serial.print("Yaw, Pitch, Roll: ");
                Serial.print(imu->yaw, 2);
                Serial.print(", ");
                Serial.print(imu->pitch, 2);
                Serial.print(", ");
                Serial.println(imu->roll, 2);

                Serial.print("rate = ");
                Serial.print((float)imu->sumCount / imu->sum, 2);
                Serial.println(" Hz");
                Serial.println("");
            }
#ifdef LCD
        // M5.Lcd.fillScreen(BLACK);
        M5.Lcd.setTextFont(2);

        M5.Lcd.setCursor(0, 0);
        M5.Lcd.print("     x       y       z ");
        M5.Lcd.setCursor(0, 24);
        M5.Lcd.printf("% 6d  % 6d  % 6d     mg   \r\n", (int)(1000 * imu->ax),
                      (int)(1000 * imu->ay), (int)(1000 * imu->az));
        M5.Lcd.setCursor(0, 44);
        M5.Lcd.printf("% 6d  % 6d  % 6d      o/s  \r\n", (int)(imu->gx),
                      (int)(imu->gy), (int)(imu->gz));
        M5.Lcd.setCursor(0, 64);
        M5.Lcd.printf("% 6d  % 6d  % 6d     mG    \r\n", (int)(imu->mx),
                      (int)(imu->my), (int)(imu->mz));

        M5.Lcd.setCursor(0, 100);
        M5.Lcd.printf("  yaw: % 5.2f    pitch: % 5.2f    roll: % 5.2f   \r\n",
                      (imu->yaw), (imu->pitch), (imu->roll));

        // With these settings the filter is updating at a ~145 Hz rate
        // using the Madgwick scheme and >200 Hz using the Mahony scheme
        // even though the display refreshes at only 2 Hz. The filter update
        // rate is determined mostly by the mathematical steps in the
        // respective algorithms, the processor speed (8 MHz for the 3.3V
        // Pro Mini), and the magnetometer ODR: an ODR of 10 Hz for the
        // magnetometer produce the above rates, maximum magnetometer ODR of
        // 100 Hz produces filter update rates of 36 - 145 and ~38 Hz for
        // the Madgwick and Mahony schemes, respectively. This is presumably
        // because the magnetometer read takes longer than the gyro or
        // accelerometer reads. This filter update rate should be fast
        // enough to maintain accurate platform orientation for
        // stabilization control of a fast-moving robot or quadcopter.
        // Compare to the update rate of 200 Hz produced by the on-board
        // Digital Motion Processor of Invensense's MPU6050 6 DoF and
        // MPU9150 9DoF sensors. The 3.3 V 8 MHz Pro Mini is doing pretty
        // well!

        // M5.Lcd.setCursor(0, 60);
        // M5.Lcd.printf("yaw:%6.2f   pitch:%6.2f   roll:%6.2f  ypr
        // \r\n",(imu->yaw), (imu->pitch), (imu->roll));
        M5.Lcd.setCursor(12, 144);
        M5.Lcd.print("rt: ");
        M5.Lcd.print((float)imu->sumCount / imu->sum, 2);
        M5.Lcd.print(" Hz");
#endif  // LCD

        imu->count = millis();
        imu->sumCount = 0;

        imu->sum = 0;
        return true;
    }  // if (imu->delt_t > 500)
#endif

    }  // if (readByte(MPU9250_ADDRESS, INT_STATUS) & 0x01)
    return false;
}
