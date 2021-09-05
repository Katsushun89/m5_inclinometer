//#define M5STACK_MPU6886
#define M5STACK_MPU9250
#include <M5Stack.h>
#include <Ticker.h>

float acc_x = 0.0F;
float acc_y = 0.0F;
float acc_z = 0.0F;

float gyro_x = 0;
float gyro_y = 0;
float gyro_z = 0;

// Ticker
// Ticker timer1;

// Accelerometer and gyro statistical data
int sample_num = 100;
int meas_interval = 10;  // ms
float theta_mean;
float theta_variance;
float theta_dot_mean;
float theta_dot_variance;
float theta_acc;

// Kalman filter (for angle estimation) variables
// Update rate
float theta_update_freq = 400;  // Hz
float theta_update_interval = 1.0 / double(theta_update_freq);

// State vector
// theta, theta dot
float theta_data_predict[2][1];
float theta_data[2][1];

// covariance matrix
float P_theta_predict[2][2];
float P_theta[2][2];

// A of the state equation
float A_theta[2][2] = {{1, -theta_update_interval}, {0, 1}};

// B of the state equation
float B_theta[2][1] = {{theta_update_interval}, {0}};

// C of the state equation
float C_theta[1][2] = {{1, 0}};

// hw_timer
volatile bool is_exec_timer = false;
volatile uint32_t last_timer_call;
volatile SemaphoreHandle_t timer_semaphore;
portMUX_TYPE timer_mux = portMUX_INITIALIZER_UNLOCKED;
hw_timer_t *timer = NULL;

// Matrix common functions
// Matrix addition
void mat_add(float *m1, float *m2, float *sol, int row, int column) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            int index = i * column + j;
            sol[index] = m1[index] + m2[index];
        }
    }
}

// Matrix subtraction
void mat_sub(float *m1, float *m2, float *sol, int row, int column) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            int index = i * column + j;
            sol[index] = m1[index] - m2[index];
        }
    }
}

// Matrix multiplication
void mat_mul(float *m1, float *m2, float *sol, int row1, int column1, int row2,
             int column2) {
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < column2; j++) {
            sol[i * column2 + j] = 0;
            for (int k = 0; k < column1; k++) {
                sol[i * column2 + j] +=
                    m1[i * column1 + k] * m2[k * column2 + j];
            }
        }
    }
}

// Matrix transposition
void mat_tran(float *m1, float *sol, int row_original, int column_original) {
    for (int i = 0; i < row_original; i++) {
        for (int j = 0; j < column_original; j++) {
            sol[j * row_original + i] = m1[i * column_original + j];
        }
    }
}

// Matrix scalar maltiplication
void mat_mul_const(float *m1, float c, float *sol, int row, int column) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            int index = i * column + j;
            sol[index] = c * m1[index];
        }
    }
}

// Matrix inversion ( by Gaussian elimination)
void mat_inv(float *m, float *sol, int row, int column) {
    // alloc
    float *temp = (float *)malloc(column * 2 * row * sizeof(float));

    // make the augmented matrix
    for (int i = 0; i < column; i++) {
        // copy original matrix
        for (int j = 0; j < row; j++) {
            temp[i * (2 * row) + j] = m[i * row + j];
        }

        // make identity matrix
        for (int j = row; j < row * 2; j++) {
            if (j - row == i) {
                temp[i + (2 * row) + j] = 1;
            } else {
                temp[i + (2 * row) + j] = 0;
            }
        }
    }

    // Sweep (down)
    for (int i = 0; i < column; i++) {
        // pviot selection
        float pivot = temp[i + (2 * row) + i];
        int pivot_index = i;
        float pivot_temp;
        for (int j = i; j < column; j++) {
            if (temp[j * (2 * row) + i] > pivot) {
                pivot = temp[j * (2 * row) + i];
                pivot_index = j;
            }
        }
        if (pivot_index != i) {
            for (int j = 0; j < 2 * row; j++) {
                pivot_temp = temp[pivot_index * (2 * row) + j];
                temp[pivot_index * (2 * row) + j] = temp[i * (2 * row) + j];
                temp[i * (2 * row) + j] = pivot_temp;
            }
        }

        // division
        for (int j = 0; j < 2 * row; j++) {
            temp[i * (2 * row) + j] /= pivot;
        }

        // sweep
        for (int j = i + 1; j < column; j++) {
            float temp2 = temp[j * (2 * row) + i];

            // sweep each row
            for (int k = 0; k < row * 2; k++) {
                temp[j * (2 * row) + k] -= temp2 * temp[i * (2 * row) + k];
            }
        }
    }

    // Sweep (up)
    for (int i = 0; i < column - 1; i++) {
        for (int j = i + 1; j < column; j++) {
            float pivot = temp[(column - 1 - j) * (2 * row) + (row - 1 - i)];
            for (int k = 0; k < 2 * row; k++) {
                temp[(column - 1 - j) * (2 * row) + k] -=
                    pivot * temp[(column - 1 - i) * (2 * row) + k];
            }
        }
    }

    // Copy result
    for (int i = 0; i < column; i++) {
        for (int j = 0; j < row; j++) {
            sol[i * row + j] = temp[i * (2 * row) + (j + row)];
        }
    }
    free(temp);
}

float get_acc_data() {
    M5.IMU.getAccelData(&acc_x, &acc_y, &acc_z);

    float theta_deg = atan(acc_z / (-1 * acc_y)) * 57.29578f;
    // float theta_deg = atan2(acc_z, (-1 * acc_y)) * 57.29578f; //use atan2
    return theta_deg;
}

float get_gyro_data() {
    M5.IMU.getGyroData(&gyro_x, &gyro_y, &gyro_z);
    return gyro_x;
}

void acc_init() {
    float theta_array[sample_num];
    for (int i = 0; i < sample_num; i++) {
        theta_array[i] = get_acc_data();
        delay(meas_interval);
    }

    // calculate mean
    theta_mean = 0;
    for (int i = 0; i < sample_num; i++) {
        theta_mean += theta_array[i];
    }
    theta_mean /= sample_num;

    float temp;
    theta_variance = 0;
    for (int i = 0; i < sample_num; i++) {
        temp = theta_array[i] - theta_mean;
        theta_variance += temp * temp;
    }
    theta_variance /= sample_num;
    Serial.printf("theta_mean %f\n", theta_mean);
    Serial.printf("theta_variance %f\n", theta_variance);
}

void gyro_init() {
    float theta_dot_array[sample_num];
    for (int i = 0; i < sample_num; i++) {
        theta_dot_array[i] = get_gyro_data();
        delay(meas_interval);
    }

    // calculate mean
    theta_dot_mean = 0;
    for (int i = 0; i < sample_num; i++) {
        theta_dot_mean += theta_dot_array[i];
    }
    theta_dot_mean /= sample_num;

    float temp;
    theta_dot_variance = 0;
    for (int i = 0; i < sample_num; i++) {
        temp = theta_dot_array[i] - theta_dot_mean;
        theta_dot_variance += temp * temp;
    }
    theta_dot_variance /= sample_num;
    Serial.printf("theta_dot_mean %f\n", theta_dot_mean);
    Serial.printf("theta_dot_variance %f\n", theta_dot_variance);
}

void IRAM_ATTR onTimer() {
    portENTER_CRITICAL_ISR(&timer_mux);
    is_exec_timer = true;
    last_timer_call = millis();
    portEXIT_CRITICAL_ISR(&timer_mux);
    // Give a semaphore that we can check in the loop
    xSemaphoreGiveFromISR(timer_semaphore, NULL);
}

// Kalman filter for theta and theta_dot_bias
void update_theta() {
    // measurement data
    theta_acc = get_acc_data();

    // input data
    float theta_dot_gyro = get_gyro_data();

#if 0
    static uint32_t cnt = 0;
    static uint32_t last = millis();
    cnt++;
    // if (cnt >= 400) {
    uint32_t now = millis();
    if (now - last >= 1000) {
        // uint32_t diff = millis() - last;
        // Serial.printf("diff %d millis() %d\n", diff, millis());
        Serial.printf("cnt %d millis() %d\n", cnt, now);
        // Serial.printf("th %f dot %f\n", theta_acc, theta_dot_gyro);
        last = now;
        cnt = 0;
    }
#endif

    // calculate Kalman gain: G = P'C^T(W+CP'C^T)^-1
    float P_CT[2][1] = {};
    float tran_C_theta[2][1] = {};
    mat_tran(C_theta[0], tran_C_theta[0], 1, 2);                        // C^T
    mat_mul(P_theta_predict[0], tran_C_theta[0], P_CT[0], 2, 2, 2, 1);  // P'C^T

    float G_temp1[1][1] = {};
    mat_mul(C_theta[0], P_CT[0], G_temp1[0], 1, 2, 2, 1);  // CP'C^T

    float G_temp2 = 1.0f / (G_temp1[0][0] + theta_variance);  //(W+CP'C^T)^-1
    float G[2][1] = {};
    mat_mul_const(P_CT[0], G_temp2, G[0], 2, 1);  // P'C^T(W+CP'C^T)^-1

    // theta_data estimation: theta = theta' + G(y - Ctheta')
    float C_theta_theta[1][1] = {};
    mat_mul(C_theta[0], theta_data_predict[0], C_theta_theta[0], 1, 2, 2,
            1);                                       // Ctheta'
    float delta_y = theta_acc - C_theta_theta[0][0];  // y - Ctheta'
    float delta_theta[2][1] = {};
    mat_mul_const(G[0], delta_y, delta_theta[0], 2, 1);
    mat_add(theta_data_predict[0], delta_theta[0], theta_data[0], 2, 1);

    // calculate covariance matrix: P=(I-GC)P'
    float GC[2][2] = {};
    float I2[2][2] = {{1, 0}, {0, 1}};
    mat_mul(G[0], C_theta[0], GC[0], 2, 1, 1, 2);  // GC
    float I2_GC[2][2] = {};
    mat_sub(I2[0], GC[0], I2_GC[0], 2, 2);                          // I-GC
    mat_mul(I2_GC[0], P_theta_predict[0], P_theta[0], 2, 2, 2, 2);  //((I-GC)P'

    // predict the next step data: theta'
    float A_theta_theta[2][1] = {};
    float B_theta_dot[2][1] = {};
    mat_mul(A_theta[0], theta_data[0], A_theta_theta[0], 2, 2, 2, 1);  // Atheta
    mat_mul_const(B_theta[0], theta_dot_gyro, B_theta_dot[0], 2, 1);   // Bu
    mat_add(A_theta_theta[0], B_theta_dot[0], theta_data_predict[0], 2,
            1);  // Atheta+Bu

    // predict covariance matrix: P'=APA^T + BUB^T
    float AP[2][2] = {};
    float APAT[2][2] = {};
    float tran_A_theta[2][2] = {};
    mat_tran(A_theta[0], tran_A_theta[0], 2, 2);           // A^T
    mat_mul(A_theta[0], P_theta[0], AP[0], 2, 2, 2, 2);    // AP
    mat_mul(AP[0], tran_A_theta[0], APAT[0], 2, 2, 2, 2);  // APA^T

    float BBT[2][2] = {};
    float tran_B_theta[1][2] = {};
    mat_tran(B_theta[0], tran_B_theta[0], 2, 1);               // B^T
    mat_mul(B_theta[0], tran_B_theta[0], BBT[0], 2, 1, 1, 2);  // BB^T

    float BUBT[2][2] = {};
    mat_mul_const(BBT[0], theta_dot_variance, BUBT[0], 2, 2);  // BB^T
    mat_add(APAT[0], BUBT[0], P_theta_predict[0], 2, 2);       // APA^T+BUB^T
}

void setup() {
    Serial.begin(115200);
    // Power ON Stabilizing...
    M5.begin();

    M5.Power.begin();

    M5.IMU.Init();
    delay(500);

    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setTextColor(GREEN);
    M5.Lcd.setTextSize(2);

    // Accelemeter and gyro calibration
    acc_init();
    gyro_init();

    // Kalman filter initialization
    theta_data_predict[0][0] = 0;
    theta_data_predict[1][0] = theta_dot_mean;

    P_theta_predict[0][0] = 1;
    P_theta_predict[0][1] = 0;
    P_theta_predict[1][0] = 0;
    P_theta_predict[1][1] = theta_dot_variance;

    Serial.printf("theta_update_interval %f\n", theta_update_interval);
    // timer1.attach(theta_update_interval, &update_theta);

    // timer
    timer_semaphore = xSemaphoreCreateBinary();
    timer = timerBegin(0, 80, true);
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 2500, true);  // 2.5ms
    timerAlarmEnable(timer);
}

void loop() {
    static uint32_t last_print = millis();
    if (millis() - last_print >= 50) {
        Serial.printf("%5.2f,%5.2f\n", theta_acc, theta_data[0][0]);
        last_print = millis();
    }

    uint32_t timer_millis;
    if (xSemaphoreTake(timer_semaphore, 0) == pdTRUE) {
        portENTER_CRITICAL(&timer_mux);
        is_exec_timer = false;
        timer_millis = last_timer_call;
        portEXIT_CRITICAL(&timer_mux);

        update_theta();
        // Serial.print("onTimer at ");
        // Serial.print(timer_millis);
        // Serial.println(" ms");
    }
#if 0 
    M5.IMU.getGyroData(&gyro_x, &gyro_y, &gyro_z);
    M5.IMU.getAccelData(&acc_x, &acc_y, &acc_z);

    M5.Lcd.fillScreen(BLACK);

    M5.Lcd.setCursor(0, 20);
    M5.Lcd.printf("%6.2f  %6.2f  %6.2f      ", gyro_x, gyro_y, gyro_z);
    M5.Lcd.setCursor(220, 42);
    M5.Lcd.print(" o/s");
    M5.Lcd.setCursor(0, 65);
    M5.Lcd.printf(" %5.2f   %5.2f   %5.2f   ", acc_x, acc_y, acc_z);
    M5.Lcd.setCursor(220, 87);
    M5.Lcd.print(" G");
    M5.Lcd.setCursor(0, 155);
    M5.Lcd.printf("Temperature : %.2f C", temp);

    delay(1);
#endif
}
