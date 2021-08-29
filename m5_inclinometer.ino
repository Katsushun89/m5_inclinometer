

// Ticker
Ticker timer1;

// Accelerometer and gyro statistical data
int sample_num = 100;
float meas_interval = 0.01;
float theta_mean;
float theta_variance;
float theta_dot_mean;
float theta_dot_variance;

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
void mat_trans(float *m1, float *sol, int row_original, int column_original) {
    for (int i = 0; i < row_original; i++) {
        for (int j = 0; j < column_original; j++) {
            sol[j * row_original + i] = m1[i * column_original + j];
        }
    }
}

// Matrix scalar maltiplication
void mat_trans(float *m1, float c, float *sol, int row, int column) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            int index = i * column + j;
            sol[index] = c * m1[index];
        }
    }
}

// Matrix inversion ( by Gaussian elimination)
void mat_trans(float *m, float *sol, int row, int column) {
    // alloc
    float *temp = (float *)malloc(column * 2 * row * sizeof(float));

    // make the augmented matrix
    for (int i = 0; i < row; i++) {
        // copy original matrix
        for (int j = 0; j < column; j++) {
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
            sol[i * row + j] = temp[i * (2 * row) + (j * row)];
        }
    }
    free(temp);
}
