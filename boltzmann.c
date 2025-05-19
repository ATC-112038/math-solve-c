/*
 * Random Linear Algebra Equation Solver with Graphing
 * 
 * This program continuously generates random linear algebra equations,
 * solves them, and graphs the solutions along with their logarithms.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>  // For sleep function
#include <float.h>

// Define constants for the application
#define MAX_MATRIX_SIZE 5
#define GRAPH_WIDTH 60
#define GRAPH_HEIGHT 20
#define SLEEP_SECONDS 0.01

// Matrix operations structures and functions
typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

// Create a new matrix with specified dimensions
Matrix* matrix_create(int rows, int cols) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) {
        fprintf(stderr, "Memory allocation failed for matrix\n");
        exit(1);
    }
    
    mat->rows = rows;
    mat->cols = cols;
    
    // Allocate memory for data
    mat->data = (double**)malloc(rows * sizeof(double*));
    if (!mat->data) {
        fprintf(stderr, "Memory allocation failed for matrix data\n");
        free(mat);
        exit(1);
    }
    
    for (int i = 0; i < rows; i++) {
        mat->data[i] = (double*)calloc(cols, sizeof(double));
        if (!mat->data[i]) {
            fprintf(stderr, "Memory allocation failed for matrix row\n");
            for (int j = 0; j < i; j++) {
                free(mat->data[j]);
            }
            free(mat->data);
            free(mat);
            exit(1);
        }
    }
    
    return mat;
}

// Free matrix memory
void matrix_free(Matrix* mat) {
    if (!mat) return;
    
    for (int i = 0; i < mat->rows; i++) {
        free(mat->data[i]);
    }
    free(mat->data);
    free(mat);
}

// Fill matrix with random values between min and max
void matrix_randomize(Matrix* mat, double min, double max) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i][j] = min + ((double)rand() / RAND_MAX) * (max - min);
        }
    }
}

// Set all elements to zero
void matrix_zero(Matrix* mat) {
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i][j] = 0.0;
        }
    }
}

// Matrix multiplication: result = mat1 * mat2
Matrix* matrix_multiply(Matrix* mat1, Matrix* mat2) {
    if (mat1->cols != mat2->rows) {
        fprintf(stderr, "Matrix dimensions incompatible for multiplication\n");
        return NULL;
    }
    
    Matrix* result = matrix_create(mat1->rows, mat2->cols);
    
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat2->cols; j++) {
            result->data[i][j] = 0.0;
            for (int k = 0; k < mat1->cols; k++) {
                result->data[i][j] += mat1->data[i][k] * mat2->data[k][j];
            }
        }
    }
    
    return result;
}

// Matrix addition: result = mat1 + mat2
Matrix* matrix_add(Matrix* mat1, Matrix* mat2) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        fprintf(stderr, "Matrix dimensions incompatible for addition\n");
        return NULL;
    }
    
    Matrix* result = matrix_create(mat1->rows, mat1->cols);
    
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
        }
    }
    
    return result;
}

// Matrix subtraction: result = mat1 - mat2
Matrix* matrix_subtract(Matrix* mat1, Matrix* mat2) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        fprintf(stderr, "Matrix dimensions incompatible for subtraction\n");
        return NULL;
    }
    
    Matrix* result = matrix_create(mat1->rows, mat1->cols);
    
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols; j++) {
            result->data[i][j] = mat1->data[i][j] - mat2->data[i][j];
        }
    }
    
    return result;
}

// Matrix transpose
Matrix* matrix_transpose(Matrix* mat) {
    Matrix* result = matrix_create(mat->cols, mat->rows);
    
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            result->data[j][i] = mat->data[i][j];
        }
    }
    
    return result;
}

// Print matrix
void matrix_print(Matrix* mat, const char* name) {
    printf("Matrix %s (%d x %d):\n", name, mat->rows, mat->cols);
    for (int i = 0; i < mat->rows; i++) {
        printf("[ ");
        for (int j = 0; j < mat->cols; j++) {
            printf("%6.2f ", mat->data[i][j]);
        }
        printf("]\n");
    }
    printf("\n");
}

// Copy matrix
Matrix* matrix_copy(Matrix* source) {
    Matrix* dest = matrix_create(source->rows, source->cols);
    
    for (int i = 0; i < source->rows; i++) {
        for (int j = 0; j < source->cols; j++) {
            dest->data[i][j] = source->data[i][j];
        }
    }
    
    return dest;
}

// Matrix determinant using Gaussian elimination
double matrix_determinant(Matrix* mat) {
    if (mat->rows != mat->cols) {
        fprintf(stderr, "Error: Determinant can only be calculated for square matrices\n");
        return 0.0;
    }
    
    int n = mat->rows;
    Matrix* m = matrix_copy(mat);
    double det = 1.0;
    
    // Gaussian elimination
    for (int i = 0; i < n; i++) {
        // Find pivot
        int pivot_row = i;
        double pivot_value = fabs(m->data[i][i]);
        
        for (int j = i+1; j < n; j++) {
            if (fabs(m->data[j][i]) > pivot_value) {
                pivot_row = j;
                pivot_value = fabs(m->data[j][i]);
            }
        }
        
        // Swap rows if needed
        if (pivot_row != i) {
            for (int j = i; j < n; j++) {
                double temp = m->data[i][j];
                m->data[i][j] = m->data[pivot_row][j];
                m->data[pivot_row][j] = temp;
            }
            det = -det; // Determinant changes sign with row swap
        }
        
        // If pivot is zero, determinant is zero
        if (fabs(m->data[i][i]) < DBL_EPSILON) {
            matrix_free(m);
            return 0.0;
        }
        
        // Scale determinant by pivot
        det *= m->data[i][i];
        
        // Eliminate below
        for (int j = i+1; j < n; j++) {
            double factor = m->data[j][i] / m->data[i][i];
            for (int k = i; k < n; k++) {
                m->data[j][k] -= factor * m->data[i][k];
            }
        }
    }
    
    matrix_free(m);
    return det;
}

// Matrix inverse using Gauss-Jordan elimination
Matrix* matrix_inverse(Matrix* mat) {
    if (mat->rows != mat->cols) {
        fprintf(stderr, "Error: Inverse can only be calculated for square matrices\n");
        return NULL;
    }
    
    int n = mat->rows;
    Matrix* m = matrix_copy(mat);
    Matrix* inv = matrix_create(n, n);
    
    // Initialize inverse as identity matrix
    for (int i = 0; i < n; i++) {
        inv->data[i][i] = 1.0;
    }
    
    // Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
        // Find pivot
        int pivot_row = i;
        double pivot_value = fabs(m->data[i][i]);
        
        for (int j = i+1; j < n; j++) {
            if (fabs(m->data[j][i]) > pivot_value) {
                pivot_row = j;
                pivot_value = fabs(m->data[j][i]);
            }
        }
        
        // Check if matrix is singular
        if (pivot_value < DBL_EPSILON) {
            fprintf(stderr, "Error: Matrix is singular and cannot be inverted\n");
            matrix_free(m);
            matrix_free(inv);
            return NULL;
        }
        
        // Swap rows if needed
        if (pivot_row != i) {
            for (int j = 0; j < n; j++) {
                double temp = m->data[i][j];
                m->data[i][j] = m->data[pivot_row][j];
                m->data[pivot_row][j] = temp;
                
                temp = inv->data[i][j];
                inv->data[i][j] = inv->data[pivot_row][j];
                inv->data[pivot_row][j] = temp;
            }
        }
        
        // Scale the pivot row
        double pivot = m->data[i][i];
        for (int j = 0; j < n; j++) {
            m->data[i][j] /= pivot;
            inv->data[i][j] /= pivot;
        }
        
        // Eliminate other rows
        for (int j = 0; j < n; j++) {
            if (j != i) {
                double factor = m->data[j][i];
                for (int k = 0; k < n; k++) {
                    m->data[j][k] -= factor * m->data[i][k];
                    inv->data[j][k] -= factor * inv->data[i][k];
                }
            }
        }
    }
    
    matrix_free(m);
    return inv;
}

// Solve linear system Ax = b
Matrix* solve_linear_system(Matrix* A, Matrix* b) {
    if (A->rows != A->cols || A->rows != b->rows || b->cols != 1) {
        fprintf(stderr, "Error: Invalid dimensions for solving linear system\n");
        return NULL;
    }
    
    // Check if matrix is invertible
    double det = matrix_determinant(A);
    if (fabs(det) < DBL_EPSILON) {
        fprintf(stderr, "Error: Matrix is singular, system may have no unique solution\n");
        return NULL;
    }
    
    // Solve using inverse: x = A^-1 * b
    Matrix* A_inv = matrix_inverse(A);
    if (!A_inv) return NULL;
    
    Matrix* x = matrix_multiply(A_inv, b);
    matrix_free(A_inv);
    
    return x;
}

// Generate random linear equation Ax = b
void generate_random_equation(int size, Matrix** A, Matrix** b) {
    // Create coefficient matrix A
    *A = matrix_create(size, size);
    
    // Fill A with random values, but ensure it's non-singular
    do {
        matrix_randomize(*A, -5.0, 5.0);
    } while (fabs(matrix_determinant(*A)) < 0.1); // Ensure determinant is not too close to zero
    
    // Create right-hand side vector b
    *b = matrix_create(size, 1);
    matrix_randomize(*b, -10.0, 10.0);
}

// ASCII bar for graphs
void draw_bar(int length, char symbol) {
    for (int i = 0; i < length; i++) {
        putchar(symbol);
    }
}

// Draw ASCII bar graph of solution vector and its logarithm
void draw_graph(Matrix* x) {
    printf("Solution vector and logarithm graph:\n");
    printf("----------------------------------------------------------\n");
    
    // Find max absolute value for scaling
    double max_abs_val = 0.0;
    for (int i = 0; i < x->rows; i++) {
        double abs_val = fabs(x->data[i][0]);
        if (abs_val > max_abs_val) max_abs_val = abs_val;
    }
    
    // Draw scale
    printf("Scale: Each '=' represents %.2f units\n", max_abs_val / GRAPH_WIDTH);
    printf("       Each '-' represents 0.2 log units\n\n");
    
    // Draw bars for each solution value
    for (int i = 0; i < x->rows; i++) {
        double val = x->data[i][0];
        int bar_length = (int)((fabs(val) / max_abs_val) * GRAPH_WIDTH);
        
        // Draw x[i] bar
        printf("x[%d] = %7.3f |", i, val);
        if (val >= 0) {
            printf(" ");
            draw_bar(bar_length, '=');
        } else {
            draw_bar(bar_length, '=');
            printf(" ");
        }
        printf("|\n");
        
        // Draw log(|x[i]|) bar (if valid)
        if (fabs(val) > 0) {
            double log_val = log10(fabs(val));
            int log_bar_length = (int)((fabs(log_val) + 3) * 10); // Scale for visibility
            
            printf("log|x[%d]|= %7.3f |", i, log_val);
            if (log_val >= 0) {
                printf(" ");
                draw_bar(log_bar_length, '-');
            } else {
                draw_bar(log_bar_length, '-');
                printf(" ");
            }
            printf("|\n");
        } else {
            printf("log|x[%d]|= -inf    (undefined for zero value)\n", i);
        }
        
        printf("\n");
    }
    
    printf("----------------------------------------------------------\n\n");
}

// Function to print the equation in a readable format
void print_equation(Matrix* A, Matrix* b) {
    printf("Linear Equation System:\n");
    
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            printf("%+.2f·x%d ", A->data[i][j], j+1);
        }
        printf("= %.2f\n", b->data[i][0]);
    }
    printf("\n");
}

// Calculate eigenvalues using the power iteration method
void calculate_eigenvalues(Matrix* A, int num_eigenvalues, double* eigenvalues) {
    int n = A->rows;
    if (num_eigenvalues > n) num_eigenvalues = n;
    
    // Create a copy of the matrix
    Matrix* A_copy = matrix_copy(A);
    
    for (int k = 0; k < num_eigenvalues; k++) {
        // Initialize random vector
        Matrix* x = matrix_create(A_copy->cols, 1);
        matrix_randomize(x, -1.0, 1.0);
        
        // Normalize x
        double norm = 0.0;
        for (int i = 0; i < x->rows; i++) {
            norm += x->data[i][0] * x->data[i][0];
        }
        norm = sqrt(norm);
        for (int i = 0; i < x->rows; i++) {
            x->data[i][0] /= norm;
        }
        
        // Power iteration
        double lambda = 0.0;
        Matrix* Ax;
        for (int iter = 0; iter < 100; iter++) {
            Ax = matrix_multiply(A_copy, x);
            
            // Compute Rayleigh quotient
            double new_lambda = 0.0;
            for (int i = 0; i < x->rows; i++) {
                new_lambda += x->data[i][0] * Ax->data[i][0];
            }
            
            // Check convergence
            if (fabs(new_lambda - lambda) < 1e-6) {
                lambda = new_lambda;
                break;
            }
            lambda = new_lambda;
            
            // Normalize Ax to get new x
            norm = 0.0;
            for (int i = 0; i < Ax->rows; i++) {
                norm += Ax->data[i][0] * Ax->data[i][0];
            }
            norm = sqrt(norm);
            
            matrix_free(x);
            x = matrix_create(Ax->rows, 1);
            for (int i = 0; i < Ax->rows; i++) {
                x->data[i][0] = Ax->data[i][0] / norm;
            }
            
            matrix_free(Ax);
        }
        
        // Store eigenvalue
        eigenvalues[k] = lambda;
        
        // Deflate the matrix to find the next eigenvalue
        // A = A - lambda * v * v^T
        Matrix* vT = matrix_transpose(x);
        Matrix* vvT = matrix_multiply(x, vT);
        
        for (int i = 0; i < A_copy->rows; i++) {
            for (int j = 0; j < A_copy->cols; j++) {
                A_copy->data[i][j] -= lambda * vvT->data[i][j];
            }
        }
        
        matrix_free(x);
        matrix_free(vT);
        matrix_free(vvT);
    }
    
    matrix_free(A_copy);
}

// Main function to continuously generate and solve linear equations
int main() {
    // Seed random number generator
    srand(time(NULL));
    
    printf("=== LinLog Calcule ===\n\n");
    printf("lin sys\n");
    printf("grtu\n\n");
    
    // Run continuously until user interrupts
    while (1) {
        // Generate random matrix size (2 to MAX_MATRIX_SIZE)
        int size = 2 + rand() % (MAX_MATRIX_SIZE - 1);
        
        // Generate random equation
        Matrix *A, *b;
        generate_random_equation(size, &A, &b);
        
        // Print the equation
        print_equation(A, b);
        
        // Calculate eigenvalues
        double* eigenvalues = (double*)malloc(size * sizeof(double));
        calculate_eigenvalues(A, size, eigenvalues);
        
        printf("Matrix eigenvalues: ");
        for (int i = 0; i < size; i++) {
            printf("%.2f ", eigenvalues[i]);
        }
        printf("\n\n");
        
        // Calculate determinant
        double det = matrix_determinant(A);
        printf("Matrix determinant: %.4f\n\n", det);
        
        // Solve the system
        Matrix* x = solve_linear_system(A, b);
        
        if (x) {
            // Print solution
            printf("Solution:\n");
            for (int i = 0; i < x->rows; i++) {
                printf("x%d = %f\n", i+1, x->data[i][0]);
            }
            printf("\n");
            
            // Draw graph of solution and logarithm
            draw_graph(x);
            
            // Verify solution by computing A*x
            Matrix* Ax = matrix_multiply(A, x);
            printf("Verification (A*x should equal b):\n");
            matrix_print(b, "b");
            matrix_print(Ax, "A*x");
            
            // Calculate error
            double error = 0.0;
            for (int i = 0; i < b->rows; i++) {
                double diff = b->data[i][0] - Ax->data[i][0];
                error += diff * diff;
            }
            error = sqrt(error);
            printf("Solution error (L2 norm of b - A*x): %e\n\n", error);
            
            matrix_free(Ax);
            matrix_free(x);
        }
        
        matrix_free(A);
        matrix_free(b);
        free(eigenvalues);
        
        printf("====================================================\n");
        printf("Generating next equation in %d seconds...\n", SLEEP_SECONDS);
        printf("====================================================\n\n");
        sleep(SLEEP_SECONDS);
    }
    
    return 0;
}