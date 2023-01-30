#include <iostream>
#include "test.h"

int main(int argc, char** argv) 
{
    unsigned A_sizes[] = {128, 512, 1024, 2048, 4096};
    unsigned t = 100; // Num of Euler's method steps
    std::cout << "Running tests!" << std::endl;

    for(auto a: A_sizes)
    {
        std::cout << "A = " << a << " t= " << t << std::endl;
        TestCUDA(a, t);
        TestCUBLAS(a, t);
    }

}