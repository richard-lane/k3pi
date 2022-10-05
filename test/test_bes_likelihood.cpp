/*
 * The BES-III likelihood (as provided to me) took a very long
 * time to run, since it open/read a ROOT file full of covariance
 * matrices every time to evaluate the likelihood.
 *
 * Instead of doing this, I've written a python file that
 * reads the covariance matrix ROOT file and writes a .cpp source
 * file - this source file can be compiled to expose a function
 * that returns a vector of TMatrices (these are the covariance
 * matrices that we need).
 *
 * We can then evaluate the BES-III likelihood by changing the right
 * values in the array of bes parameters; it should be the same
 * for both the ROOT file and hard-coded versions.
 *
 */
#include <cmath>
#include <cassert>
#include <iostream>

// Forward declare because I can't be bothered to write a header file
extern "C" double besChi2(const short, const double, const double, const double, const double);
extern "C" double besChi2Standalone(const short, const double, const double, const double, const double);

int main(void)
{
    const short  binNumber{0};
    const double zRe{0.8};
    const double zIm{0.8};
    const double x{0.003};
    const double y{0.006};

    // Find Chi2 from the "old" interface (by opening the ROOT file)
    const auto theirChi2 = besChi2(binNumber, zRe, zIm, x, y);

    // Find chi2 with the new interface (the compiled cov matrices)
    const auto myChi2 = besChi2Standalone(binNumber, zRe, zIm, x, y);

    std::cout << "value from ROOT file:\t" << theirChi2 << std::endl;
    std::cout << "value from generated C++ file:\t" << myChi2 << " (should be the same)" << std::endl;

    assert(fabs(theirChi2 - myChi2) < 0.00001);
}
