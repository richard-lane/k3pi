#ifndef WRAPPER_UTILS_H
#define WRAPPER_UTILS_H

#include <complex>

/*
 * Helpful things for the wrapper functions giving a python-compatible interface for the amplitude models
 *
 */
namespace AmpGenWrapper
{
typedef struct Complex {
    double real{};
    double imag{};
} Complex_t;

extern "C" Complex_t
wrapper(double event[16], const int& kaonCharge, std::complex<double> (*ampFcn)(double const*, const int&))
{
    auto retval{ampFcn(event, kaonCharge)};
    return Complex_t{retval.real(), retval.imag()};
}

/*
 * We might end up allocating memory which won't get freed by the python garbage collector,
 * so provide a function here to explicitly free an array
 *
 * Not sure if its strictly required to define this here
 *
 */
extern "C" void
manual_free(double* ptr){ free(ptr); }

} // namespace AmpGenWrapper

#endif // WRAPPER_UTILS_H
