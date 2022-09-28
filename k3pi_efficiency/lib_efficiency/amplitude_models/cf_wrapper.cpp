#include "wrapper_utils.h"

#include "cf.cpp"

/*
 * Return the CF amplitude of an event as an AmpGenWrapper::Complex_t
 *
 * event:      array of 16 doubles (K_px, K_py, K_py, K_E, ...) etc. for K, pi1, pi2, pi3
 * kaonCharge: charge of the kaon, either +1 or -1
 *
 */
extern "C" AmpGenWrapper::Complex_t cf_wrapper(double event[16], const int kCharge)
{
    // The definition of AMP is provided in cf.cpp
    return AmpGenWrapper::wrapper(event, kCharge, AMP);
}

/*
 * Return the CF amplitudes of a series of events
 *
 * Returns an array of [real imag real imag real imag ...]
 *
 */
extern "C" double* cf_wrapper_array(double const* const kPx,
                                    double const* const kPy,
                                    double const* const kPz,
                                    double const* const kE,
                                    double const* const pi1Px,
                                    double const* const pi1Py,
                                    double const* const pi1Pz,
                                    double const* const pi1E,
                                    double const* const pi2Px,
                                    double const* const pi2Py,
                                    double const* const pi2Pz,
                                    double const* const pi2E,
                                    double const* const pi3Px,
                                    double const* const pi3Py,
                                    double const* const pi3Pz,
                                    double const* const pi3E,
                                    const size_t        nEvts,
                                    const int           kCharge) {
    double* rv = (double*)calloc(2 * nEvts, sizeof(double));

    for (int i=0; i<nEvts; ++i){
        double evt[16] {kPx[i], kPy[i], kPz[i], kE[i],
                       pi1Px[i], pi1Py[i], pi1Pz[i], pi1E[i],
                       pi2Px[i], pi2Py[i], pi2Pz[i], pi2E[i],
                       pi3Px[i], pi3Py[i], pi3Pz[i], pi3E[i]};

        AmpGenWrapper::Complex_t tmp = cf_wrapper(evt, kCharge);

        rv[2*i] = tmp.real;
        rv[2*i + 1] = tmp.imag;
    }

    return rv;
}

