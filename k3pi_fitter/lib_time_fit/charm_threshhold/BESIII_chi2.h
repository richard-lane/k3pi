/*
 * BES-III chi^2 from the BES-III D->K3pi analysis
 *
 * Sent to me via email by Yu Zhang (Oxford); Tim Evans and Sneha Malde
 * might also know things about this if you have any questions
 *
 * I've reformatted this file + changed some minor things
 * (vector -> std::vector) so that it compiles nicely without requiring
 * "using namespace std" or anything in the source file that includes this file
 *
 */
#include <TFile.h>
#include <TMath.h>
#include <TMatrixD.h>
#include <TTree.h>
#include <vector>

Double_t total(const Double_t* par, std::vector<TMatrixD*> BESIII_CovMat)
{
    Double_t par0 = par[22];
    Double_t par1 = par[23];
    Double_t par2 = par[24];
    Double_t par3 = par[25];
    Double_t fnorm2_kspipi[64];

    for (int i = 0; i < 8; i++) {
        fnorm2_kspipi[i * 4 + 0] =
            par[11] *
            (par[62 + 15 - i] + par0 * par0 * par[62 + 7 - i] -
             2. * par0 * sqrt(par[62 + 15 - i] * par[62 + 7 - i]) * par[0] *
                 (par[35 - i] * cos(par[4] * TMath::Pi() / 180.) + par[43 - i] * sin(par[4] * TMath::Pi() / 180.)));
        fnorm2_kspipi[i * 4 + 1] =
            par[12] *
            (par[62 + 15 - i] + par1 * par1 * par[62 + 7 - i] -
             2. * par1 * sqrt(par[62 + 15 - i] * par[62 + 7 - i]) * par[1] *
                 (par[35 - i] * cos(par[5] * TMath::Pi() / 180.) + par[43 - i] * sin(par[5] * TMath::Pi() / 180.)));
        fnorm2_kspipi[i * 4 + 2] =
            par[13] *
            (par[62 + 15 - i] + par2 * par2 * par[62 + 7 - i] -
             2. * par2 * sqrt(par[62 + 15 - i] * par[62 + 7 - i]) * par[2] *
                 (par[35 - i] * cos(par[6] * TMath::Pi() / 180.) + par[43 - i] * sin(par[6] * TMath::Pi() / 180.)));
        fnorm2_kspipi[i * 4 + 3] =
            par[14] *
            (par[62 + 15 - i] + par3 * par3 * par[62 + 7 - i] -
             2. * par3 * sqrt(par[62 + 15 - i] * par[62 + 7 - i]) * par[3] *
                 (par[35 - i] * cos(par[7] * TMath::Pi() / 180.) + par[43 - i] * sin(par[7] * TMath::Pi() / 180.)));
    }
    for (int i = 0; i < 8; i++) {
        fnorm2_kspipi[(i + 8) * 4 + 0] =
            par[11] *
            (par[62 + i] + par0 * par0 * par[62 + 8 + i] -
             2. * par0 * sqrt(par[62 + i] * par[62 + 8 + i]) * par[0] *
                 (par[28 + i] * cos(par[4] * TMath::Pi() / 180.) - par[36 + i] * sin(par[4] * TMath::Pi() / 180.)));
        fnorm2_kspipi[(i + 8) * 4 + 1] =
            par[12] *
            (par[62 + i] + par1 * par1 * par[62 + 8 + i] -
             2. * par1 * sqrt(par[62 + i] * par[62 + 8 + i]) * par[1] *
                 (par[28 + i] * cos(par[5] * TMath::Pi() / 180.) - par[36 + i] * sin(par[5] * TMath::Pi() / 180.)));
        fnorm2_kspipi[(i + 8) * 4 + 2] =
            par[13] *
            (par[62 + i] + par2 * par2 * par[62 + 8 + i] -
             2. * par2 * sqrt(par[62 + i] * par[62 + 8 + i]) * par[2] *
                 (par[28 + i] * cos(par[6] * TMath::Pi() / 180.) - par[36 + i] * sin(par[6] * TMath::Pi() / 180.)));
        fnorm2_kspipi[(i + 8) * 4 + 3] =
            par[14] *
            (par[62 + i] + par3 * par3 * par[62 + 8 + i] -
             2. * par3 * sqrt(par[62 + i] * par[62 + 8 + i]) * par[3] *
                 (par[28 + i] * cos(par[7] * TMath::Pi() / 180.) - par[36 + i] * sin(par[7] * TMath::Pi() / 180.)));
    }

    TMatrixD delta_kspipi(1, 64, fnorm2_kspipi);
    TMatrixD Mrho_kspipi_exp(delta_kspipi - (*BESIII_CovMat.at(10)));
    TMatrixD Trho_kspipi_exp(TMatrixD::kTransposed, Mrho_kspipi_exp);

    TMatrixD fnorm3_kspipi1(64, 64);
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            if (i == j) {
                fnorm3_kspipi1(i, j) = 0.;
            } else {
                fnorm3_kspipi1(i, j) = 0.;
            }
        }
    }
    TMatrixD COVS_kspipi_((*BESIII_CovMat.at(11)) + fnorm3_kspipi1);
    TMatrixD TCOVS_kspipi(TMatrixD::kInverted, COVS_kspipi_);
    TMatrixD _Mkspipi(Mrho_kspipi_exp, TMatrixD::kMult, TCOVS_kspipi);
    TMatrixD Mkspipi(_Mkspipi, TMatrixD::kMult, Trho_kspipi_exp);
    Double_t chi2_kspipi = Mkspipi(0, 0);

    Double_t fnorm2_kspipi2[16];
    for (int i = 0; i < 8; i++) {
        fnorm2_kspipi2[i] =
            par[61] *
            (par[62 + 15 - i] + par[8] * par[8] * par[62 + 7 - i] -
             2. * par[8] * sqrt(par[62 + 15 - i] * par[62 + 7 - i]) * par[9] *
                 (par[35 - i] * cos(par[10] * TMath::Pi() / 180.) + par[43 - i] * sin(par[10] * TMath::Pi() / 180.)));
        // c1 :par[17], s1:par[25]
    }
    for (int i = 0; i < 8; i++) {
        fnorm2_kspipi2[(i + 8)] =
            par[61] *
            (par[62 + i] + par[8] * par[8] * par[62 + 8 + i] -
             2. * par[8] * sqrt(par[62 + i] * par[62 + 8 + i]) * par[9] *
                 (par[28 + i] * cos(par[10] * TMath::Pi() / 180.) - par[36 + i] * sin(par[10] * TMath::Pi() / 180.)));
    }

    TMatrixD delta_kspipi2(1, 16, fnorm2_kspipi2);
    TMatrixD Mrho_kspipi2_exp(delta_kspipi2 - (*BESIII_CovMat.at(8)));
    TMatrixD Trho_kspipi2_exp(TMatrixD::kTransposed, Mrho_kspipi2_exp);
    TMatrixD _Mkspipi2(Mrho_kspipi2_exp, TMatrixD::kMult, (*BESIII_CovMat.at(9)));
    TMatrixD Mkspipi2(_Mkspipi2, TMatrixD::kMult, Trho_kspipi2_exp);
    Double_t chi2_kspipi2 = Mkspipi2(0, 0);

    Double_t fnorm2_kpi_CP[56];
    TMatrixD fnorm3_kpi_CP(56, 56);
    for (Int_t i = 6; i < 14; i++) {
        fnorm2_kpi_CP[i * 4 + 0] =
            ((1 + 2. * par0 * par[0] * cos(par[4] * TMath::Pi() / 180.) + par0 * par0)) *
            (par[18] * (par[20] * 0.240971 + 0.243827) /
             (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                         par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21])))) *
            (1 + par[21]) / (1 + 2. * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
        fnorm2_kpi_CP[i * 4 + 1] =
            ((1 + 2. * par1 * par[1] * cos(par[5] * TMath::Pi() / 180.) + par1 * par1)) *
            (par[18] * (par[20] * 0.242867 + 0.209245) /
             (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                         par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21])))) *
            (1 + par[21]) / (1 + 2. * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
        fnorm2_kpi_CP[i * 4 + 2] =
            ((1 + 2. * par2 * par[2] * cos(par[6] * TMath::Pi() / 180.) + par2 * par2)) *
            (par[18] * (par[20] * 0.246835 + 0.218068) /
             (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                         par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21])))) *
            (1 + par[21]) / (1 + 2. * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
        fnorm2_kpi_CP[i * 4 + 3] =
            ((1 + 2. * par3 * par[3] * cos(par[7] * TMath::Pi() / 180.) + par3 * par3)) *
            (par[18] * (par[20] * 0.216858 + 0.279841) /
             (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                         par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21])))) *
            (1 + par[21]) / (1 + 2. * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
        for (Int_t ii = i * 4 + 0; ii < i * 4 + 4; ii++) {
            for (Int_t jj = i * 4 + 0; jj < i * 4 + 4; jj++) {
                fnorm3_kpi_CP(ii, jj) = fnorm2_kpi_CP[ii] * fnorm2_kpi_CP[jj] *
                                        ((*BESIII_CovMat.at(1))(0, i) * (*BESIII_CovMat.at(1))(0, i) +
                                         (*BESIII_CovMat.at(20))(0, i) * (*BESIII_CovMat.at(20))(0, i) +
                                         (*BESIII_CovMat.at(21))(0, i) * (*BESIII_CovMat.at(21))(0, i) +
                                         0.005 * 0.005 * 6 + 0.005 * 0.005 * 2 + 0.015 * 0.015);
            }
        }
    }
    for (Int_t i = 0; i < 6; i++) {
        fnorm2_kpi_CP[i * 4 + 0] =
            ((1 - 2. * par0 * par[0] * cos(par[4] * TMath::Pi() / 180.) + par0 * par0)) *
            (par[18] * (par[20] * 0.240971 + 0.243827) /
             (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                         par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21])))) *
            (1 + par[21]) / (1 - 2. * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
        fnorm2_kpi_CP[i * 4 + 1] =
            ((1 - 2. * par1 * par[1] * cos(par[5] * TMath::Pi() / 180.) + par1 * par1)) *
            (par[18] * (par[20] * 0.242867 + 0.209245) /
             (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                         par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21])))) *
            (1 + par[21]) / (1 - 2. * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
        fnorm2_kpi_CP[i * 4 + 2] =
            ((1 - 2. * par2 * par[2] * cos(par[6] * TMath::Pi() / 180.) + par2 * par2)) *
            (par[18] * (par[20] * 0.246835 + 0.218068) /
             (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                         par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21])))) *
            (1 + par[21]) / (1 - 2. * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
        fnorm2_kpi_CP[i * 4 + 3] =
            ((1 - 2. * par3 * par[3] * cos(par[7] * TMath::Pi() / 180.) + par3 * par3)) *
            (par[18] * (par[20] * 0.216858 + 0.279841) /
             (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                         par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21])))) *
            (1 + par[21]) / (1 - 2. * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
        if (i >= 5) {
            fnorm2_kpi_CP[i * 4 + 0] =
                ((1 - 2. * (2. * par[60] - 1.) * par0 * par[0] * cos(par[4] * TMath::Pi() / 180.) + par0 * par0)) *
                (par[18] * (par[20] * 0.240971 + 0.243827) /
                 (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                             par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21])))) *
                (1 + par[21]) /
                (1 - 2. * (2. * par[60] - 1.) * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
            fnorm2_kpi_CP[i * 4 + 1] =
                ((1 - 2. * (2. * par[60] - 1.) * par1 * par[1] * cos(par[5] * TMath::Pi() / 180.) + par1 * par1)) *
                (par[18] * (par[20] * 0.242867 + 0.209245) /
                 (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                             par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21])))) *
                (1 + par[21]) /
                (1 - 2. * (2. * par[60] - 1.) * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
            fnorm2_kpi_CP[i * 4 + 2] =
                ((1 - 2. * (2. * par[60] - 1.) * par2 * par[2] * cos(par[6] * TMath::Pi() / 180.) + par2 * par2)) *
                (par[18] * (par[20] * 0.246835 + 0.218068) /
                 (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                             par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21])))) *
                (1 + par[21]) /
                (1 - 2. * (2. * par[60] - 1.) * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
            fnorm2_kpi_CP[i * 4 + 3] =
                ((1 - 2. * (2. * par[60] - 1.) * par3 * par[3] * cos(par[7] * TMath::Pi() / 180.) + par3 * par3)) *
                (par[18] * (par[20] * 0.216858 + 0.279841) /
                 (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                             par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21])))) *
                (1 + par[21]) /
                (1 - 2. * (2. * par[60] - 1.) * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
        }

        for (Int_t ii = i * 4 + 0; ii < i * 4 + 4; ii++) {
            for (Int_t jj = i * 4 + 0; jj < i * 4 + 4; jj++) {
                fnorm3_kpi_CP(ii, jj) = fnorm2_kpi_CP[ii] * fnorm2_kpi_CP[jj] *
                                        ((*BESIII_CovMat.at(1))(0, i) * (*BESIII_CovMat.at(1))(0, i) +
                                         (*BESIII_CovMat.at(20))(0, i) * (*BESIII_CovMat.at(20))(0, i) +
                                         (*BESIII_CovMat.at(21))(0, i) * (*BESIII_CovMat.at(21))(0, i) +
                                         0.005 * 0.005 * 6 + 0.005 * 0.005 * 2 + 0.015 * 0.015);
            }
        }
    }

    TMatrixD delta_CP(1, 56, fnorm2_kpi_CP);
    TMatrixD _Sk3pi(TMatrixD::kTransposed, (*BESIII_CovMat.at(2)));
    TMatrixD Mrho_CP_exp(delta_CP - (*BESIII_CovMat.at(2)));
    TMatrixD Trho_CP_exp(TMatrixD::kTransposed, Mrho_CP_exp);
    TMatrixD CPCOV_(fnorm3_kpi_CP + (*BESIII_CovMat.at(0)));
    TMatrixD TCPCOV(TMatrixD::kInverted, CPCOV_);

    TMatrixD _MCP(Mrho_CP_exp, TMatrixD::kMult, TCPCOV);
    TMatrixD MCP(_MCP, TMatrixD::kMult, Trho_CP_exp);
    Double_t chi2_CP = MCP(0, 0);

    Double_t fnorm2_kpi2_CP[14];
    TMatrixD fnorm3_kpi2_CP(14, 14);
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            fnorm3_kpi2_CP(i, j) = 0.;
        }
    }
    for (Int_t i = 6; i < 14; i++) {
        fnorm2_kpi2_CP[i] =
            ((1 + 2. * par[8] * par[9] * cos(par[10] * TMath::Pi() / 180.) + par[8] * par[8]) / (1 + par[27])) *
            (par[17] * (1 + par[19]) /
             (par[16] * (1 + (par[21]) - par[27] * cos(par[15] * TMath::Pi() / 180) * sqrt(par[21]) +
                         par[26] * sin(par[15] * TMath::Pi() / 180) * sqrt(par[21]) +
                         (par[26] * par[26] + par[27] * par[27]) / 2.))) *
            (1 + par[27]) / (1 + 2. * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + (par[21]));
        fnorm3_kpi2_CP(i, i) =
            fnorm2_kpi2_CP[i] * fnorm2_kpi2_CP[i] * (*BESIII_CovMat.at(1))(0, i) * (*BESIII_CovMat.at(1))(0, i) +
            0.005 * 0.005 * 4 + 0.01 * 0.01 * 1 + 0.005 * 0.005 * 2 + 0.008 * 0.008;
    }
    for (Int_t i = 0; i < 6; i++) {
        if (i < 5) {
            fnorm2_kpi2_CP[i] =
                ((1 - 2. * par[8] * par[9] * cos(par[10] * TMath::Pi() / 180.) + par[8] * par[8]) / (1 - par[27])) *
                (par[17] * (1 + par[19]) /
                 (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180) * sqrt(par[21]) +
                             par[26] * sin(par[15] * TMath::Pi() / 180) * sqrt(par[21]) +
                             (par[26] * par[26] + par[27] * par[27]) / 2.))) *
                (1 - par[27]) / (1 - 2. * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
        } else {
            fnorm2_kpi2_CP[i] =
                ((1 - 2. * (2. * par[60] - 1) * par[8] * par[9] * cos(par[10] * TMath::Pi() / 180.) + par[8] * par[8]) /
                 (1 - par[27] * (2. * par[60] - 1))) *
                (par[17] * (1 + par[19]) /
                 (par[16] * (1 + par[21] - par[27] * cos(par[15] * TMath::Pi() / 180) * sqrt(par[21]) +
                             par[26] * sin(par[15] * TMath::Pi() / 180) * sqrt(par[21]) +
                             (par[26] * par[26] + par[27] * par[27]) / 2.))) *
                (1 - par[27] * (2. * par[60] - 1)) /
                (1 - 2. * (2. * par[60] - 1) * sqrt(par[21]) * cos(par[15] * TMath::Pi() / 180.) + par[21]);
        }
        fnorm3_kpi2_CP(i, i) =
            fnorm2_kpi2_CP[i] * fnorm2_kpi2_CP[i] * (*BESIII_CovMat.at(1))(0, i) * (*BESIII_CovMat.at(1))(0, i) +
            0.005 * 0.005 * 4 + 0.01 * 0.01 * 1 + 0.005 * 0.005 * 2 + 0.008 * 0.008;
    }
    TMatrixD delta_CP2(1, 14, fnorm2_kpi2_CP);
    TMatrixD _Sk3pi2(TMatrixD::kTransposed, (*BESIII_CovMat.at(5)));
    TMatrixD Mrho_CP2_exp(delta_CP2 - (*BESIII_CovMat.at(5)));
    TMatrixD Trho_CP2_exp(TMatrixD::kTransposed, Mrho_CP2_exp);
    TMatrixD COV2_(fnorm3_kpi2_CP + (*BESIII_CovMat.at(22)));
    TMatrixD TCOV2(TMatrixD::kInverted, COV2_);
    TMatrixD _MCP2(Mrho_CP2_exp, TMatrixD::kMult, TCOV2);
    TMatrixD MCP2(_MCP2, TMatrixD::kMult, Trho_CP2_exp);
    Double_t chi2_CP2 = MCP2(0, 0);

    Double_t fnorm2_LS[18];
    Double_t rhop[18];
    rhop[0] =
        (1 - 2. * (par0 / sqrt(par[21])) * par[0] * cos(par[15] * TMath::Pi() / 180. - par[4] * TMath::Pi() / 180.) +
         (par0 * par0 / par[21])) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par[21])) -
         (par[27] * cos(par[15] * TMath::Pi() / 180.) - par[26] * sin(par[4] * TMath::Pi() / 180.)) / sqrt(par[21]));
    rhop[1] =
        (1 - 2. * (par1 / sqrt(par[21])) * par[1] * cos(par[15] * TMath::Pi() / 180. - par[5] * TMath::Pi() / 180.) +
         (par1 * par1 / par[21])) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par[21])) -
         (par[27] * cos(par[15] * TMath::Pi() / 180.) - par[26] * sin(par[5] * TMath::Pi() / 180.)) / sqrt(par[21]));
    rhop[2] =
        (1 - 2. * (par2 / sqrt(par[21])) * par[2] * cos(par[15] * TMath::Pi() / 180. - par[6] * TMath::Pi() / 180.) +
         (par2 * par2 / par[21])) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par[21])) -
         (par[27] * cos(par[15] * TMath::Pi() / 180.) - par[26] * sin(par[6] * TMath::Pi() / 180.)) / sqrt(par[21]));
    rhop[3] =
        (1 - 2. * (par3 / sqrt(par[21])) * par[3] * cos(par[15] * TMath::Pi() / 180. - par[7] * TMath::Pi() / 180.) +
         (par3 * par3 / par[21])) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par[21])) -
         (par[27] * cos(par[15] * TMath::Pi() / 180.) - par[26] * sin(par[7] * TMath::Pi() / 180.)) / sqrt(par[21]));

    rhop[4] =
        (1 - 2. * (par[8] / par0) * par[9] * par[0] * cos((par[10] - par[4]) * TMath::Pi() / 180.) +
         par[8] * par[8] / (par0 * par0)) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par0 * par0)) -
         (par[0] / par0) * (par[27] * cos(par[4] * TMath::Pi() / 180.) - par[26] * sin(par[4] * TMath::Pi() / 180.)));
    rhop[5] =
        (1 - 2. * (par[8] / par1) * par[9] * par[1] * cos((par[10] - par[5]) * TMath::Pi() / 180.) +
         par[8] * par[8] / (par1 * par1)) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par1 * par1)) -
         (par[1] / par1) * (par[27] * cos(par[5] * TMath::Pi() / 180.) - par[26] * sin(par[5] * TMath::Pi() / 180.)));
    rhop[6] =
        (1 - 2. * (par[8] / par2) * par[9] * par[2] * cos((par[10] - par[6]) * TMath::Pi() / 180.) +
         par[8] * par[8] / (par2 * par2)) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par2 * par2)) -
         (par[2] / par2) * (par[27] * cos(par[6] * TMath::Pi() / 180.) - par[26] * sin(par[6] * TMath::Pi() / 180.)));
    rhop[7] =
        (1 - 2. * (par[8] / par3) * par[9] * par[3] * cos((par[10] - par[7]) * TMath::Pi() / 180.) +
         par[8] * par[8] / (par3 * par3)) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par3 * par3)) -
         (par[3] / par3) * (par[27] * cos(par[7] * TMath::Pi() / 180.) - par[26] * sin(par[7] * TMath::Pi() / 180.)));

    rhop[8] =
        (1 - par[0] * par[0]) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * par0 * par0) -
         (par[0] / par0) * (par[27] * cos(par[4] * TMath::Pi() / 180.) - par[26] * sin(par[4] * TMath::Pi() / 180.)));
    rhop[9] =
        (1 - par[1] * par[1]) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * par1 * par1) -
         (par[1] / par1) * (par[27] * cos(par[5] * TMath::Pi() / 180.) - par[26] * sin(par[5] * TMath::Pi() / 180.)));
    rhop[10] =
        (1 - par[2] * par[2]) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * par2 * par2) -
         (par[2] / par2) * (par[27] * cos(par[6] * TMath::Pi() / 180.) - par[26] * sin(par[6] * TMath::Pi() / 180.)));
    rhop[11] =
        (1 - par[3] * par[3]) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * par3 * par3) -
         (par[3] / par3) * (par[27] * cos(par[7] * TMath::Pi() / 180.) - par[26] * sin(par[7] * TMath::Pi() / 180.)));

    rhop[12] =
        (1 - 2. * (par1 / par0) * par[0] * par[1] * cos((par[5] - par[4]) * TMath::Pi() / 180.) +
         par1 * par1 / (par0 * par0)) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par0 * par0)) -
         (par[0] / par0) * (par[27] * cos(par[4] * TMath::Pi() / 180.) - par[26] * sin(par[4] * TMath::Pi() / 180.)));
    rhop[13] =
        (1 - 2. * (par2 / par0) * par[0] * par[2] * cos((par[6] - par[4]) * TMath::Pi() / 180.) +
         par2 * par2 / (par0 * par0)) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par0 * par0)) -
         (par[0] / par0) * (par[27] * cos(par[4] * TMath::Pi() / 180.) - par[26] * sin(par[4] * TMath::Pi() / 180.)));
    rhop[14] =
        (1 - 2. * (par3 / par0) * par[0] * par[3] * cos((par[7] - par[4]) * TMath::Pi() / 180.) +
         par3 * par3 / (par0 * par0)) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par0 * par0)) -
         (par[0] / par0) * (par[27] * cos(par[4] * TMath::Pi() / 180.) - par[26] * sin(par[4] * TMath::Pi() / 180.)));
    rhop[15] =
        (1 - 2. * (par2 / par1) * par[1] * par[2] * cos((par[6] - par[5]) * TMath::Pi() / 180.) +
         par2 * par2 / (par1 * par1)) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par1 * par1)) -
         (par[1] / par1) * (par[27] * cos(par[5] * TMath::Pi() / 180.) - par[26] * sin(par[5] * TMath::Pi() / 180.)));
    rhop[16] =
        (1 - 2. * (par3 / par1) * par[1] * par[3] * cos((par[7] - par[5]) * TMath::Pi() / 180.) +
         par3 * par3 / (par1 * par1)) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par1 * par1)) -
         (par[1] / par1) * (par[27] * cos(par[5] * TMath::Pi() / 180.) - par[26] * sin(par[5] * TMath::Pi() / 180.)));
    rhop[17] =
        (1 - 2. * (par3 / par2) * par[2] * par[3] * cos((par[7] - par[6]) * TMath::Pi() / 180.) +
         par3 * par3 / (par2 * par2)) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par2 * par2)) -
         (par[2] / par2) * (par[27] * cos(par[6] * TMath::Pi() / 180.) - par[26] * sin(par[6] * TMath::Pi() / 180.)));

    fnorm2_LS[0] =
        rhop[0] *
        (0.240971 * par[20] + 0.243827 * (par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                                          par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21]))) /
        (1 + (0.240971 / 0.243827) * par[20] /
                 ((par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                   par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21]))));
    fnorm2_LS[1] =
        rhop[1] *
        (0.242867 * par[20] + 0.209245 * (par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                                          par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21]))) /
        (1 + (0.242867 / 0.209245) * par[20] /
                 ((par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                   par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21]))));
    fnorm2_LS[2] =
        rhop[2] *
        (0.246835 * par[20] + 0.218068 * (par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                                          par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21]))) /
        (1 + (0.246835 / 0.218068) * par[20] /
                 ((par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                   par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21]))));
    fnorm2_LS[3] =
        rhop[3] *
        (0.216858 * par[20] + 0.279841 * (par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                                          par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21]))) /
        (1 + (0.216858 / 0.279841) * par[20] /
                 ((par[21] - par[27] * cos(par[15] * TMath::Pi() / 180.) * sqrt(par[21]) +
                   par[26] * sin(par[15] * TMath::Pi() / 180.) * sqrt(par[21]))));

    fnorm2_LS[4] =
        rhop[4] * (0.240971 * par[20] + 0.243827 * par[19]) / (1 + par[19] / ((0.240971 / 0.243827) * (par[20])));
    fnorm2_LS[5] =
        rhop[5] * (0.242867 * par[20] + 0.209245 * par[19]) / (1 + par[19] / ((0.242867 / 0.209245) * (par[20])));
    fnorm2_LS[6] =
        rhop[6] * (0.246835 * par[20] + 0.218068 * par[19]) / (1 + par[19] / ((0.246835 / 0.218068) * (par[20])));
    fnorm2_LS[7] =
        rhop[7] * (0.216858 * par[20] + 0.279841 * par[19]) / (1 + par[19] / ((0.216858 / 0.279841) * (par[20])));

    fnorm2_LS[8]  = 2. * rhop[8] * (0.240971 * par[20] * 0.243827);
    fnorm2_LS[9]  = 2. * rhop[9] * (0.242867 * par[20] * 0.209245);
    fnorm2_LS[10] = 2. * rhop[10] * (0.246835 * par[20] * 0.218068);
    fnorm2_LS[11] = 2. * rhop[11] * (0.216858 * par[20] * 0.279841);

    fnorm2_LS[12] = 2. * rhop[12] * ((0.240971 * 0.209245 + 0.242867 * 0.243827) * par[20]) /
                    (1 + 0.242867 * 0.243827 / (0.209245 * 0.240971));
    fnorm2_LS[13] = 2. * rhop[13] * ((0.240971 * 0.218068 + 0.246835 * 0.243827) * par[20]) /
                    (1 + 0.246835 * 0.243827 / (0.218068 * 0.240971));
    fnorm2_LS[14] = 2. * rhop[14] * ((0.240971 * 0.279841 + 0.216858 * 0.243827) * par[20]) /
                    (1 + 0.216858 * 0.243827 / (0.279841 * 0.240971));
    fnorm2_LS[15] = 2. * rhop[15] * ((0.242867 * 0.218068 + 0.246835 * 0.209245) * par[20]) /
                    (1 + 0.246835 * 0.209245 / (0.218068 * 0.242867));
    fnorm2_LS[16] = 2. * rhop[16] * ((0.242867 * 0.279841 + 0.216858 * 0.209245) * par[20]) /
                    (1 + 0.216858 * 0.209245 / (0.279841 * 0.242867));
    fnorm2_LS[17] = 2. * rhop[17] * ((0.246835 * 0.279841 + 0.216858 * 0.218068) * par[20]) /
                    (1 + 0.216858 * 0.218068 / (0.279841 * 0.246835));

    TMatrixD fnorm3_LS(18, 18);
    for (Int_t ii = 0; ii < 4; ii++) {
        for (Int_t jj = 0; jj < 4; jj++) {
            fnorm3_LS(ii, jj) = fnorm2_LS[ii] * fnorm2_LS[jj] *
                                ((*BESIII_CovMat.at(4))(0, 0) * (*BESIII_CovMat.at(4))(0, 0) + 0.02 * 0.02 * 1);
        }
    }
    for (Int_t ii = 4 + 0; ii < 4 + 4; ii++) {
        for (Int_t jj = 4 + 0; jj < 4 + 4; jj++) {
            fnorm3_LS(ii, jj) = fnorm2_LS[ii] * fnorm2_LS[jj] *
                                ((*BESIII_CovMat.at(4))(0, 4) * (*BESIII_CovMat.at(4))(0, 4) + 0.02 * 0.02 * 1);
        }
    }
    for (Int_t ii = 4 * 2 + 0; ii < 4 * 2 + 10; ii++) {
        for (Int_t jj = 4 * 2 + 0; jj < 4 * 2 + 4; jj++) {
            fnorm3_LS(ii, jj) = fnorm2_LS[ii] * fnorm2_LS[jj] *
                                ((*BESIII_CovMat.at(4))(0, 8) * (*BESIII_CovMat.at(4))(0, 8) + 0.03 * 0.03 * 1);
        }
    }

    TMatrixD delta_LS(1, 18, fnorm2_LS);
    TMatrixD _SLS(TMatrixD::kTransposed, (*BESIII_CovMat.at(23)));
    TMatrixD Mrho_LS_exp(delta_LS - (*BESIII_CovMat.at(23)));
    TMatrixD Trho_LS_exp(TMatrixD::kTransposed, Mrho_LS_exp);
    TMatrixD LSCOV_(fnorm3_LS + (*BESIII_CovMat.at(3)));
    TMatrixD TLSCOV(TMatrixD::kInverted, (LSCOV_));

    TMatrixD _MLS(Mrho_LS_exp, TMatrixD::kMult, TLSCOV);
    TMatrixD MLS(_MLS, TMatrixD::kMult, Trho_LS_exp);
    Double_t chi2_LS = MLS(0, 0);

    Double_t fnorm2_LSp[2];
    Double_t rhopp[2];
    rhopp[0] =
        (1 - 2. * (par[8] / sqrt(par[21])) * par[9] * cos(par[15] * TMath::Pi() / 180. - par[10] * TMath::Pi() / 180.) +
         (par[8] * par[8] / par[21])) /
        (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * (par[21])) -
         (par[27] * cos(par[15] * TMath::Pi() / 180.) - par[26] * sin(par[15] * TMath::Pi() / 180.)) / sqrt(par[21]));
    rhopp[1] = (1 - par[9] * par[9]) / (1 + (par[26] * par[26] + par[27] * par[27]) / (2. * par[8] * par[8]) -
                                        (par[9] / par[8]) * (par[27] * cos(par[10] * TMath::Pi() / 180.) -
                                                             par[26] * sin(par[10] * TMath::Pi() / 180.)));

    fnorm2_LSp[0] = rhopp[0] *
                    (par[19] + (par[21] - par[27] * cos(par[15] * TMath::Pi() / 180) * sqrt(par[21]) +
                                par[26] * sin(par[15] * TMath::Pi() / 180) * sqrt(par[21]) +
                                (par[26] * par[26] + par[27] * par[27]) / 2.)) /
                    (1 + (par[19] / ((((par[21]) - par[27] * cos(par[15] * TMath::Pi() / 180) * sqrt(par[21]) +
                                       par[26] * sin(par[15] * TMath::Pi() / 180) * sqrt(par[21]) +
                                       (par[26] * par[26] + par[27] * par[27]) / 2.)))));
    fnorm2_LSp[1] = 2. * rhopp[1] * (par[19]);

    TMatrixD fnorm3_LSp(2, 2);
    for (Int_t ii = 0; ii < 2; ii++) {
        for (Int_t jj = 0; jj < 2; jj++) {
            if (ii == jj) {
                fnorm3_LSp(ii, jj) =
                    fnorm2_LSp[ii] * fnorm2_LSp[jj] *
                    ((*BESIII_CovMat.at(7))(0, jj) * (*BESIII_CovMat.at(7))(0, ii) + 0.035 * 0.035 * 1);
            } else {
                fnorm3_LSp(ii, jj) = 0.;
            }
        }
    }

    TMatrixD delta_LSp(1, 2, fnorm2_LSp);
    TMatrixD _SLSp(TMatrixD::kTransposed, (*BESIII_CovMat.at(24)));
    TMatrixD Mrho_LS_expp(delta_LSp - (*BESIII_CovMat.at(24)));
    TMatrixD Trho_LS_expp(TMatrixD::kTransposed, Mrho_LS_expp);
    TMatrixD LSCOVp_(fnorm3_LSp + (*BESIII_CovMat.at(6)));
    TMatrixD TLSCOVp(TMatrixD::kInverted, (LSCOVp_));

    TMatrixD _MLSp(Mrho_LS_expp, TMatrixD::kMult, TLSCOVp);
    TMatrixD MLSp(_MLSp, TMatrixD::kMult, Trho_LS_expp);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (i != j)
                continue;
        }
    }
    Double_t chi2_LSp = MLSp(0, 0);

    Double_t chi2 = chi2_LS + chi2_CP + chi2_kspipi + chi2_CP2 + chi2_kspipi2 + chi2_LSp;

    chi2 += (par[8] * par[8] - par[27] * par[9] * cos(par[10] * TMath::Pi() / 180) * par[8] +
             par[26] * par[9] * sin(par[10] * TMath::Pi() / 180) * par[8] +
             (par[26] * par[26] + par[27] * par[27]) / 2. - par[19]) *
            (par[8] * par[8] - par[27] * par[9] * cos(par[10] * TMath::Pi() / 180) * par[8] +
             par[26] * par[9] * sin(par[10] * TMath::Pi() / 180) * par[8] +
             (par[26] * par[26] + par[27] * par[27]) / 2. - par[19]) /
            (0.00007 * 0.00007);
    chi2 += (par[22] * par[22] - par[27] * par[0] * cos(par[4] * TMath::Pi() / 180) * par[22] +
             par[26] * par[0] * sin(par[4] * TMath::Pi() / 180) * par[22] - par[20] * 0.240971 / 0.243827) *
            (par[22] * par[22] - par[27] * par[0] * cos(par[4] * TMath::Pi() / 180) * par[22] +
             par[26] * par[0] * sin(par[4] * TMath::Pi() / 180) * par[22] - par[20] * 0.240971 / 0.243827) /
            (0.00011 * 0.00011);
    chi2 += (par[23] * par[23] - par[27] * par[1] * cos(par[5] * TMath::Pi() / 180) * par[23] +
             par[26] * par[1] * sin(par[5] * TMath::Pi() / 180) * par[23] - par[20] * 0.242867 / 0.209245) *
            (par[23] * par[23] - par[27] * par[1] * cos(par[5] * TMath::Pi() / 180) * par[23] +
             par[26] * par[1] * sin(par[5] * TMath::Pi() / 180) * par[22] - par[20] * 0.242867 / 0.209245) /
            (0.00011 * 0.00011);
    chi2 += (par[24] * par[24] - par[27] * par[2] * cos(par[6] * TMath::Pi() / 180) * par[24] +
             par[26] * par[2] * sin(par[6] * TMath::Pi() / 180) * par[24] - par[20] * 0.246835 / 0.218068) *
            (par[24] * par[24] - par[27] * par[2] * cos(par[6] * TMath::Pi() / 180) * par[24] +
             par[26] * par[2] * sin(par[6] * TMath::Pi() / 180) * par[24] - par[20] * 0.246835 / 0.218068) /
            (0.00011 * 0.00011);
    chi2 += (par[25] * par[25] - par[27] * par[3] * cos(par[7] * TMath::Pi() / 180) * par[25] +
             par[26] * par[3] * sin(par[7] * TMath::Pi() / 180) * par[25] - par[20] * 0.216858 / 0.279841) *
            (par[25] * par[25] - par[27] * par[3] * cos(par[7] * TMath::Pi() / 180) * par[25] +
             par[26] * par[3] * sin(par[7] * TMath::Pi() / 180) * par[25] - par[20] * 0.216858 / 0.279841) /
            (0.00011 * 0.00011);

    TMatrixD ciExp(1, 16);
    for (int ici = 0; ici < 16; ici++) {
        ciExp(0, ici) = par[28 + ici];
    }
    TMatrixD kiExp(1, 16);
    TMatrixD kip_Exp(1, 16);
    for (int iki = 0; iki < 16; iki++) {
        kiExp(0, iki) = par[44 + iki];
        if (iki < 8) {
            kip_Exp(0, iki) = par[62 + iki] - sqrt(par[62 + iki] * par[iki + 8 + 62]) *
                                                  (par[27] * par[44 + iki] - par[26] * par[52 + iki]);
        } else {
            kip_Exp(0, iki) = par[62 + iki] - sqrt(par[62 + iki] * par[iki - 8 + 62]) *
                                                  (par[27] * par[44 + iki - 8] + par[26] * par[52 + iki - 8]);
        }
    }
    TMatrixD BrExp(1, 5);
    for (int iki = 0; iki < 5; iki++) {
        BrExp(0, iki) = par[16 + iki];
    }
    TMatrixD deltakpiExp(1, 4);
    deltakpiExp(0, 0) = par[26];
    deltakpiExp(0, 1) = par[27];
    deltakpiExp(0, 2) = par[15];
    deltakpiExp(0, 3) = par[21];

    TMatrixD Mci_exp((*BESIII_CovMat.at(16)) - ciExp);
    TMatrixD Mki_exp((*BESIII_CovMat.at(17)) - kiExp);
    TMatrixD Mkip_exp((*BESIII_CovMat.at(17)) - kip_Exp);
    TMatrixD MBr_exp((*BESIII_CovMat.at(18)) - BrExp);
    TMatrixD Mdeltakpi_exp((*BESIII_CovMat.at(19)) - deltakpiExp);

    TMatrixD TciCOV(TMatrixD::kInverted, (*BESIII_CovMat.at(12)));
    TMatrixD TkiCOV(TMatrixD::kInverted, (*BESIII_CovMat.at(13)));
    TMatrixD TBrCOV(TMatrixD::kInverted, (*BESIII_CovMat.at(14)));
    TMatrixD TdeltakpiCOV(TMatrixD::kInverted, (*BESIII_CovMat.at(15)));

    TMatrixD Tci_exp(TMatrixD::kTransposed, Mci_exp);
    TMatrixD Tki_exp(TMatrixD::kTransposed, Mki_exp);
    TMatrixD Tkip_exp(TMatrixD::kTransposed, Mkip_exp);
    TMatrixD TBr_exp(TMatrixD::kTransposed, MBr_exp);
    TMatrixD Tdeltakpi_exp(TMatrixD::kTransposed, Mdeltakpi_exp);

    TMatrixD _Mci(Mci_exp, TMatrixD::kMult, TciCOV);
    TMatrixD Mci(_Mci, TMatrixD::kMult, Tci_exp);
    TMatrixD _Mki(Mki_exp, TMatrixD::kMult, TkiCOV);
    TMatrixD _Mkip(Mkip_exp, TMatrixD::kMult, TkiCOV);
    TMatrixD Mki(_Mki, TMatrixD::kMult, Tki_exp);
    TMatrixD Mkip(_Mkip, TMatrixD::kMult, Tkip_exp);
    TMatrixD _MBr(MBr_exp, TMatrixD::kMult, TBrCOV);
    TMatrixD MBr(_MBr, TMatrixD::kMult, TBr_exp);
    TMatrixD _Mdeltakpi(Mdeltakpi_exp, TMatrixD::kMult, TdeltakpiCOV);
    TMatrixD Mdeltakpi(_Mdeltakpi, TMatrixD::kMult, Tdeltakpi_exp);

    chi2 += Mci(0, 0) + MBr(0, 0) + Mki(0, 0) + Mdeltakpi(0, 0) + Mkip(0, 0);
    chi2 += (par[60] - 0.973) * (par[60] - 0.973) / (0.017 * 0.017);

    return chi2;
}

Double_t BESIII_chi2(const Double_t* parameters)
{
    std::string            file_path = __FILE__;
    std::string            dir_path  = file_path.substr(0, file_path.rfind("/"));

    // This would be a LOT faster if we didn't have to open the ROOT file every time
    // TODO that^
    TFile*                 fmatrix   = new TFile((dir_path + "/BESIII_CovMat.root").c_str(), "open");
    std::vector<TMatrixD*> BESIII_CovMat;
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("CPCOV"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("REkpi"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("Sk3pi"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("LSCOV"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("REOS"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("Sk3pi2"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("LSCOVp"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("REOSp"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("_NM_kspipi2"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("TCOVS_kspipi2"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("_NM_kspipi"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("_COVS_kspipi"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("ciCOV"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("kiCOV"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("BrCOV"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("deltakpiCOV"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("ciVal"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("kiVal"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("BrVal"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("deltakpiVal"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("REeff1"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("REeff3"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("COV2"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("Sk3piLS"));
    BESIII_CovMat.push_back((TMatrixD*)fmatrix->Get("Sk3piLSp"));
    fmatrix->Close();
    return total(parameters, BESIII_CovMat);
}
