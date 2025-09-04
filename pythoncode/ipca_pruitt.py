'''
Personal notes:
RZ = df containing returns and characteristics in a 2-level multiIndex, with 0-level = date,
     1-level = asset ID.
R = return series
Z = characteristics
self.X = characteristics weighted portfolios  (self.Q in Kellys code)
'''

'''
IPCA:
IPCA estimation class
version 1.0.3
copyright Seth Pruitt (2020)
'''

import sys
import pandas as pd
import numpy as np
import scipy.linalg as sla
# import pdb
import copy
from datetime import datetime
from timeit import default_timer as timer
# from numba import jit
from joblib import Parallel, delayed
from tqdm import tqdm

# matrix left/right division (following MATLAB function naming)


def _mldivide(denom, numer): return sla.lstsq(np.array(denom), np.array(numer))[0]
def _mrdivide(numer, denom): return (sla.lstsq(np.array(denom).T, np.array(numer).T)[0]).T


class ipca(object):
    def __init__(self, RZ=None, return_column=0, X=None, W=None, Nts=None, add_constant=True):
        '''
        [Inputs]
        RZ : df(TotalObs x L0+1) : returns and characteristics in a df with a 2-level MultiIndex index where Date is level 0 and
        AssetID is level 1. In a balanced dataset, TotalObs would equal the number of unique AssetIDs times the number
        of unique Dates, T; however, data is generally unbalanced and therefore the number of AssetIDs present varies
        over Dates. We split RZ into R df(TotalObs x 1) and Z df(TotalObs x L0). The
        code assumes that Z and R have been properly timed outside of IPCA, such that the rows of R with Date=d
        correspond to the rows of Z with Date=d according to the IPCA model in mind. E.g. the conditional APT of
        Kelly, Pruitt, Su (JFE 2019) says that the characteristics known at Date=d-1 determine the exposures associated
        with the returns realized at Date=d; hence, here we should have shifted the characteristics in Z relative to
        the returns in R. L0 is number of characteristics, and the unique CharNames are the 1-level column index of df Z.
            RZ is optional because we might use this code by passing in only the cross-sectional second moment df W,
            described below. E.g. when using the code for bootstrapped tests.

        return_column : int or str : which column of RZ, if given, is the return; all others are assumed to be chars.
            Default is 0: the first column. An int can be supplied to determine which column, or a column name str

        X : df(L x T) : Managed portfolio returns in a df with an index of CharNames (plus perhaps constant) and columns of the unique Dates
            X is optional because we might pass in R and Z directly and create X.
            See add_constant input description about dimension L.

        W : df(TL x L) : Cross-sectional second moments of characteristics, for the T unique Dates.
            W is optional because we might pass in Z directly and create W.
            See add_constant input description about dimension L.

        Nts : srs(T) : number of available assets for each Date
            Nts is optional because we can construct it from Z and R, if those are passed. Otherwise it is only
            useful if we are passing X and W, but not Z and R

        add_constant : bool : True (default) adds a constant to the calculation of X, W and it becomes the last char listed in
        in their indices. If true, then L = L0+1; if false, then L=L0

        [Transformed Inputs]
        X and W and Nts : df(L x T) and df(TL x L) and srs(T) : if Z,R input, these are created
        '''

        # inputs
        self.RZ, self.X, self.W, self.Nts = RZ, X, W, Nts
        self.has_RZ = False if RZ is None else True
        self.has_X = False if X is None else True
        self.has_W = False if W is None else True

        # check for Nts if X,W passed but not RZ
        if self.has_X and self.has_W and not self.has_RZ and self.Nts == None:
            raise ValueError('RZ are not passed; X,W were passed with no Nts')

        if self.has_RZ:
            if isinstance(return_column, int):  # pick return column using iloc
                if return_column >= RZ.shape[1]:
                    raise ValueError('The return_column int is greate'
                                     'r than the number of columns in RZ')
                self.R = RZ.iloc[:, return_column].to_frame()
            elif isinstance(return_column, str):  # pick return column using loc
                self.R = RZ.loc[:, return_column].to_frame()
            elif type(return_column) is tuple:  # pick return column using loc, with multiindex
                self.R = RZ.loc[:, return_column].to_frame()
            else:
                raise ValueError('Did not know how to pick return column')
            self.Z = RZ.drop(columns=self.R.columns)

        # create X and W if not provided
        if not self.has_X or not self.has_W:
            # the characteristics used for X, W (the index names of X df, column and index of each df in the W dict)
            if add_constant:
                charlist = list(self.Z.columns) + ['Constant']
            else:
                charlist = list(self.Z.columns)

            # the unique dates used for X, W (the column names of X df, key names of W dict)
            datelist = self.Z.index.get_level_values(0).unique().sort_values()  # CHANGED by Jorge.

            if not self.has_X:
                self.X = pd.DataFrame(index=charlist, columns=datelist, data=np.nan)
            if not self.has_W:
                self.W = pd.DataFrame(data=0.,
                                      index=pd.MultiIndex.from_product((datelist, charlist), names=['date', 'char']),
                                      columns=charlist)
            # ignoring a Nts if given
            self.Nts = pd.Series(index=datelist, data=np.nan)

            # create X and W
            for t in datelist:
                Zt = self.Z.loc[t, :].values
                if add_constant:
                    Zt = np.concatenate((Zt, np.ones((Zt.shape[0], 1))), axis=1)
                self.Nts[t] = Zt.shape[0]
                if not self.has_X:
                    self.X[t] = Zt.T.dot(self.R.loc[t]) / self.Nts[t]
                if not self.has_W:
                    self.W.loc[t] = Zt.T.dot(Zt) / self.Nts[t]

        # set properties
        self.Chars, self.Dates, self.L, self.add_constant = self.X.index, self.X.columns, self.X.shape[0], add_constant
        self.T = self.X.shape[1]
        self.N = len(np.unique(self.RZ.index.to_frame().values[:, 1]))
        # create ndarrays of X and W to speed up estimation
        self._X = self.X.values
        self._W = np.zeros((self.L, self.L, self.X.shape[1]), dtype='float32')
        ct = 0
        for t in self.Dates:
            self._W[:, :, ct] = self.W.loc[t].values
            ct += 1

    def fit(self, K=1, OOS=False, gFac=None, normalization_choice='PCA_positivemean',
            normalization_choice_specs=None, OOS_window='recursive',
            OOS_window_specs=60, factor_mean='constant', R_fit=True,
            Beta_fit=False, dispIters=False, minTol=1e-4, maxIters=5000, F_names=None,
            G_names=None, R2_bench='zero', dispItersInt=100):
        '''
        [Inputs]

        self : initialized IPCA object

        K : int : number of latent factors. We define KM = K + M and KM must be at least 1

        OOS : bool : false (default); True selects out-of-sample estimation, using other options listed below with
        prefix "OOS_"; the default False does in-sample estimation

        gFac : df(M x T) :  pre-specified factors or an anomaly term which is constant for all Dates
        gFac is optional because we might include no pre-specified factors or anomaly term

        normalization_choice : str : the normalization scheme for Factors and Gamma
            'PCA_positivemean' (default) mimics the typical PCA normalization (as in Kelly, Pruitt, Su JFE 2019) where
                - Gamma is unitary matrix
                - Factors are orthogonal to each other
                - Factor means are non-negative

            'Identity' enforces that each of K of the L characteristics have a loading of 1 on one of the factors and
            loadings of 0 on the remaining factors. This must be accompanied by K-list in normalization_choice_specs
            which determines which characteristics define this normalization. The ordering of the normalized factors
            follows the ordering of characteristics in that K-list

        normalization_choice_specs : list of str : if normalization_choice='Identity', then this K-list defines which
        characteristics have unit loading on one factor and zero loading on all others, using the characteristic names

        OOS_window : str : determines training window choice
            'recursive' (default) is an expanding window using data from the beginning
            'rolling' is a rolling window of fixed length

        OOS_window_specs : int : number defining the OOS_window specification
            if OOS_window='recursive', this is the minimum number of time periods required to construct estimates
            if OOS_window='rolling', this is the window length
            Default is 60

        factor_mean : str : choice of factor mean to use
            'constant' (default) uses the mean of the Factors on the training data
            'VAR1' estimates a VAR(1) on the training data and uses it to construct conditional mean estimates for a
            time-varying lambda

        R_fit : bool : if True then calculate the R fitted values (only possible if Z and R were input). If
        estimation is out-of-sample, then these are the out-of-sample fits

        Beta_fit : bool : if True then return the Betas (only possible if Z was input). If estimation is
        out-of-sample, then these are the Betas used for the out-of-sample fits

        dispIters : bool : if True then iteration/convergence information is displayed

        dispItersInt : int : if dispIters is True, then how many iterations to wait until showing convergence info

        minTol : scalar : small number used to define convergence in ALS algorithm

        maxIters : scalar : large number that is the maximum number of iterations allowed

        F_names : list : list with K elements that name the latent factors

        G_names : list : list with M elements that name the prespecified factors

        R2_bench : str : what is the benchmark against which to calculate the R2
            'zero' (default) uses a constant 0 forecast for every asset and managed portfolio
            'mean' uses a mean of every asset and managed portfolio; if out-of-sample, this is a mean calculated
            from the training data

        [Outputs] dict with following keys

        xfits : dict(4) of 2 df(L x T) and 2 scalar :
            'Fits_Total' df(L x T) with factor-fits (using factor realization)
            'Fits_Pred' df(L x T) with lambda-fits (using lambda, the expectation of factor)
            'R2_Total' is R2 using 'Fits_Total'
            'R2_Pred' is R2 using 'Fits_Pred'

        Gamma : df(L x KM) or df(TL x KM) : if in-sample estimation, it is a single df. If out-of-sample
        estimation, then there is a separate Gamma for each training sample, indexed by the first element of the index
        The first columns are always the K estimated ones, with the remaining the gFac

        Factor : df(KM x T) : the in-sample factor estimates, or if out-of-sample the out-of-sample factor
        realizations, plus the prespecified factors
        The first rows are always the K estimated ones, with the remaining the gFac

        Lambda : dict(2) : 'estimate'  is srs(KM) or df(KM x T), 'VAR1' is df(KM x KM+1)  : 'estimate' has
        the main output: if factor_mean='constant' then a srs; if factor_mean='VAR' then a df with the
        conditional means of every factor realization. 'VAR1' is the VAR1 coefficients with constant first

        rfits : dict(2) of 1 df(TotalObs x 2) and 2 scalars:
            Is None if R_fit=False
            Otherwise is analogous to xfits dict described above

        fittedBeta : df(TotalObs x KM) : there is a beta for every Date,AssetID index row, and for each factor column
            Is None if Beta_return=False
        '''

        self.K = K
        self.gFac = gFac

        fitstart = datetime.now()

        if gFac is None:
            has_gFac = False
            _gFac = None
        else:
            has_gFac = True
            _gFac = gFac.copy()

        if has_gFac:
            M = gFac.shape[0]
            if G_names is None:
                G_names = list(gFac.index)
        else:
            M = 0

        if M == 0:
            has_gFac = False

        KM = K + M

        # check for at least one factor
        if KM == 0:
            raise ValueError('K+M=0: there must be some factor used/estimated')

        if F_names is None and K > 0:
            F_names = list(range(K))
        if G_names is None and M > 0:
            G_names = gFac.index
        if K == KM:
            Factor_names = F_names
        elif M == KM:
            Factor_names = G_names
        else:
            Factor_names = F_names + G_names

        self.F_names, self.G_names, self.Factor_names = F_names, G_names, Factor_names

        if OOS:
            IS_or_OOS = 'OOS'
        else:
            IS_or_OOS = 'IS'

        # normalization_choice_specs turns into row indices for Gamma array
        if normalization_choice == 'Identity':
            normalization_choice_specs = self._find_sublist(normalization_choice_specs)
            if np.min(normalization_choice_specs) == -1:
                raise ValueError('normalization problem')

        #######
        # In-sample (Full-sample) estimation
        #######
        if IS_or_OOS == 'IS':
            Gamma0, Factor0 = self._svd_initial(K=K, M=M, gFac=gFac)
            tol, iters = float('inf'), 0
            timerstart = timer()
            while iters < maxIters and tol > minTol:
                iters += 1
                Gamma1, Factor1 = self._linear_als_estimation(Gamma0=Gamma0.copy(), gFac=_gFac,
                                                              K=K, M=M, KM=KM,
                                                              normalization_choice=normalization_choice,
                                                              normalization_choice_specs=normalization_choice_specs)
                tolGam = np.max(np.abs(Gamma1 - Gamma0))
                tolFac = np.max(np.abs(Factor1 - Factor0))
                tol = np.max((tolGam, tolFac))
                if dispIters and iters % dispItersInt == 0:
                    print('iters %i: tol = %0.8f' % (iters, tol))
                # replace 0 with 1, for next iteration
                Gamma0, Factor0 = Gamma1.copy(), Factor1.copy()

            Gamma, Factor = Gamma0.copy(), Factor0.copy()
            numerical_stats = {'tol': tol, 'minTol': minTol, 'iters': iters, 'maxIters': maxIters,
                               'time': timer() - timerstart}
            print('ipca.fit finished estimation after %i seconds and %i iterations'
                  % (numerical_stats['time'], numerical_stats['iters']))

            Rdo, Betado, fittedX, fittedR, fittedBeta = self._setup_fits(R_fit, Beta_fit)

            Gamma = pd.DataFrame(data=Gamma, index=self.Chars, columns=Factor_names)
            if factor_mean == 'constant':
                Lambda = pd.DataFrame(data=np.mean(Factor, axis=1), index=Factor_names)
                B = np.hstack((np.zeros((KM, KM)), np.mean(Factor, axis=1).reshape((-1, 1)))).T
            elif factor_mean == 'VAR1':
                Lambda = pd.DataFrame(data=np.nan, index=Factor_names, columns=self.Dates)
                Lambda.iloc[:, 0] = np.mean(Factor, axis=1)  # for initial forecast
                B = self._VARB(X=Factor)
            Factor = pd.DataFrame(data=Factor, index=Factor_names, columns=self.Dates)

            for t in self.Dates:
                if factor_mean == 'constant':
                    lamt = Lambda
                elif factor_mean == 'VAR1':
                    if t == self.Dates[0]:
                        lamt = Lambda[t].values  # use unconditonal mean forecast
                    else:
                        lamt = B.T.dot(np.hstack((Factor[t - 1].values, 1)).reshape((-1, 1)))
                    Lambda[t] = lamt.reshape((-1, 1))
                fittedX['Fits_Total'][t] = self.W.loc[t].dot(Gamma).dot(Factor[t]).values.reshape((-1, 1))
                fittedX['Fits_Pred'][t] = self.W.loc[t].dot(Gamma).dot(lamt).values.reshape((-1, 1))
                if Rdo:  # have to construct beta if Rdo=True
                    if self.add_constant:
                        Betat = self.Z.loc[t].dot(Gamma.iloc[:-1, :]) + Gamma.iloc[-1, :]
                    else:
                        Betat = self.Z.loc[t].dot(Gamma)
                    fittedR['Fits_Total'].loc[t] = Betat.dot(Factor[t]).values.reshape((-1, 1))
                    fittedR['Fits_Pred'].loc[t] = Betat.dot(lamt).values.reshape((-1, 1))
                    if Betado:
                        fittedBeta.loc[t] = Betat.values
                elif Betado:  # could have Rdo=False and Betado=True
                    if self.add_constant:
                        Betat = self.Z.loc[t].dot(Gamma.iloc[:-1, :]) + Gamma.iloc[-1, :]
                    else:
                        Betat = self.Z.loc[t].dot(Gamma)
                    fittedBeta.loc[t] = Betat.values

            if R2_bench == 'zero':
                benchR2 = None
            elif R2_bench == 'mean':
                benchR2 = pd.DataFrame(data=np.tile(self.X.mean(axis=1), self.X.shape[1]),
                                       index=self.X.index, columns=self.X.columns)
            fittedX['R2_Total'], fittedX['R2_Pred'] = self._R2_calc(
                reals=self.X, fits_total=fittedX['Fits_Total'], fits_pred=fittedX['Fits_Pred'],
                benchR2=benchR2)

            if Rdo:
                if R2_bench == 'zero':
                    benchR2 = None
                elif R2_bench == 'mean':
                    benchR2 = pd.DataFrame(data=np.ones((self.R.shape[0], 1)) * self.R.mean().values,
                                           index=self.R.index, columns=self.R.columns)
                fittedR['R2_Total'], fittedR['R2_Pred'] = self._R2_calc(
                    reals=self.R, fits_total=fittedR['Fits_Total'], fits_pred=fittedR['Fits_Pred'],
                    benchR2=benchR2)

        #######
        # Out of sample
        #######
        elif IS_or_OOS == 'OOS':
            Gamma0, Factor0 = self._svd_initial(K=K, M=M, gFac=gFac,
                                                Dates=self.Dates[:OOS_window_specs])
            Rdo, Betado, fittedX, fittedR, fittedBeta = self._setup_fits(R_fit, Beta_fit)

            Lambda = pd.DataFrame(data=np.nan, index=Factor_names, columns=self.Dates)
            Factor = Lambda.copy()
            Gamma = pd.DataFrame(data=np.nan, columns=Factor_names,
                                 index=pd.MultiIndex.from_product([self.Dates, self.Chars]))
            if R2_bench == 'mean':
                benchX = pd.DataFrame(data=0., columns=self.Dates, index=self.Chars)
            elif R2_bench == 'zero':
                benchX = None

            if Rdo:
                if R2_bench == 'mean':
                    benchR = pd.DataFrame(data=0., index=self.R.index, columns=self.R.columns)
                elif R2_bench == 'zero':
                    benchR = None

            numerical_stats = {'minTol': minTol, 'maxIters': maxIters,
                               'tol': pd.DataFrame(data=np.nan, columns=self.Dates, index=[0]),
                               'iters': pd.DataFrame(data=np.nan, columns=self.Dates, index=[0]),
                               'time': pd.DataFrame(data=np.nan, columns=self.Dates, index=[0])}
            ct = 0
            for t in tqdm(self.Dates[OOS_window_specs:], desc="OOS fit", dynamic_ncols=True,
                          leave=True, file=sys.stdout):
                tol, iters = float('inf'), 0
                if OOS_window == 'rolling':
                    datest = self.X.loc[:, :t - OOS_window_specs:t - 1].columns  # dates through t-1, not including t
                else:
                    datest = self.X.loc[:, :t - 1].columns  # dates through t-1, not including t
                # Using datest below means that we estimate Gamma0, Factor0 on data through t-1, which include returns through t-1
                # and chars through t-2. Then the last Factor realization is for t-1
                timerstart = timer()
                while iters < maxIters and tol > minTol:
                    iters += 1
                    Gamma1, Factor1 = self._linear_als_estimation(Gamma0=Gamma0.copy(), gFac=gFac,
                                                                  K=K, M=M, KM=KM,
                                                                  normalization_choice=normalization_choice,
                                                                  normalization_choice_specs=normalization_choice_specs,
                                                                  Dates=datest)
                    tolGam = np.max(np.abs(Gamma1 - Gamma0))
                    # tolFac = np.max(np.abs(Factor1 - Factor0))
                    # tol = np.max((tolGam, tolFac))
                    if K > 0:
                        tolFac = np.max(np.abs(Factor1 - Factor0))
                        tol = np.max((tolGam, tolFac))
                    else:
                        tol = tolGam
                    if dispIters and iters % dispItersInt == 0:
                        print('iters {}: tol = {}'.format(iters, tol))
                    Gamma0, Factor0 = Gamma1.copy(), Factor1.copy()

                numerical_stats['tol'][t] = tol
                numerical_stats['iters'][t] = iters
                numerical_stats['time'][t] = timer() - timerstart
                Gamma.loc[t] = Gamma0
                # "OOS" factor realization: use Gamma0 and W known at time t-1, but X known at time t
                if M == 0:
                    Factor[t] = _mldivide(Gamma.loc[t].T.dot(self.W.loc[t]).dot(Gamma.loc[t]),
                                          Gamma.loc[t].T.dot(self.X[t]))
                else:
                    if K > 0:
                        tmp = _mldivide(Gamma.loc[t, F_names].T.dot(self.W.loc[t]).dot(Gamma.loc[t, F_names]),
                                        Gamma.loc[t, F_names].T.dot(self.X[t]))
                        # Factor[t].loc[F_names] = tmp
                        # Factor[t].loc[G_names] = gFac[t].values
                        Factor.loc[F_names, t] = tmp
                    # prespecified factors (always, if M>0)
                    Factor.loc[G_names, t] = gFac.loc[G_names, t].values
                fittedX['Fits_Total'][t] = self.W.loc[t].dot(Gamma.loc[t]).dot(Factor[t]).values.reshape((-1, 1))
                # calculate Lambda and fill predictive fits
                if factor_mean == 'constant':
                    Lambda[t] = Factor0.mean(axis=1)
                    B = np.hstack((np.zeros((KM, KM)), Factor0.mean(axis=1).reshape((-1, 1)))).T
                elif factor_mean == 'VAR1':
                    B = self._VARB(X=Factor0)
                    Lambda[t] = B.T.dot(np.hstack((Factor0[:, -1], 1)).reshape((-1, 1)))

                # predictive fittedX
                fittedX['Fits_Pred'][t] = self.W.loc[t].dot(Gamma.loc[t]).dot(Lambda[t]).values.reshape((-1, 1))
                if R2_bench == 'mean':
                    benchX[t] = self.X[:t].values.mean()

                # fittedR and fittedBeta
                if Rdo:
                    if self.add_constant:
                        Betat = self.Z.loc[t].dot(Gamma0[:-1, :]) + Gamma0[-1, :]
                    else:
                        Betat = self.Z.loc[t].dot(Gamma0)
                    fittedR['Fits_Total'].loc[t] = Betat.dot(Factor[t].values).values.reshape((-1, 1))
                    fittedR['Fits_Pred'].loc[t] = Betat.dot(Lambda[t].values).values.reshape((-1, 1))

                    if R2_bench == 'mean':
                        benchR[t] = self.R.loc[:t].values.mean()
                    if Betado:
                        fittedBeta.loc[t] = Betat
                elif Betado:
                    if self.add_constant:
                        Betat = self.Z.loc[t].dot(Gamma0[:-1, :]) + Gamma0[-1, :]
                    else:
                        Betat = self.Z.loc[t].dot(Gamma0)
                    fittedBeta.loc[t] = Betat.values

                # Prepare for next period through loop
                if K > 0:
                    Factor0 = np.concatenate((Factor0.copy(), np.zeros((Factor0.shape[0], 1))), axis=1)
                else:
                    Factor0 = gFac.loc[:, datest].values

                ct += 1
                if dispIters and ct % 12 == 0:
                    tqdm.write(f"{t} done in {numerical_stats['iters'][t].item()} iterations, "
                               f"{numerical_stats['time'][t].item():.2f} seconds")

            # R2s
            fittedX['R2_Total'], fittedX['R2_Pred'] = self._R2_calc(
                reals=self.X.loc[:, self.Dates[OOS_window_specs:]],
                fits_total=fittedX['Fits_Total'].loc[:, self.Dates[OOS_window_specs:]],
                fits_pred=fittedX['Fits_Pred'].loc[:, self.Dates[OOS_window_specs:]],
                benchR2=benchX)
            if Rdo:
                fittedR['R2_Total'], fittedR['R2_Pred'] = self._R2_calc(
                    reals=self.R.loc[self.Dates[OOS_window_specs]:],
                    fits_total=fittedR['Fits_Total'].loc[self.Dates[OOS_window_specs]:],
                    fits_pred=fittedR['Fits_Pred'].loc[self.Dates[OOS_window_specs]:],
                    benchR2=benchR)

        # Fill the Lambda dict
        Lambda_dict = {}
        Lambda_dict['estimate'] = Lambda
        Lambda_dict['VAR1'] = pd.DataFrame(data=B.T, index=Factor_names, columns=(Factor_names + ['cons']))

        # numerical information
        fitend = datetime.now()
        numerical_stats['fit_start_time'], numerical_stats['fit_end_time'] = fitstart, fitend

        # indices, metad = self._prep_metad()
        # Assign attributes instead of just return them in dict
        self.Gamma, self.Factors = Gamma.values, Factor.values

        return {'xfits': fittedX,
                'Gamma': Gamma,
                'Factor': Factor,
                'Lambda': Lambda_dict,
                'rfits': fittedR,
                'fittedBeta': fittedBeta,
                'numerical': numerical_stats
                }

    def _linear_als_estimation(self, Gamma0, K, M, KM, normalization_choice, normalization_choice_specs,
                               gFac, Dates=None, X=None, W=None):
        '''
        Runs one iteration of the alternating least squares estimation process

        [Inputs]
        Gamma0 : df(L x KM) : previous iteration's Gamma estimate

        Dates : None, index : if None, uses self.Dates (full sample); otherwise uses the given Dates index which have
        T2 entries

        Other inputs of this particular fitted object, which should simply be passed

        [Outputs]
        Gamma1 : df(L x KM) : current iteration's Gamma estimate

        Factor : df(KM x T2) : current iteration's latent Factor estimate, or None
        '''
        # dataframe-based, using self.X, self.W
        if Dates is None:
            Dates = self.Dates
        # # ndarray-based, using self._X, self._W
        # if Dates is None:
        #     Dates = range(0, len(self.Dates))

        # Added Jorge
        if X is None:
            X = self.X

        if W is None:
            W = self.W

        if K == KM:
            GammaF = Gamma0
            GammaG = None
        elif M == KM:
            GammaF = None
            GammaG = Gamma0

        else:
            GammaF, GammaG = Gamma0[:, 0:K], Gamma0[:, K:]
        # 1. estimate latent factor, if K>0
        if K > 0:
            FactorF = np.zeros((K, len(Dates)))
            # dataframe-based
            ct = 0
            for t in Dates:
                if M > 0:
                    FactorF[:, ct] = _mldivide(
                        GammaF.T.dot(W.loc[t].values).dot(GammaF),
                        GammaF.T.dot(X[t].values - W.loc[t].values.dot(GammaG).dot(gFac[t].values))
                    )
                else:
                    FactorF[:, ct] = _mldivide(
                        GammaF.T.dot(W.loc[t].values).dot(GammaF),
                        GammaF.T.dot(X[t].values)
                    )
                ct += 1
            # # ndarray-based
            # ct=0
            # for t in Dates:
            #     if M>0:
            #         FactorF[:, ct] = _mldivide(
            #             GammaF.T.dot(self._W[:, :, t]).dot(GammaF),
            #             GammaF.T.dot(self._X[:, t] - self._W[:, :, t].values.dot(GammaG).dot(gFac[:, t]))
            #         )
            #     else:
            #         FactorF[:, ct] = _mldivide(
            #             GammaF.T.dot(self._W[:, :, t]).dot(GammaF),
            #             GammaF.T.dot(self._X[:, t])
            #         )
            #     ct+=1
        else:
            FactorF = None
        # 2. estimate Gamma, for latent or prespecified factors which are present
        if K == KM:
            Factor = FactorF
        elif M == KM:
            # dataframe-based
            Factor = gFac[Dates].values
            # # ndarray-based
            # Factor = gFac[:, Dates]
        else:
            # dataframe-based
            Factor = np.concatenate((FactorF.copy(), gFac[Dates].values.copy()), axis=0)
            # # ndarray-based
            # Factor = np.concatenate((FactorF, gFac[:, Dates]), axis=0)

        numer, denom = np.zeros(self.L * KM), np.zeros((self.L * KM, self.L * KM))
        # dataframe-based
        ct = 0
        for t in Dates:
            numer += np.kron(X[t].values, Factor[:, ct]) * self.Nts[t]
            denom += np.kron(W.loc[t].values, np.outer(Factor[:, ct], Factor[:, ct])) * self.Nts[t]
            ct += 1
        # # ndarray-based
        # ct=0
        # for t in Dates:
        #     numer += np.kron(self._X[:, t], Factor[:, ct]) * self.Nts[t]
        #     denom += np.kron(self._W[:, :, t], np.outer(Factor[:, ct], Factor[:, ct])) * self.Nts[t]
        #     ct+=1

        Gamma1 = np.reshape(_mldivide(denom, numer), (self.L, KM))
        # 3. normalization
        if K > 0:
            Gamma1, Factor1 = self._normalization_choice(Gamma=Gamma1.copy(), Factor=Factor.copy(),
                                                         K=K, M=M, KM=KM,
                                                         normalization_choice=normalization_choice,
                                                         normalization_choice_specs=normalization_choice_specs)
        else:
            Factor1 = Factor.copy()
        return Gamma1, Factor1

    # calculate SVD for initial conditions for ALS algorithm

    def _svd_initial(self, K, M, gFac=None, Dates=None, X=None):
        '''
        Dates : None, index : if None, uses self.Dates (full sample); otherwise uses the given Dates index

        gFac : None, (M x T)array :
        '''
        if Dates is None:
            Dates = self.Dates

        # Added Jorge
        if X is None:
            X = self.X
        # END

        Gamma, Factor, = None, None

        if K > 0:
            U, s, VT = sla.svd(X.loc[:, Dates].values, full_matrices=False)
            Gamma = U[:, 0:K]
            Factor = np.diag(s[0:K]).dot(VT[0:K, :])
        if M > 0 and K > 0:
            Gamma = np.concatenate((Gamma.copy(), np.zeros((self.L, M))), axis=1)
            Factor = np.concatenate((Factor.copy(), gFac[Dates].values), axis=0)
        elif M > 0 and K == 0:
            Gamma = np.zeros((self.L, M))
            Factor = gFac.values
        return Gamma, Factor

    # normalize Gamma, Factor

    def _normalization_choice(self, Gamma, Factor, K, M, KM,
                              normalization_choice, normalization_choice_specs):
        if M == KM:
            raise ValueError('do not use _normalization_choice with no latent factors (M==KM)')
        if normalization_choice == 'PCA_positivemean':
            if K == KM:
                GammaF, FactorF = Gamma, Factor
            else:
                GammaF, GammaG = Gamma[:, 0:K], Gamma[:, K:]
                FactorF, FactorG = Factor[0:K, :], Factor[K:, :]
            R1 = sla.cholesky(GammaF.T.dot(GammaF))
            R2, _, _ = sla.svd(R1.dot(FactorF).dot(FactorF.T).dot(R1.T))
            GammaF = _mrdivide(GammaF, R1).dot(R2)
            FactorF = _mldivide(R2, R1.dot(FactorF))
            # sign convention: FactorF has positive mean
            sign_conv = np.sign(np.mean(FactorF, axis=1)).reshape((-1, 1))
            sign_conv[sign_conv == 0] = 1
            FactorF = FactorF * sign_conv
            GammaF = GammaF * sign_conv.T
            # orthogonality between GammaF and GammaG
            if M > 0:
                GammaG = (np.identity(self.L) - GammaF.dot(GammaF.T)).dot(GammaG)
                FactorF = FactorF + (GammaF.T.dot(GammaG)).dot(FactorG)
            # sign convention: FactorF has positive mean
            sign_conv = np.sign(np.mean(FactorF, axis=1)).reshape((-1, 1))
            sign_conv[sign_conv == 0] = 1
            FactorF = FactorF * sign_conv
            GammaF = GammaF * sign_conv.T
        elif normalization_choice == 'Identity':
            if K == KM:
                GammaF, FactorF = Gamma, Factor
            else:
                GammaF, GammaG = Gamma[:, 0:K], Gamma[:, K:]
                FactorF, FactorG = Factor[0:K, :], Factor[K:, :]
            R = GammaF[normalization_choice_specs, :]
            GammaF = _mrdivide(GammaF, R)
            FactorF = R.dot(FactorF)
            # orthogonality between GammaF and GammaG
            if M > 0:
                GammaG = (np.identity(self.L) - GammaF.dot(GammaF.T)).dot(GammaG)
                FactorF = FactorF + (GammaF.T.dot(GammaG)).dot(FactorG)

        if M > 0:
            Gamma = np.concatenate((GammaF.copy(), GammaG.copy()), axis=1)
            Factor = np.concatenate((FactorF.copy(), FactorG.copy()), axis=0)
        else:
            Gamma = GammaF.copy()
            Factor = FactorF.copy()
        return Gamma, Factor

    # estimate a VAR1 coefficent matrix (including constant)

    def _VARB(self, X):
        Xtil = np.concatenate((X.copy(), np.ones((1, X.shape[1]))), axis=0)
        B = sla.lstsq(Xtil[:, :-1].T, X[:, 1:].T)[0]
        return B

    # internal R2 calculation code

    def _R2_calc(self, reals, fits_total, fits_pred, benchR2=None):
        if benchR2 is None:
            R2_Total = 1 - ((reals - fits_total) ** 2).values.sum() / (reals ** 2).values.sum()
            R2_Pred = 1 - ((reals - fits_pred) ** 2).values.sum() / (reals ** 2).values.sum()
        else:
            R2_Total = 1 - ((reals - fits_total) ** 2).values.sum() / (
                (reals - benchR2) ** 2).values.sum()
            R2_Pred = 1 - ((reals - fits_pred) ** 2).values.sum() / (
                (reals - benchR2) ** 2).values.sum()

        return R2_Total, R2_Pred

    # external R2 calculation code

    def R2_of_fits(self, results=None, date_range=None, benchR2=None, R2name=None, inplace=True):
        '''
        Calculates R2 over given datetime ranges

        [Inputs]

        results : dict : the output of an object output from ipca.IPCA.fit()

        date_range : datetime index : a choice of dates in the dfs within results over which to calculate the R2

        benchR2 : str, None : choice of benchmark; in the following the in-sample period is the one decided by date_range,
        but for out-of-sample we use data from the beginning of the available dates, even if this is prior to date_range (which
        will then be used to determine which periods' fits to put into the R2 calculation)
            - if None (default) then benchmark is taken as 0
            - if 'all_mean' then use in-sample mean of all observations
            - if 'individual_mean' then use in-sample mean of each cross-sectional unit's observations
            - if 'all_mean_recursive' then use recursively estimated out-of-sample mean of all observations
            - if 'individual_mean_recursive' then use recursively estimated out-of-sample mean of each cross-sectional unit's observations

        R2name : str, None : if None (default) then the suffix of the name of the R2s are given by the first and last date in date_range
        using strftime with the '%Y%m%d' format chosen (eg 'R2_Total_19650130_20161231')
            If a string is passed, that is instead the name of the R2 (eg 'R2_Total_mychosenstring')

        inplace : bool : if True (default) then the passed results dict is modified and None is passed out; if False, then the
        passed results dict is copied and then modified and passed out

        [Outputs]

        results : a dict with 'rfits' and 'xfits' keys of the results dict; if the the original ipca object had to R, then only 'xfits' is output
        if inplace=True (default) then the passed results dict is modified; if inplace=False then the passed results dict is copied and sent out

        '''
        # error checking
        if results is None:
            raise ValueError('ipca.R2_of_fits requires a results dict to be passed')
        if date_range is None:
            raise ValueError('ipca.R2_of_fits requires a date_range list or index to be passed')

        # calculate the appropriate benchmarks
        if benchR2 is None:
            benchX = pd.DataFrame(data=0., index=self.X.index, columns=self.X.columns)
            if self.has_RZ:
                benchR = pd.DataFrame(data=0., index=self.R.index, columns=self.R.columns)
        elif benchR2 == 'all_mean':
            benchX = pd.DataFrame(data=np.mean(self.X.values), index=self.X.index, columns=self.X.columns)
            if self.has_RZ:
                benchR = pd.DataFrame(data=np.mean(self.R.values), index=self.R.index, columns=self.R.columns)
        elif benchR2 == 'individual_mean':
            benchX = pd.DataFrame(data=np.tile(self.X.T.loc[date_range].T.mean(axis=1).values.reshape((-1, 1)), (1, self.X.shape[1])),
                                  index=self.X.index, columns=self.X.columns)
            if self.has_RZ:
                benchR = self.R.loc[date_range].groupby(level=1).transform(lambda x: x.mean())
        elif benchR2 == 'all_mean_recursive':
            benchX = pd.DataFrame(data=0., index=self.X.index, columns=self.X.columns)
            for t in self.X.columns[1:]:
                benchX[t] = np.mean(self.X.T.loc[:t - 1].values)
            if self.has_RZ:
                benchR = pd.DataFrame(data=0., index=self.R.index, columns=self.R.columns)
                for t in self.X.columns[1:]:
                    benchR.loc[t] = np.mean(self.R.loc[:t].values)
        elif benchR2 == 'individual_mean_recursive':
            benchX = self.X.T.shift(1).expanding().mean().T
            if self.has_RZ:
                benchR = self.R.groupby(level=1).shift(1).expanding().mean()

        # calculate the R2s
        r2_x_t = (1 -
                  ((self.X.T.loc[date_range] - results['xfits']['Fits_Total'].T.loc[date_range])**2).values.sum() /
                  ((self.X.T.loc[date_range] - benchX.T.loc[date_range])**2).values.sum()
                  )
        r2_x_p = (1 -
                  ((self.X.T.loc[date_range] - results['xfits']['Fits_Pred'].T.loc[date_range])**2).values.sum() /
                  ((self.X.T.loc[date_range] - benchX.T.loc[date_range])**2).values.sum()
                  )
        if self.has_RZ:
            r2_r_t = (1 -
                      ((self.R.loc[date_range] - results['rfits']['Fits_Total'].loc[date_range])**2).values.sum() /
                      ((self.R.loc[date_range] - benchR.loc[date_range])**2).values.sum()
                      )
            r2_r_p = (1 -
                      ((self.R.loc[date_range] - results['rfits']['Fits_Pred'].loc[date_range])**2).values.sum() /
                      ((self.R.loc[date_range] - benchR.loc[date_range])**2).values.sum()
                      )

        # find the name
        if R2name is None:
            R2name = date_range[0].strftime('%Y%m%d') + '-' + date_range[-1].strftime('%Y%m%d')

        # inplace
        if inplace:
            results['rfits']['R2_Total_' + R2name] = r2_r_t
            results['rfits']['R2_Pred_' + R2name] = r2_r_p
            results['xfits']['R2_Total_' + R2name] = r2_x_t
            results['xfits']['R2_Pred_' + R2name] = r2_x_p
            newresults = None
        else:
            newresults = copy.deepcopy(results)
            newresults['rfits']['R2_Total_' + R2name] = r2_r_t
            newresults['rfits']['R2_Pred_' + R2name] = r2_r_p
            newresults['xfits']['R2_Total_' + R2name] = r2_x_t
            newresults['xfits']['R2_Pred_' + R2name] = r2_x_p

        return newresults

    # Set-up function using in both IS and OOS routines

    def _setup_fits(self, R_fit, Beta_return):
        Rdo = False
        if (R_fit and not self.has_RZ):
            print('ipca.fit() was given R_fit=True but is not given a RZ; None is being passed out as '
                  '"fittedR"')
            Rdo = False
        elif R_fit:
            Rdo = True

        Betado = False
        if Beta_return and not self.has_RZ:
            print('ipca.fit() was given Beta_return=True but is not given a Z; None is being passed out as '
                  '"fittedBeta"')
            Betado = False
        elif Beta_return:
            Betado = True

        fittedX = {'Fits_Total': pd.DataFrame(np.nan, index=self.X.index, columns=self.Dates),
                   'Fits_Pred': pd.DataFrame(np.nan, index=self.X.index, columns=self.Dates)}

        if Rdo:
            fittedR = {'Fits_Total': pd.DataFrame(np.nan, index=self.R.index, columns=self.R.columns),
                       'Fits_Pred': pd.DataFrame(np.nan, index=self.R.index, columns=self.R.columns)}
        else:
            fittedR = None

        if Betado:
            fittedBeta = pd.DataFrame(np.nan, index=self.R.index, columns=self.Factor_names)
        else:
            fittedBeta = None

        return Rdo, Betado, fittedX, fittedR, fittedBeta

    # find sublist

    def _find_sublist(self, sub):
        pos = list()
        ct = 0
        for subj in sub:
            try:
                pos.append(list(self.Chars).index(subj))
            except:
                pos.append(-1)
                print('We did not find %s in the list of chars' % subj)
            ct += 1
        return pos

    ########################################################################
    # Extra code based on ipca python package
    ########################################################################

    def X_as_array(self):
        """Convert Pruitt's X DataFrame to a (L, T) NumPy array like Kelly's Q."""
        chars = list(self.X.index)
        dates = list(self.X.columns)
        # Ordering match self.Chars and self.metad["dates"]
        X_df_sorted = self.X.loc[chars, dates]
        return X_df_sorted.values

    def W_as_array(self):
        """Convert Pruitt's W DataFrame to a (L, L, T) NumPy array."""
        chars = list(self.X.index)
        dates = list(self.X.columns)

        # Reorder W to match the chars/dates ordering
        W_df_sorted = self.W.loc[
            pd.MultiIndex.from_product([dates, chars], names=['date', 'char']),
            chars
        ]

        L = len(chars)
        T = len(dates)

        # Reshape to (T, L, L) then transpose to (L, L, T)
        W_array = (
            W_df_sorted.values
            .reshape(T, L, L)
            .transpose(1, 2, 0)
        )
        return W_array

    def BS_Walpha(self, ndraws=1000, n_jobs=1, backend='loky',
                  minTol=1e-4, maxIters=1000):
        """
        Bootstrap inference on the hypothesis Gamma_alpha = 0

        Parameters
        ----------

        ndraws  : integer, default=1000
            Number of bootstrap draws and re-estimations to be performed

        backend : optional
            Value is either 'loky' or 'multiprocessing'

        n_jobs  : integer
            Number of workers to be used. If -1, all available workers are
            used.

        Returns
        -------

        pval : float
            P-value from the hypothesis test H0: Gamma_alpha=0
        """

        if not self.add_constant:
            raise ValueError('Need to fit model with intercept first.')

        # fail if model isn't estimated
        if not hasattr(self, "X"):
            raise ValueError("Bootstrap can only be run on fitted model.")

        N, L, T = self.N, self.L, self.T

        # Compute Walpha
        Walpha = self.Gamma[:, -1].T.dot(self.Gamma[:, -1])

        # Compute residuals
        d = np.full((L, T), np.nan)

        X_array = self.X_as_array()
        W_array = self.W_as_array()
        dates = self.Dates.values

        for t_i, t in enumerate(dates):
            d[:, t_i] = X_array[:, t_i] - W_array[:, :, t_i].dot(self.Gamma)\
                .dot(self.Factors[:, t_i])

        print("Starting Bootstrap...")
        Walpha_b = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(_BS_Walpha_sub)(self, n, d, minTol, maxIters) for n in range(ndraws))
        print("Done!")

        # print(Walpha_b, Walpha)
        pval = np.sum(Walpha_b > Walpha) / ndraws
        return pval

    def BS_Wbeta(self, l, ndraws=1000, n_jobs=1, backend='loky',
                 minTol=1e-4, maxIters=1000):
        """
        Test of instrument significance.
        Bootstrap inference on the hypothesis  l-th column of Gamma_beta = 0.

        Parameters
        ----------

        l   : integer
            Position of the characteristics for which the bootstrap is to be
            carried out. For example, if there are 10 characteristics, l is in
            the range 0 to 9 (left-/right-inclusive).

        ndraws  : integer, default=1000
            Number of bootstrap draws and re-estimations to be performed

        n_jobs  : integer
            Number of workers to be used for multiprocessing.
            If -1, all available Workers are used.

        backend : optional

        Returns
        -------

        pval : float
            P-value from the hypothesis test H0: Gamma_alpha=0
        """

        # if self.PSFcase:
        #     raise ValueError('Need to fit model without intercept first.')

        # fail if model isn't estimated
        if not hasattr(self, "X"):
            raise ValueError("Bootstrap can only be run on fitted model.")

        N, L, T = self.N, self.L, self.T

        # Compute Wbeta_l if l-th characteristics is set to zero
        Wbeta_l = np.sum(np.square(self.Gamma[l, :]))

        X_array = self.X_as_array()
        W_array = self.W_as_array()
        dates = self.Dates.values

        # Compute residuals
        d = np.full((L, T), np.nan)
        for t_i, t in enumerate(dates):
            d[:, t_i] = X_array[:, t_i] - W_array[:, :, t_i].dot(self.Gamma)\
                .dot(self.Factors[:, t_i])

        print("Starting Bootstrap...")
        Wbeta_l_b = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(_BS_Wbeta_sub)(self, n, d, l, minTol, maxIters) for n in range(ndraws))
        print("Done!")

        pval = np.sum(Wbeta_l_b > Wbeta_l) / ndraws

        return pval

    def BS_Wdelta(self, ndraws=1000, n_jobs=1, backend='loky',
                  minTol=1e-4, maxIters=1000):
        """
        Bootstrap inference on the hypothesis Gamma_delta = 0
        (significance of pre-specified factor).

        Parameters
        ----------
        ndraws  : int
            Number of bootstrap draws.
        n_jobs  : int
            Number of workers for parallelization.
        backend : str
            Joblib backend.
        minTol, maxIters : controls ALS convergence in re-estimation

        Returns
        -------
        pval : float
            p-value from test H0: Gamma_delta = 0
        """

        if not hasattr(self, "X"):
            raise ValueError("Bootstrap can only be run on a fitted model.")

        if self.G_names is None or len(self.G_names) != 1:
            raise ValueError("Implemented only for exactly one pre-specified factor.")

        N, L, T = self.N, self.L, self.T

        # Compute Wdelta (test statistic in actual sample)
        Wdelta = self.Gamma[:, -1].T.dot(self.Gamma[:, -1])

        # Compute residuals
        d = np.full((L, T), np.nan)
        X_array = self.X_as_array()
        W_array = self.W_as_array()
        dates = self.Dates.values

        for t_i, t in enumerate(dates):
            d[:, t_i] = X_array[:, t_i] - W_array[:, :, t_i].dot(self.Gamma).dot(self.Factors[:, t_i])

        print("Starting Bootstrap for delta...")
        Wdelta_b = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(_BS_Wdelta_sub)(self, n, d, minTol, maxIters) for n in range(ndraws)
        )
        print("Done!")

        pval = np.sum(Wdelta_b > Wdelta) / ndraws
        return pval

    def _fit(self, X, W, OOS=False, gFac=None, normalization_choice='PCA_positivemean',
             normalization_choice_specs=None, dispIters=False, minTol=1e-4, maxIters=5000, F_names=None,
             G_names=None, dispItersInt=100):
        '''
        fit function adapted to run the alpha and beta bootstraps
        '''

        K = self.K

        if gFac is None:
            has_gFac = False
            _gFac = None
        else:
            has_gFac = True
            _gFac = gFac.copy()

        if has_gFac:
            M = gFac.shape[0]
            if G_names is None:
                G_names = list(gFac.index)
        else:
            M = 0

        if M == 0:
            has_gFac = False

        KM = K + M

        # check for at least one factor
        if KM == 0:
            raise ValueError('K+M=0: there must be some factor used/estimated')

        if F_names is None and K > 0:
            F_names = list(range(K))
        if G_names is None and M > 0:
            G_names = gFac.index
        if K == KM:
            Factor_names = F_names
        elif M == KM:
            Factor_names = G_names
        else:
            Factor_names = F_names + G_names

        self.F_names, self.G_names, self.Factor_names = F_names, G_names, Factor_names

        if OOS:
            IS_or_OOS = 'OOS'
        else:
            IS_or_OOS = 'IS'

        # normalization_choice_specs turns into row indices for Gamma array
        if normalization_choice == 'Identity':
            normalization_choice_specs = self._find_sublist(normalization_choice_specs)
            if np.min(normalization_choice_specs) == -1:
                raise ValueError('normalization problem')

        #######
        # In-sample (Full-sample) estimation
        #######
        if IS_or_OOS == 'IS':
            Gamma0, Factor0 = self._svd_initial(X=X, K=K, M=M, gFac=gFac)
            tol, iters = float('inf'), 0
            while iters < maxIters and tol > minTol:
                iters += 1
                Gamma1, Factor1 = self._linear_als_estimation(X=X, W=W, Gamma0=Gamma0.copy(), gFac=_gFac,
                                                              K=K, M=M, KM=KM,
                                                              normalization_choice=normalization_choice,
                                                              normalization_choice_specs=normalization_choice_specs)
                tolGam = np.max(np.abs(Gamma1 - Gamma0))
                tolFac = np.max(np.abs(Factor1 - Factor0))
                tol = np.max((tolGam, tolFac))
                if dispIters and iters % dispItersInt == 0:
                    print('iters %i: tol = %0.8f' % (iters, tol))
                # replace 0 with 1, for next iteration
                Gamma0, Factor0 = Gamma1.copy(), Factor1.copy()

            Gamma, Factor = Gamma0.copy(), Factor0.copy()

            Gamma = pd.DataFrame(data=Gamma, index=self.Chars, columns=Factor_names)
            Factor = pd.DataFrame(data=Factor, index=Factor_names, columns=self.Dates)

        else:
            raise ValueError('Function only works for IS estimation')

        # indices, metad = self._prep_metad()
        # Assign attributes instead of just return them in dict
        Gamma_New, Factor_New = Gamma.values, Factor.values

        return Gamma_New, Factor_New


def _BS_Walpha_sub(model, n, d, minTol=1e-4, maxIters=1000):
    L, T = model.L, model.T
    X_b = np.full((L, T), np.nan)
    np.random.seed(n)

    X_array = model.X_as_array()
    W_array = model.W_as_array()

    # Re-estimate unrestricted model
    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:, np.random.randint(0, high=T)]
                X_b[:, t] = W_array[:, :, t].dot(model.Gamma[:, :-1])\
                    .dot(model.Factors[:-1, t]) + d_temp

            # convert to dataframe to match input in _fit()
            X_b_df = pd.DataFrame(X_b, index=model.Chars,
                                  columns=model.Dates)
            Gamma, Factors = model._fit(X=X_b_df, W=model.W,
                                        minTol=minTol, maxIters=maxIters,
                                        gFac=model.gFac)
        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                            Observation discarded.")
            pass

    # Compute and store Walpha_b
    Walpha_b = Gamma[:, -1].T.dot(Gamma[:, -1])

    return Walpha_b


def _BS_Wbeta_sub(model, n, d, l, minTol=1e-4, maxIters=1000):
    L, T = model.L, model.T
    X_b = np.full((L, T), np.nan)
    np.random.seed(n)
    # Modify Gamma_beta such that its l-th row is zero
    Gamma_beta_l = np.copy(model.Gamma)
    Gamma_beta_l[l, :] = 0

    X_array = model.X_as_array()
    W_array = model.W_as_array()

    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:, np.random.randint(0, high=T)]
                X_b[:, t] = W_array[:, :, t].dot(Gamma_beta_l)\
                    .dot(model.Factors[:, t]) + d_temp

            # convert to dataframe to match input in _fit()
            X_b_df = pd.DataFrame(X_b, index=model.Chars,
                                  columns=model.Dates)
            Gamma, Factors = model._fit(X=X_b_df, W=model.W,
                                        minTol=minTol, maxIters=maxIters,
                                        gFac=model.gFac)

        except np.linalg.LinAlgError:
            warnings.warn("Encountered singularity in bootstrap iteration.\
                           Observation discarded.")
            pass

    # Compute and store Walpha_b
    Wbeta_l_b = Gamma[l, :].dot(Gamma[l, :].T)
    Wbeta_l_b = np.trace(Wbeta_l_b)
    return Wbeta_l_b


def _BS_Wdelta_sub(model, n, d, minTol=1e-4, maxIters=1000):
    L, T = model.L, model.T
    X_b = np.full((L, T), np.nan)
    np.random.seed(n)

    X_array = model.X_as_array()
    W_array = model.W_as_array()

    # Modify Gamma such that last column (Gamma_delta) = 0
    Gamma_delta = np.copy(model.Gamma)
    Gamma_delta[:, -1] = 0

    Gamma = None
    while Gamma is None:
        try:
            for t in range(T):
                d_temp = np.random.standard_t(5)
                d_temp *= d[:, np.random.randint(0, high=T)]
                X_b[:, t] = W_array[:, :, t].dot(Gamma_delta).dot(model.Factors[:, t]) + d_temp

            # convert to DataFrame to match _fit() input
            X_b_df = pd.DataFrame(X_b, index=model.Chars, columns=model.Dates)
            Gamma, Factors = model._fit(X=X_b_df, W=model.W,
                                        minTol=minTol, maxIters=maxIters,
                                        gFac=model.gFac)

        except np.linalg.LinAlgError:
            warnings.warn("Singularity in bootstrap iteration. Discarding draw.")
            pass

    # Compute Wdelta_b
    Wdelta_b = Gamma[:, -1].T.dot(Gamma[:, -1])
    return Wdelta_b

    # def _prep_metad(self):
    #     """handle mapping from different inputs type to consistent internal data

    #     Parameters
    #     ----------
    #     self : initialized IPCA object

    #     Returns
    #     -------
    #     indices : numpy array
    #         array containing the panel indices.  Should consist of two
    #         columns:

    #         - Column 1: entity id (i)
    #         - Column 2: time index (t)

    #         The panel may be unbalanced. The number of unique entities is
    #         n_samples, the number of unique dates is T, and the number of
    #         characteristics used as instruments is L.

    #     metad : dict
    #         contains metadata on inputs:

    #         dates : array-like
    #             unique dates in panel
    #         ids : array-like
    #             unique ids in panel
    #         chars : array-like
    #             labels for X chars/columns
    #         T : scalar
    #             number of time periods
    #         N : scalar
    #             number of ids
    #         L : scalar
    #             total number of characteristics
    #     """

    #     # extract numpy array and labels from multiindex
    #     indices = self.RZ.index.to_frame().values
    #     dates = np.unique(indices[:, 0])
    #     ids = np.unique(indices[:, 1])
    #     indices[:, 0] = np.unique(indices[:, 0], return_inverse=True)[1]
    #     indices[:, 1] = np.unique(indices[:, 1], return_inverse=True)[1]

    #     # init data dimensions
    #     L = self.L  # or self.L
    #     T = self.X.shape[1]
    #     N = len(np.unique(self.RZ.index.to_frame().values[:, 1]))

    #     # prep metadata
    #     metad = {}
    #     metad["dates"] = dates
    #     metad["ids"] = ids
    #     metad["chars"] = self.Chars
    #     metad["T"] = T
    #     metad["N"] = N
    #     metad["L"] = L

    #     return indices, metad
