'''
Note: This Python implementation od Scaled Conjugate Gradient Descent has been adopted from
       Prof. Charles Anderson, Department of Comp. Science, CSU. I've slightly modified the code.
'''
from copy import copy
import numpy as np
import sys
from math import sqrt, ceil
from collections import deque
import pdb

floatPrecision = sys.float_info.epsilon

def scaledconjugategradient(x, f,gradf,*fargs, **params):
    """scg:
    Example:
    def parabola(x,xmin,s):
        d = x - xmin
        return np.dot( np.dot(d.T, s), d)
    def parabolaGrad(x,xmin,s):
        d = x - xmin
        return 2 * np.dot(s, d)
    center = np.array([5,5])
    S = np.array([[5,4],[4,5]])
    firstx = np.array([-1.0,2.0])
    r = scg(firstx, parabola, parabolaGrad, center, S,
            xPrecision=0.001, nIterations=1000)
    print('Optimal: point',r[0],'f',r[1])"""

    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision",0) 
    fPrecision = params.pop("fPrecision",0)
    xtracep = params.pop("xtracep",False)
    ftracep = params.pop("ftracep",False)
    verbose = params.pop("verbose",False)
    iterationVariable = params.pop("iterationVariable",None)

### from Nabney's netlab matlab library
  
    nvars = len(x)
    sigma0 = 1.0e-6
    fold = f(x, *fargs)
    fnow = fold
    gradnew = gradf(x, *fargs)
    gradold = copy(gradnew)
    d = -gradnew                # Initial search direction.
    success = True              # Force calculation of directional derivs.
    nsuccess = 0                # nsuccess counts number of successes.
    beta = 1.0e-6               # Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15           # Lower bound on scale.
    betamax = 1.0e20            # Upper bound on scale.
    j = 1               # j counts number of iterations.
    tmpW = []
    trainingMSE = []
    validationMSE = []
    allW=[]
    
    if xtracep:
        xtrace = np.zeros((nIterations+1,len(x)))
        xtrace[0,:] = x
    else:
        xtrace = None

    if ftracep:
        ftrace = np.zeros(nIterations+1)
        ftrace[0] = fold
    else:
        ftrace = None
        
    ### Main optimization loop.
    #pdb.set_trace()
    while j <= nIterations:

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if np.isnan(mu): print("mu is NaN")
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            '''
            if False and kappa < floatPrecision:
                print( kappa)
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,'ftrace':ftrace[:j] if ftracep else None,
                        'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,'weights':allW,'bestItr':None,
                        'reason':"limit on machine precision"}
            '''
            if kappa==0:
                print('Terminating as kappa is zero. Can\'t proceed further.')
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,'ftrace':ftrace[:j] if ftracep else None,
                        'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,'weights':allW,'bestItr':None,
                        'reason':"limit on machine precision"}                
            sigma = sigma0/sqrt(kappa)
            xplus = x + sigma * d
            gplus = gradf(xplus, *fargs)
            theta = np.dot(d, gplus - gradnew)/sigma

        ## Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if np.isnan(delta): print("delta is NaN")
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta/kappa
        alpha = -mu/delta
        
        ## Calculate the comparison ratio.
        xnew = x + alpha * d
        fnew = f(xnew, *fargs)
        #print('fnew:',fnew)
        Delta = 2 * (fnew - fold) / (alpha*mu)
        # if np.isnan(Delta):
        #     pdb.set_trace()
        if not np.isnan(Delta) and Delta  >= 0:
            success = True
            nsuccess += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold
        #pdb.set_trace()
        #if verbose and (j % ceil(0.25*nIterations) == 0):
        if verbose:
            print("SCG: Iteration",j,"fValue",evalFunc(fnow),"Scale",beta)
            
        if xtracep:
            xtrace[j,:] = x
        if ftracep:
            ftrace[j] = fnow
            
        if success:
            ## Test for termination

            if max(abs(alpha*d)) < xPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,
                        'weights':allW,'bestItr':None,'reason':"limit on x Precision"}
            elif abs(fnew-fold) < fPrecision:
                return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None,
                        'ftrace':ftrace[:j] if ftracep else None,'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,
                        'weights':allW,'bestItr':None,'reason':"limit on f Precision"}
            else:
                ## Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *fargs)
                ## If the gradient is zero then we are done.
                if np.dot(gradnew, gradnew) == 0:
                    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],
                            'weightNorm':tmpW,'trMSE':trainingMSE,'valMSE':validationMSE,'weights':allW,'bestItr':None,
                            'reason':"zero gradient"}

        ## Adjust beta according to comparison ratio.
        if np.isnan(Delta) or Delta < 0.25:
            beta = min(4.0*beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5*beta, betamin)

        ## Update search direction using Polak-Ribiere formula, or re-start 
        ## in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d = -gradnew
            nsuccess = 0
        elif success:
            gamma = np.dot(gradold - gradnew, gradnew/mu)
            d = gamma * d - gradnew

        j += 1

    return {'x':x, 'f':fnow, 'nIterations':j, 'xtrace':xtrace[:j,:] if xtracep else None, 'ftrace':ftrace[:j],'reason':"did not converge"}
