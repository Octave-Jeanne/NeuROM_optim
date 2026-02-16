import ngsolve as ngs
from netgen.geom2d import SplineGeometry
import matplotlib.pyplot as plt
import numpy as np

def plotLines(pnts, curves, fac = 5e-3):
    for i in range(len(curves)):
        curve = curves[i][0]
        if curve[0] == "line" :
            if type(pnts[curve[1]]) is tuple:
                p1 = pnts[curve[1]]
                p2 = pnts[curve[2]]
            else:
                p1 = pnts[curve[1]][0]
                p2 = pnts[curve[2]][0]
            plt.plot([p1[0],p2[0]], [p1[1],p2[1]])
            d = np.array([p2[0]-p1[0], p2[1]-p1[1]])
            d = d/np.linalg.norm(d)
            dLeft= np.array([[0,-1],[1,0]]) @ d
            pavg=((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
            plt.text(pavg[0] + fac*dLeft[0], pavg[1] + fac*dLeft[1], str(curves[i][1]["leftdomain"]), color = "r")
            plt.text(pavg[0] - fac*dLeft[0], pavg[1] - fac*dLeft[1], str(curves[i][1]["rightdomain"]), color = "r")
        elif curve[0] == "spline3":
            if type(pnts[curve[1]]) is tuple:
                p1 = pnts[curve[1]]
                p2 = pnts[curve[2]]
                p3 = pnts[curve[3]]
            else:
                p1 = pnts[curve[1]][0]
                p2 = pnts[curve[2]][0]
                p3 = pnts[curve[3]][0]
            plt.plot([p1[0],p2[0],p3[0]], [p1[1],p2[1],p3[1]])
            d = np.array([p3[0]-p1[0], p3[1]-p1[1]])
            d = d/np.linalg.norm(d)
            dLeft= np.array([[0,-1],[1,0]]) @ d
            plt.text(p2[0] + fac*dLeft[0], p2[1] + fac*dLeft[1], str(curves[i][1]["leftdomain"]), color = "r")
            plt.text(p2[0] - fac*dLeft[0], p2[1] - fac*dLeft[1], str(curves[i][1]["rightdomain"]), color = "r")
    plt.axis("equal")

def plotPoints(pnts):
    for i, xy in enumerate(pnts):
        if type(xy) is not tuple:
            xy = xy[0]
        plt.scatter(xy[0],xy[1],c = "r",marker="+")
        plt.text(xy[0],xy[1],"p"+str(i))
    plt.axis("equal")


cm = 1e-2

def meshInductance(L : float = 10*cm,
                   h : float= 3*cm,
                   l : float=1*cm,
                   a : float=3*cm,
                   meshsize_factor : float = 1,
                   name_Out : str = "out",
                   name_xP : str = "xP",
                   name_xM : str = "xM",
                   name_domainCore : str = "core",
                   name_domainAir  : str= "air",
                   name_domainCoilP : str = "coilP",
                   name_domainCoilN : str = "coilN",
                   
                   ):   
    """ generate the mesh of the inductance"""

    domainAir = 1
    domainCore = 2
    domainCoilP1 = 3
    domainCoilP2 = 4
    domainCoilN1 = 5
    domainCoilN2 = 6

    e=L-2*a-2*l
    b=(L-h)/2


    dh_corner = L/50 * meshsize_factor
    dh_domain = L/10 * meshsize_factor
    dh_coil = max(dh_domain*5, L/30)


    ############################
    ## Define the point geometry
    ############################

    ## Domain
    pnts = [[(0, 0), {"maxh" : dh_domain}],
            [(L, 0), {"maxh" : dh_domain}],
            [(L, L), {"maxh" : dh_domain}],
            [(0, L), {"maxh" : dh_domain}]
    ]

    ## Coil 1
    pnts += [[(a, b), {"maxh" : dh_corner}],
            [(a+l, b), {"maxh" : dh_corner}],
            [(a+l, b+h), {"maxh" : dh_corner}],
            [(a, b+h), {"maxh" : dh_corner}]
    ]

    ## Coil 2
    pnts += [[(a+l+e, b), {"maxh" : dh_corner}],
            [(a+e+2*l, b), {"maxh" : dh_corner}],
            [(a+e+2*l, b+h), {"maxh" : dh_corner}],
            [(a+e+l, b+h), {"maxh" : dh_corner}]
    ]


    pnts += [[(a-e/2, b-e/2), {"maxh" : dh_domain}],
            [(a+2*(e+l)-e/2,  b-e/2), {"maxh" : dh_domain}],
            [(a+2*(e+l)-e/2, b+h+e/2), {"maxh" : dh_domain}],
            [(a-e/2, b+h+e/2), {"maxh" : dh_domain}],
    ]

    ## Vector Potential Positive Source
    pnts += [[(a+l/2, b+h/2), {"maxh" : dh_coil, "name" : name_xP}]]

    ## Vector Potential Negative Source
    pnts += [[(L-a-l/2, b+h/2), {"maxh" : dh_coil, "name" : name_xM}]]



    #####################################
    ## Define lines connecting the points
    #####################################

    lines = [[["line",0,1], {"leftdomain": domainAir, "rightdomain": 0, "maxh": dh_domain, "bc": name_Out}],
            [["line",1,2], { "leftdomain": domainAir, "rightdomain": 0, "maxh": dh_domain, "bc": name_Out}],
            [["line",2,3], { "leftdomain": domainAir, "rightdomain": 0, "maxh": dh_domain, "bc": name_Out}],
            [["line",3,0], { "leftdomain": domainAir, "rightdomain": 0, "maxh": dh_domain, "bc": name_Out}],
    ]

    ## Coil 1
    lines += [[["line",4,5], {"leftdomain": domainCoilP1, "rightdomain": domainCore, "maxh": dh_coil}],
            [["line",5,6], { "leftdomain": domainCoilP1, "rightdomain": domainCore, "maxh": dh_coil}],
            [["line",6,7], { "leftdomain": domainCoilP2, "rightdomain": domainCore, "maxh": dh_coil}],
            [["line",7,4], { "leftdomain": domainCoilP2, "rightdomain": domainCore, "maxh": dh_coil}],
    ]

    lines += [[["line",4,16], {"leftdomain": domainCoilP2, "rightdomain": domainCoilP1, "maxh": dh_coil}],
            [["line",16,6], { "leftdomain": domainCoilP2, "rightdomain": domainCoilP1, "maxh": dh_coil}],
    ]

    ## Coil 2
    lines += [[["line",8,9], {"leftdomain": domainCoilN1, "rightdomain": domainCore, "maxh": dh_coil}],
            [["line",9,10], { "leftdomain": domainCoilN1, "rightdomain": domainCore, "maxh": dh_coil}],
            [["line",10,11], { "leftdomain": domainCoilN2, "rightdomain": domainCore, "maxh": dh_coil}],
            [["line",11,8], { "leftdomain": domainCoilN2, "rightdomain": domainCore, "maxh": dh_coil}],
    ]

    lines += [[["line",8,17], {"leftdomain": domainCoilN2, "rightdomain": domainCoilN1, "maxh": dh_coil}],
            [["line",17,10], { "leftdomain": domainCoilN2, "rightdomain": domainCoilN1, "maxh": dh_coil}],
    ]

    ## out
    lines += [[["line",12,13], {"leftdomain": domainCore, "rightdomain": domainAir}],
            [["line",13,14], { "leftdomain": domainCore, "rightdomain": domainAir}],
            [["line",14,15], { "leftdomain": domainCore, "rightdomain": domainAir}],
            [["line",15,12], { "leftdomain": domainCore, "rightdomain": domainAir}],
    ]

    geo = SplineGeometry()
    for pnt, props in pnts:
        geo.AppendPoint(*pnt, **props)

    for line, props in lines:
        geo.Append(line, **props)

    geo.SetMaterial(domainCore, name_domainCore)
    geo.SetMaterial(domainAir, name_domainAir)
    geo.SetMaterial(domainCoilP1, name_domainCoilP)
    geo.SetMaterial(domainCoilN1, name_domainCoilN)
    geo.SetMaterial(domainCoilP2, name_domainCoilP)
    geo.SetMaterial(domainCoilN2, name_domainCoilN)

    geo.SetDomainMaxH(domainCore, dh_domain)
    geo.SetDomainMaxH(domainAir, dh_domain)
    geo.SetDomainMaxH(domainCoilN1, dh_coil)
    geo.SetDomainMaxH(domainCoilN2, dh_coil)
    geo.SetDomainMaxH(domainCoilP1, dh_coil)
    geo.SetDomainMaxH(domainCoilP1, dh_coil)

    mesh = ngs.Mesh(geo.GenerateMesh())
    return mesh



import ngsolve as ngs
from numpy.linalg import norm
from numpy import isnan
from ngsolve.webgui import Draw
from time import time

def solve(fes : ngs.FESpace,                                                        # finite element space
          residual : callable,                                                      # residual(state, test)
          residual_derivative : callable = None,                                    # residual_derivative(state, trial, test) (optional)
          initial_guess :  ngs.GridFunction |  ngs.CoefficientFunction = ngs.CF(0), # initial guess (CoefficientFunction or GridFunction)
          # Inspection parameters
          verbosity : int = 1,                                                      # verbosity level (0 - silent to 3 - detailed)
          draw : bool = False,                                                      # draw intermediate solutions
          # Newton parameters
          maxit_newton : int = 50,             # maximum number of Newton outer iterations
          tol_dec : float = 1e-8,              # (absolute) tolerance on Newton decrement : sqrt( < residual(uOld), du > )
          tol_res : float = 1e-8,              # (absolute) tolerance on residual 
          rtol_res : float = 1e-10,            # relative tolerance on the residual between 2 iterations (to save 1 useless iteration in case of linear problem)
          # Line search parameters
          linesearch : bool = True,            # flag to enable line search (recommended)
          maxit_linesearch : int = 20,         # maximum iteration number within the line search
          minstep_linesearch : float = 1e-12,  # minimum step size allowed in the line search 
          armijo_linesearch : float = 0.1,     # Armijo coefficient in [0, 1) such that |residual(u-step*du)|² < residual²(u) - armijo_linesearch*step*(|residual(u)|²)'(du)
          step_factor_linesearch : float = 0.5,# step size reduction factor in (0, 1) to reduce the step if too big 
          # Solver
          solver = "sparsecholesky"            # solver type
          ) -> dict:
    
    """
    Solve a nonlinear PDE using Newton method.

    Parameters
    ----------
    fes : ngs.FESpace
        The finite element space.

    residual : callable
        Function taking (state, test function) and returning the residual form.

    residual_derivative : callable, optional
        Function taking (state, trial function, test function) and returning
        the bilinear form of the derivative. If None, symbolic differentiation is used.

    initial_guess : ngs.GridFunction or ngs.CoefficientFunction, optional
        Initial solution guess. Default is 0 everywhere.

    verbosity : int, optional
        Verbosity level (0 = silent, 3 = very detailed). Default is 1.

    draw : bool, optional
        Whether to visualize intermediate results. Default is False.

    maxit_newton : int, optional
        Maximum number of Newton iterations. Default is 50.

    tol : float, optional
        Absolute convergence tolerance on the Newton decrement. Default is 1e-8.

    rtol_res : float, optional
        relative tolerance on the residual between 2 iterations (to save 1 useless 
        iteration in case of linear problem). Default is 1e-10.

    linesearch : bool, optional
        Enable or disable line search. Default is True.

    maxit_linesearch : int, optional
        Maximum number of line search iterations. Default is 20.

    minstep_linesearch : float, optional
        Minimum allowable step size during line search. Default is 1e-12.

    armijo_linesearch : float, optional
        Armijo condition coefficient for line search. Default is 0.1.

    step_factor_linesearch : float, optional
        Multiplicative factor to reduce step size in line search. Default is 0.3.
    
    solver : str, optional
        Type of solver ("sparsecholesky", "pardiso",...). Default is "sparsecholesky".

    Returns
    -------
    results : dict
        A dictionary containing:
        - "solution" : final solution (ngs.GridFunction)
        - "status" : integer code indicating termination reason (see below)
        - "linear_detected" : True if linear problem detected early
        - "iteration" : number of Newton iterations performed
        - "last_inverse" : last tangent matrix decomposition (for reuse or debugging)
        - "residual" : list of residual norms per iteration
        - "decrement" : list of Newton decrement values per iteration
        - "wall_time" : total computation time in seconds

    Status codes:
    -------------
    0 : ✅ SUCCESS — Newton converged successfully.
    1 : ❌ FAILURE — Maximum number of Newton iterations reached.
    2 : ❌ FAILURE — Line search failed: minimum step size reached.
    3 : ❌ FAILURE — Line search failed: max number of iterations reached.
    4 : ❌ FAILURE — NaN encountered in the residual (after line search if enabled).
    """

    # I) Initialization

    tStart = time()
    if verbosity >= 2 : print(f"-------------------- START NEWTON ---------------------")
    if verbosity >= 3 : print(f"Initializing  ..... ", end = "")
    du, v = fes.TnT()
    res2 = lambda sol : (norm(ngs.LinearForm(residual(sol, v)).Assemble().vec.FV().NumPy()[fes.FreeDofs()]))**2
    state, state_linesearch, descent = ngs.GridFunction(fes), ngs.GridFunction(fes), ngs.GridFunction(fes)
    if type(initial_guess) is ngs.GridFunction and initial_guess.space == fes:
        state.vec.data = initial_guess.vec.data
    else : state.Set(initial_guess)
    counter_newton = 0
    decrement_list = []
    res2_state = res2(state)
    residual_list = [ngs.sqrt(res2_state)]
    status = 0
    linear = False

    if draw : scene = Draw(state)
    if verbosity >= 3 : print(f"done ({(time()-tStart) * 1000 :.2f} ms).")
    if verbosity >= 2 : print(f"Initial residual : {residual_list[-1] :.5e}")
    if verbosity >= 3 : print(f"Start loop  ....... ")

    # II) Loop

    while 1:
        counter_newton += 1
        if verbosity >= 2 : print(f" It {counter_newton} -------------------------------------------------")

        # a) Assembly
        tStartAssembly = time()
        if verbosity >= 3 : print(f" - Assembly ....... ", end = "")
        res = ngs.LinearForm(residual(state, v)).Assemble()
        if residual_derivative is None : # symbolic differentiation (recommended)
            dres = ngs.BilinearForm(residual(du, v))
            dres.AssembleLinearization(state.vec)
        else :
            dres = ngs.BilinearForm(residual_derivative(state, du, v)).Assemble()
        if verbosity >= 3 : print(f"done ({(time()-tStartAssembly) * 1000 :.2f} ms).")
        tStartSolve = time()
        if verbosity >= 3 : print(f" - Solve .......... ", end = "")
        Kinv  = dres.mat.Inverse(freedofs=fes.FreeDofs(), inverse = solver) 
        descent.vec.data = Kinv * res.vec
        if verbosity >= 3 : print(f"done ({(time()-tStartSolve) * 1000 :.2f} ms).")

        decrement_list.append(ngs.sqrt(abs(ngs.Integrate(residual(state, descent), fes.mesh))))

        # b) Line search
        if linesearch :
            tStartLineSearch = time()
            if verbosity >= 2 : print(f" - Line search .... ")
            step = 1.
            counter_linesearch = 0
            state_linesearch.vec.data = state.vec - step * descent.vec
            res2_ls = res2(state_linesearch)
            if verbosity >= 2 : print(f"   it {counter_linesearch} : ||residual|| = {ngs.sqrt(res2_ls) :.5e} | step = {step :.2e}")

            while not res2_ls < (1-2*armijo_linesearch*step) * res2_state : # enter the line search even if the residual is nan
                step *= step_factor_linesearch
                state_linesearch.vec.data = state.vec - step * descent.vec
                res2_ls = res2(state_linesearch)
                counter_linesearch += 1
                if verbosity >= 2 : print(f"   it {counter_linesearch} : ||residual|| = {ngs.sqrt(res2_ls) :.5e} | step = {step :.2e}")

                if counter_linesearch >= maxit_linesearch:
                    if verbosity >= 1 : print(f"❌ FAILURE: maximal number of line search iterations reached !!")
                    status = 3
                    break 

                if step < minstep_linesearch:
                    if verbosity >= 1 : print(f"❌ FAILURE: minimal line search step reached !!")
                    status = 2
                    break 
            
            if verbosity >= 3 : print(f" - Line search done ({(time()-tStartLineSearch) * 1000 :.2f} ms).")

            if not status:
                state.vec.data = state_linesearch.vec
                            
        else :
            state.vec.data = state.vec - descent.vec

        if isnan(res2_state):
            status = 4
            if verbosity >= 1 : 
                print(f"❌ FAILURE: NaN detected ", end = "")
                if linesearch : print("after line search ", end = "")
            print("!!")
            break

        if status:
            break

        # c) stop criterion
        res2_state = res2(state)
        residual_list.append(ngs.sqrt(res2_state))
        
        if verbosity >= 2 : print(f" - Conv : ||residual|| = {residual_list[-1]:.5e} | decr = {decrement_list[-1] :.5e}")
        if draw : scene.Redraw(state)
        if verbosity >= 3 : print(f" - Newton iteration done ({(time()-tStartAssembly) * 1000 :.2f} ms).")


        if residual_list[-1] / residual_list[-2] < rtol_res:
            if verbosity >= 2 : print(f"Stop because linear problem detected.")
            linear = True
            break
        
        if decrement_list[-1] < tol_dec : 
            if verbosity >= 2 : print(f"Stop because decrement is lower than tol_dec.")
            break

        if residual_list[-1] < tol_res : 
            if verbosity >= 2 : print(f"Stop because residual is lower than tol_res.")
            break

        if counter_newton >= maxit_newton: 
            if verbosity >= 1 : print(f"❌ FAILURE: maximum number of Newton iterations reached !!")
            status = 1
            break
    
    # III) Export results

    if verbosity >=2 and not status : 
        print(f"-------------------------------------------------------")  
        print(f" ✅ SUCCESS: Newton has converged in {counter_newton} iteration", end = "")
        if  counter_newton > 1 : print("s.")
        else : print(".") 
    if verbosity >=2 :  print(f" Total wall time: {(time() - tStart) :.2f} s.")
    results = {"solution" : state, 
               "status" : status, 
               "linear_detected" : linear,
               "iteration": counter_newton, 
               "last_inverse" : Kinv, 
               "residual" : residual_list,
               "decrement": decrement_list,
               "wall_time" : time() - tStart}
    if verbosity >=2 : print(f" --------------------- END NEWTON --------------------- ")  
    return results

###############################################################################################################################
# Tests

if __name__ == "__main__" : # simple tests

    from ngsolve.webgui import Draw
    mesh = meshInductance(meshsize_factor = 1)
    Draw(mesh)
    print(mesh.GetBBoundaries())





