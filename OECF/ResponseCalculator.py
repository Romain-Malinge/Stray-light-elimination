import scipy.interpolate
import cvxpy
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr

from scipy.interpolate import BSpline

# Import MATLAB if possible
try:
    import matlab.engine
except RuntimeError:
    print("Warning: Matlab license not found. MATLAB mode will fail.")
except ImportError:
    print("Warning: matlab.engine not found. MATLAB mode will fail.")


class ResponseCalculator:
    def __init__(self):
        pass
    
    def gsolve_matlab(self, Z, B, l, w):
        print("\nStarting MATLAB Engine...")
        eng = matlab.engine.start_matlab()
        
        Z_mat = matlab.double(Z.tolist())
        B_mat = matlab.double(np.array(B).reshape(-1,1).tolist())
        l_mat = float(l)
        w_mat = matlab.double(np.array(w).tolist())
        n_mat = float(self.__bit_per_sample)
        
        print("Running gsolve.m in MATLAB...")
        g, lE = eng.gsolve(Z_mat, B_mat, l_mat, w_mat, n_mat, nargout=2)
        
        eng.quit()
        return np.array(g).flatten(), np.array(lE).flatten()

    def gsolve_python(self, Z, B, l, w):
        n = self.__bit_per_sample 
        Z1 = np.size(Z, 0) 
        Z2 = np.size(Z, 1)
        
        if self.sparse_method:
            A = lil_matrix((Z1 * Z2 + n + 1, n + Z1), dtype=np.float32)
        else: 
            A = np.zeros((Z1 * Z2 + n + 1, n + Z1), dtype=np.float32)
        
        b = np.zeros(np.size(A, 0), dtype=np.float32)
        
        k = 0
        for i in range(Z1):
            for j in range(Z2):
                z = int(Z[i, j])
                wij = w[z]
                A[k, z] = wij
                A[k, n + i] = -wij
                b[k] = wij * B[j]
                k += 1
        
        A[k, n//2] = 1
        k += 1

        for i in range(n-2):
            A[k, i]   =    l*w[i+1]
            A[k, i+1] = -2*l*w[i+1]
            A[k, i+2] =    l*w[i+1]
            k += 1
        
        print(np.shape(A))
        print(np.shape(b))
        
        # Solve the system using SVD
        
        # pseudoA = np.linalg.pinv(A.toarray())
        if self.sparse_method:
            print("Using sparse solver...")
            A = A.tocsr()
            x, istop, itn = lsqr(A, b)[:3]
            #print (istop, itn)
            g = x[:n]
            lE = x[n:]
        else:
            print("Using dense solver, may take a while...")
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            g = x[:n].flatten()
            lE = x[n:].flatten()
        

        return g, lE
        
    def derivative(self, vals):
        n_vals = np.shape(vals)[-1]
        k = np.reciprocal(np.diff(1.0 * vals))
        Dx = scipy.sparse.diags_array([-k, k], offsets = [0, 1], shape = (n_vals-1, n_vals))
        return Dx

    def data_attachement(self,grey, times):
        (n_pix, n_time)= np.shape(grey)
        vals, vals_indices = np.unique(grey, return_inverse=True)
        vals_indices_flat = np.reshape(vals_indices, (-1,), order='C')
        n_vals = np.shape(vals)[-1]
        A_model = scipy.sparse.kron(scipy.sparse.eye(n_pix), -times[:,None])
        A_func = scipy.sparse.coo_array((np.ones(n_pix*n_time),(np.arange(n_pix*n_time), vals_indices_flat)), shape=(n_pix*n_time, n_vals))
        A =  scipy.sparse.hstack([A_model, A_func])
        return A, vals

    def inv_acp(self, responses, n_params, vals, eps = 0.001):
        sample = (vals - np.min(vals))/(np.max(vals)- np.min(vals))
        irrads = np.linspace(0, 1, responses.shape[-1])
        regular_responses = (np.maximum.accumulate(responses, axis=-1) + irrads * eps)/(1+eps)
        inv_responses = np.asarray([scipy.interpolate.PchipInterpolator(regular_response, irrads)(sample) for regular_response in regular_responses])
        mean_inv_response = np.mean(inv_responses, axis=0)
        _, s, acp = scipy.sparse.linalg.svds(inv_responses - mean_inv_response, k=n_params)
        base = np.hstack([acp[np.argsort(s), :].T, mean_inv_response[:, None]])
        return base
    
    def base_polynomiale(self, n_params, vals):
        # 1. Normalisation des entrées entre 0 et 1
        # C'est crucial pour que les puissances élevées n'explosent pas
        x_norm = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
        
        # 2. Construction de la matrice de Vandermonde
        # colonnes : [x^n, x^(n-1), ..., x^1, 1]    
        # n_params définit ici le degré du polynôme
        base = np.vander(x_norm, N=n_params + 1, increasing=True)
        
        return base
    
    def base_chebyshev(self, n_params, vals):
        x_norm = 2 * (vals - np.min(vals)) / (np.max(vals) - np.min(vals)) - 1 # Range [-1, 1]
        base = chebvander(x_norm, n_params)
        return base

    def base_lagrange(self, n_params, vals):

        x_norm = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
        nodes = np.linspace(0, 1, n_params + 1)

        base = np.zeros((len(vals), n_params + 1))

        for j in range(n_params + 1):
            Lj = np.ones_like(x_norm)
            for k in range(n_params + 1):
                if k != j:
                    Lj *= (x_norm - nodes[k]) / (nodes[j] - nodes[k])
            base[:, j] = Lj

        return base
    

    def base_bspline(self, n_params, vals, degree=3):

        # normalisation [0,1]
        x = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))

        # nombre de fonctions de base
        n_basis = n_params

        # nombre de nœuds
        n_knots = n_basis + degree + 1

        # nœuds internes uniformes
        knots = np.linspace(0, 1, n_knots - 2*degree)

        # clamped spline (multiplicité aux bords)
        knots = np.concatenate((
            np.zeros(degree),
            knots,
            np.ones(degree)
        ))

        # matrice de base
        base = np.zeros((len(x), n_basis))

        for i in range(n_basis):
            c = np.zeros(n_basis)
            c[i] = 1
            spline = BSpline(knots, c, degree)
            base[:, i] = spline(x)

        return base

    def get_response_params(self, grey, times, responses, n_params=10):
        """
        grey (Z) : ndarray, shape (n_pix, n_time)
        times (B) : ndarray, shape (n_time,)
        responses : ndarray, shape (n_responses, n_ech)
        n_params : int
        """
        grey = grey.astype(np.uint16)
        (n_pix, n_time)= np.shape(grey)
        A, vals = self.data_attachement(grey, times)
        n_vals = np.shape(vals)[-1]
        
        # base = self.inv_acp(responses, n_params, vals) # tester base polynomiale
        base = self.base_bspline(n_params, vals,3)
        B = scipy.sparse.block_array([[scipy.sparse.eye(n_pix), None], [None, base]])
        
        S = scipy.sparse.hstack([scipy.sparse.csr_array((n_vals, n_pix)), scipy.sparse.eye(n_vals)])
        Dx = self.derivative(vals)
        # x = cvxpy.Variable(n_pix + n_params + 1)
        x = cvxpy.Variable(n_pix + base.shape[1]) #bspline
        objective = cvxpy.Minimize(cvxpy.sum_squares(A @ B @ x))
        constraints = [Dx @ S @ B @ x >= 0.0, x[-1] == 1]
        #constraints = [Dx @ S @ B @ x >= 0.0, (S @ B @ x)[-1] == 0]
        prob = cvxpy.Problem(objective, constraints)
        prob.solve(solver=cvxpy.OSQP, verbose=True)
        E, inv_response = x.value[:n_pix], S @ B @ x.value
        
        irrads = np.linspace(0, 1, inv_response.shape[-1])
        eps = 0.001
        regular_inv_response = (np.maximum.accumulate(inv_response, axis=-1) + irrads * eps)/(1+eps)
        
        complete_inv_response = scipy.interpolate.PchipInterpolator(vals, regular_inv_response, extrapolate=False)(np.arange(np.iinfo(grey.dtype).max+1))
        complete_response = scipy.interpolate.PchipInterpolator(regular_inv_response, vals, extrapolate=False) #evaluer avec complete_response(x)
        
        complete_response = complete_response(np.linspace(0, 1, 1000))
        
        return E, complete_inv_response, complete_response