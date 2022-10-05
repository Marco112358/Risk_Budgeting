import numpy as np

asset_sds = np.matrix([0.15, 0.2, 0.05])  # Asset standard deviations
asset_corrs = np.matrix([[1.0, 0.8, 0.2], [0.8, 1.0, -0.1], [0.2, -0.1, 1.0]])  # Correlation Matrix
b = np.matrix([0.4, 0.2, 0.4])  # target risk contributions (PCTRs)

asset_covars = np.multiply(asset_sds.transpose(), np.multiply(asset_corrs, asset_sds))

# f_total = lambda x: 0.5 * np.matmul(x, np.matmul(asset_covars, x.transpose()))[0, 0] \
#               - np.matmul(b, np.log(x).transpose())[0, 0]

# Functions used in newton algo
fn = lambda x: np.matmul(asset_covars, x.transpose()) - (b / x).transpose()
f_prime = lambda x: asset_covars + np.diagflat((b / np.multiply(x, x)))


# Newton Algo
def my_newton(f, df, x0, tol, iter_tot):
    iter_n = 0
    if np.sum(abs(f(x0))) <= tol:
        return x0, iter_n
    else:
        x_new = x0 - np.matmul(np.linalg.inv(df(x0)), f(x0)).transpose()
        for i in np.arange(0, iter_tot):
            if np.sum(abs(f(x_new))) > tol:
                x_new = x_new - np.matmul(np.linalg.inv(df(x_new)), f(x_new)).transpose()
                iter_n = i
            else:
                return x_new, iter_n
        return x_new, iter_n


# Initial Guess for Algo
x0_test = np.matrix([np.sum(asset_sds, 1)[0, 0] / b[0, 0],
                     np.sum(asset_sds, 1)[0, 0] / b[0, 1],
                     np.sum(asset_sds, 1)[0, 0] / b[0, 2]])

# Run Algo, get weights, print
x_out, itn = my_newton(fn, f_prime, x0_test, 0.000001, 10000)
wghts_out = x_out / np.matmul(np.matrix(np.ones(x_out.shape[1])), x_out.transpose())
print(wghts_out)
print(itn)
port_sd = np.sqrt(np.matmul(wghts_out, np.matmul(asset_covars, wghts_out.transpose())))[0, 0]
mctrs = np.matmul(asset_covars, wghts_out.transpose()) / port_sd
ctrs = np.multiply(mctrs, wghts_out.transpose())
pctrs = ctrs / port_sd
print(port_sd)
print(pctrs)
