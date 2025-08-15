import numpy as np
import matplotlib.pyplot as plt
import emcee
import astropy.units as u
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
import matplotlib as mpl
from bs4 import BeautifulSoup
from astropy.constants import c 
from astropy.stats.info_theory import bayesian_info_criterion


# --- Data Loading ---
#14 newer HIIGx
with open("C:/Users/yuvap/OneDrive/Documents/SURE Project/Project 2/Table 1. Open in new tab Data set... _ Oxford Academic.html", "r", encoding="utf-8") as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, 'html.parser')

tables = soup.find_all("table")
table = tables[0]

rows = table.find_all("tr")
raw_data=[]
entry=[]
# Loop through rows
for row in rows:
    cells = row.find_all(["td", "th"])  # Both header and data
    raw_data.append([cell.get_text(strip=True) for cell in cells])
for i,row in enumerate(raw_data):
    if i in [0, 1, 2, 8, 13]:
        continue  # Skip header or label rows
    del row[4]
    del row[0]
    for item in row:
        item=item.replace('$','').replace('|','').strip()
        sub_item=item.split("\\pm")
        try:
            mean = float(sub_item[0].strip())
            error= float(sub_item[1].split('^')[0].strip())
        except:
            mean=None
            error=None
        entry.append((mean,error))
z_vals1=[]
zerr_vals1=[]
logsigma_vals1=[]
logsigmaerr_vals1=[]
logf_vals1=[]
logferr_vals1=[]
i=0
for i in range(0,len(entry),3):
    z,zerr=entry[i]
    z_vals1.append(z)
    zerr_vals1.append(zerr)
    logsigma,logsigmaerr=entry[i+1]
    logsigma_vals1.append(logsigma)
    logsigmaerr_vals1.append(logsigmaerr)
    logf,logferr=entry[i+2]
    logf_vals1.append(logf)
    logferr_vals1.append(logferr)
z_vals1 = np.array(z_vals1)
zerr_vals1 = np.array(zerr_vals1)
logsigma_vals1 = np.array(logsigma_vals1)
logsigmaerr_vals1 = np.array(logsigmaerr_vals1)
logf_vals1 = np.array(logf_vals1)
logferr_vals1 = np.array(logferr_vals1)

#181 HIIGx samples
data = np.loadtxt("C:/Users/yuvap/Downloads/stab1385_supplementary_file/supplementary_material_compressed/Table A3.csv",delimiter=',',usecols=(1,2,3,4,5),skiprows=1)
z_vals2=data[:,0]
zerr_vals2= np.full_like(z_vals2,1e-4)
logsigma_vals2=data[:,1]
logsigmaerr_vals2=data[:,2]
logf_vals2=data[:,3]
logferr_vals2=data[:,4]

#181+14 HIIGx samples
z_h2gx=np.concatenate((z_vals2,z_vals1))
zerr_h2gx=np.concatenate((zerr_vals2,zerr_vals1))
logsigma_h2gx=np.concatenate((logsigma_vals2,logsigma_vals1))
logsigmaerr_h2gx=np.concatenate((logsigmaerr_vals2,logsigmaerr_vals1))
logf_h2gx=np.concatenate((logf_vals2,logf_vals1))
logferr_h2gx=np.concatenate((logferr_vals2,logferr_vals1))

#36 anchor samples
import pandas as pd

# Load Excel file
df = pd.read_excel("C:/Users/yuvap/OneDrive/Documents/SURE Project/Project 2/anchor_cleaned.xlsx")

# Convert to NumPy array 
data1 = df.to_numpy()
logsigma_anc=data1[:,3]
logsigmaerr_anc=data1[:,4]
logL_anc=data1[:,5]
logLerr_anc=data1[:,6]
arr_loglike = []

# --- Bayesian Model ---
def loglikelihood(theta, z_h2gx,zerr_h2gx, logsigma_h2gx,logsigmaerr_h2gx,logf_h2gx,logferr_h2gx,logL_anc, logLerr_anc, logsigma_anc,logsigmaerr_anc):
    alpha,beta,H0,sigma_int = theta
    H0_val = H0 * u.km / u.s / u.Mpc
    def Dl(z):
        return (c*(1+z)*np.log(1+z)/H0_val).to(u.Mpc)
    Dl_err = (Dl(z_h2gx+zerr_h2gx).value-Dl(z_h2gx-zerr_h2gx).value)/2
    Dl_z = Dl(z_h2gx).value
    mu_th = 5*np.log10(Dl_z)+25 #change later 
    mu_obs=2.5*(beta*logsigma_h2gx+alpha-logf_h2gx)-100.2
    sigma_obs2=6.25*(sigma_int**2+beta**2*logsigmaerr_h2gx**2+logferr_h2gx**2)
   
    eps_h2gx2= sigma_obs2+(5*Dl_err/(np.log(10)*Dl_z))**2
    eps_anc2 = sigma_int**2+logLerr_anc**2 + beta**2*logsigmaerr_anc**2
    l_h2gx = -195*np.log(2*np.pi)/2-np.log(np.prod(eps_h2gx2))-np.sum((mu_obs-mu_th)**2/(2*eps_h2gx2))
    l_anc= -36*np.log(2*np.pi)/2-np.log(np.prod(eps_anc2))-np.sum((logL_anc-beta*logsigma_anc-alpha)**2/(2*eps_anc2))
    total = l_h2gx + l_anc
    arr_loglike.append(total)
    return total

def logprior(theta):
    alpha,beta,H0,sigma_int = theta
    
    if not (33 <= alpha <= 35 and 4 <= beta <= 5 and  60<= H0 <=110 and 0<=sigma_int<=1):
        return -np.inf
    return 0

def logposterior(theta, z_h2gx,zerr_h2gx, logsigma_h2gx,logsigmaerr_h2gx,logf_h2gx,logferr_h2gx,logL_anc, logLerr_anc, logsigma_anc,logsigmaerr_anc):
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood(theta, z_h2gx,zerr_h2gx, logsigma_h2gx,logsigmaerr_h2gx,logf_h2gx,logferr_h2gx,logL_anc, logLerr_anc, logsigma_anc,logsigmaerr_anc)

# --- MCMC Sampling ---
Nens = 100
Nburnin = 500
Nsamples = 500 

ndims = 4

alpha_init = np.random.uniform(33, 35, Nens) 
beta_init = np.random.uniform(4, 5, Nens) 
H0_init = np.random.uniform(60, 110, Nens)
sigma_int_init=np.random.uniform(0,1,Nens)
inisamples = np.array([alpha_init,beta_init, H0_init,sigma_int_init]).T


argslist = (z_h2gx,zerr_h2gx, logsigma_h2gx,logsigmaerr_h2gx,logf_h2gx,logferr_h2gx,logL_anc, logLerr_anc, logsigma_anc,logsigmaerr_anc)
sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)

sampler.run_mcmc(inisamples, Nsamples + Nburnin, progress=True)
alpha_acl, beta_acl, H0_acl,sigma_acl= sampler.get_autocorr_time(quiet=True)

samples_emcee = sampler.get_chain(flat=True, discard=Nburnin, thin=int(max([alpha_acl, beta_acl, H0_acl,sigma_acl])))

# Check convergence after sampling
print("Autocorrelation times:", sampler.get_autocorr_time(quiet=True))
print("Acceptance fraction:", np.mean(sampler.acceptance_fraction))
max_loglike=max(arr_loglike)
print(max_loglike)


# --- Plotting and Analysis ---
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'serif'

names = ['alpha', 'beta', 'h0','sigma_int']
labels = [
r'\alpha',
r'\beta',
r'\mathcal{H}_{\mathrm{0}}',
r'\mathcal{Sigma}_{\mathrm{int}}'
]
gdsamples = MCSamples(samples=samples_emcee, names=names, labels=labels)

g = plots.get_subplot_plotter()
g.settings.figure_legend_frame = False
g.settings.alpha_filled_add = 0.4
g.settings.title_limit_labels = False
g.settings.axis_marker_lw = 0.0
g.settings.linewidth_contour = 1.5
g.settings.num_plot_contours = 2

g.settings.lab_fontsize = 22       # axis labels
g.settings.axes_fontsize = 18      # tick numbers
g.settings.legend_fontsize = 18    # legend text

blue = '#1f77b4'
g.triangle_plot(
    [gdsamples],
    ['alpha', 'beta', 'h0','sigma_int'],
    filled=True,
    contour_colors=[blue],
    line_args=[{'ls': '-', 'color': blue}],
    markers=None
)

plt.show()


def median_and_cred(samples):
    median = np.mean(samples)
    low = np.percentile(samples, 16)
    high = np.percentile(samples, 84)
    return median, high - median, median - low

alpha, alpha_hi, alpha_lo = median_and_cred(samples_emcee[:, 0])
beta, beta_hi, beta_lo = median_and_cred(samples_emcee[:, 1])
h0, h0_hi, h0_lo = median_and_cred(samples_emcee[:, 2])
sigma_int, sigma_int_hi, sigma_int_lo = median_and_cred(samples_emcee[:, 3])

print(f"alpha: {alpha:.2f} +{alpha_hi:.2f} -{alpha_lo:.2f} ")
print(f"beta: {beta:.2f} +{beta_hi:.2f} -{beta_lo:.2f}")
print(f"h0: {h0:.2f} +{h0_hi:.2f} -{h0_lo:.2f}")
print(f"sigma_int: {sigma_int:.2f} +{sigma_int_hi:.2f} -{sigma_int_lo:.2f}")

n_params = ndims
n_samples= len(z_h2gx) + len(logL_anc)
bic = bayesian_info_criterion(max_loglike, n_params, n_samples)
print(bic)
