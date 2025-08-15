import numpy as np
import astropy.units as u
from astropy.cosmology import FlatwCDM
from bs4 import BeautifulSoup

with open("C:/Users/yuvap/OneDrive/Documents/SURE Project/Project 2/Table 1. Open in new tab Data set... _ Oxford Academic.html", "r", encoding="utf-8") as file:
    html_content = file.read()

# Parse the HTML with BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Print to preview the structure (optional)

# Find all tables
tables = soup.find_all("table")

# If you know it's the first table
table = tables[0]

# Extract rows
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

# --- Dynesty ---

def prior_transform(theta): 
    alphaprime,betaprime,H0prime,Omega_mprime,wprime,sigma_prime = theta
    min1 = 33
    max1 = 35
    min2 = 4
    max2 = 5
    min3 = 60
    max3 = 110
    min4 = 0
    max4 = 1
    min5 = -3
    max5 = 0
    alpha = alphaprime * (max1 - min1) + min1
    beta = betaprime * (max2 - min2) + min2
    H0 = H0prime * (max3 - min3) + min3
    Omega_m = Omega_mprime * (max4 - min4) + min4
    w = wprime * (max5 - min5) + min5
    sigma_int=sigma_prime*(max4-min4)+min4
    return (alpha, beta, H0, Omega_m,w,sigma_int)


def loglikelihood_dynesty(theta):
    alpha,beta,H0,Omega_m,w,sigma_int = theta
    cosmo = FlatwCDM(H0*u.km/u.s/u.Mpc,Om0=Omega_m, w0=w)
    def Dl(z):
        return cosmo.luminosity_distance(z)
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
    return total

nlive = 1024
bound = 'multi'
ndims = 6
sample = 'unif'
tol = 0.1

from dynesty import DynamicNestedSampler
dsampler = DynamicNestedSampler(loglikelihood_dynesty,prior_transform, ndims, bound=bound, sample=sample) 
dsampler.run_nested(nlive_init=nlive,print_progress= True)
dres= dsampler.results 

# --- Print evidence ---
logZdynestydynamic=dres.logz[-1] 
logZerrdynestydynamic=dres.logzerr[-1] 
print("Dynamic: log(Z) = {} ± {}".format(logZdynestydynamic, logZerrdynestydynamic))