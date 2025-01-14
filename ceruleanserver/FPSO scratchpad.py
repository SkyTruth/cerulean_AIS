#%%
# FPSO Scractchpad

ssvids_all = ['205753000', '224805000', '232005797', '232011493', '235010630', '235101762', '235107000', '235599000', '247256000', '247311500', '249320000', '251263440', '251826940', '257069000', '257161000', '257804000', '258277000', '259644000', '261001203', '265588700', '273335620', '273387920', '273414140', '306239000', '306483000', '306811000', '308108000', '308297000', '308384000', '308419000', '308424000', '308496000', '308608000', '308626000', '308788000', '308944000', '309227000', '309275000', '309499000', '309669000', '309745000', '309776000', '309817000', '309864000', '309999000', '310594000', '311000019', '311000116', '311000132', '311000143', '311000168', '311000245', '311000248', '311000254', '311000270', '311000289', '311000307', '311000320', '311000563', '311000768', '311000797', '311000852', '311000917', '311000945', '311001008', '311001017', '311010200', '311014200', '311021700', '311027200', '311038200', '311048800', '311050100', '311050200', '311062000', '311070500', '311075500', '311076500', '311304000', '311426000', '311483000', '311488000', '311711000', '311875000', '316317000', '345050029', '345070329', '345070383', '351237000', '351279000', '352001052', '352001345', '352001554', '352299000', '352325000', '352511000', '352785000', '352857000', '352916000', '352978133', '353044000', '353558000', '353713000', '353874000', '354446000', '354894000', '354903000', '355338000', '355626000', '355631000', '355866000', '356055000', '356058000', '356790000', '356819000', '357765000', '370823000', '371817000', '372131000', '372750000', '373695000', '373752000', '374027000', '375114000', '375565000', '376465000', '412301170', '412301540', '412464450', '412477590', '413021000', '413275000', '413300030', '413356530', '414030000', '419001524', '503000059', '503511000', '503576000', '503597000', '503694000', '503700000', '511100701', '525005313', '525007035', '525008063', '525011166', '525017099', '525019531', '525019569', '525020250', '525021310', '525023187', '525100350', '525119090', '525157800', '533000241', '533001120', '533062800', '533111111', '533130268', '533130543', '533130934', '533150076', '533180105', '533189000', '533192000', '533368000', '533371000', '538001311', '538001905', '538002131', '538002388', '538002512', '538003021', '538003076', '538003137', '538003426', '538003496', '538003899', '538004129', '538005076', '538005090', '538005508', '538005509', '538005621', '538006306', '538006484', '538006746', '538006869', '538007000', '538008397', '538008638', '563009300', '563030000', '563085600', '563085800', '563086300', '563086600', '563094900', '563095100', '564070000', '564176000', '564674000', '565077000', '566489000', '566806000', '566888000', '567532000', '567545000', '567581000', '567601000', '572350220', '574002600', '574161151', '574999010', '603500186', '603500187', '613002810', '626004011', '636009681', '636009993', '636010539', '636011776', '636012462', '636012580', '636014487', '636014671', '636015683', '636017131', '636017687', '636017688', '636017704', '636017705', '636017939', '636017940', '636018032', '636018033', '636018074', '636018174', '636019527', '636019750', '636021407', '642122000', '642122040', '657126100', '657126200', '657161000', '657203000', '657210000', '657384000', '657430000', '657485000', '657830000', '657991000', '672708000', '775072000', '777777777']
#%%
subset = []

from subprocess import call
for ssvid in subset:
    call(['open', '-a' '/Applications/QGIS3.8.app', f"/Users/jonathanraphael/git/ceruleanserver/local/temp/outputs/ssvid_search_ais/ssvid_split/{ssvid}.geojson"])
# %%
import geopandas as gpd
bulk_df = gpd.read_file(f"/Users/jonathanraphael/git/ceruleanserver/local/temp/outputs/ssvid_search_ais/FPSOs_2022-07.geojson")

# %%
# This plots a given FPSO's AIS gap histogram and a lognormal distribution on top of it
import geopandas as gpd
from math import log
from scipy.stats import norm, probplot
import dateutil
import matplotlib.pyplot as plt
import numpy as np

# ssvid = 224805000 # strong consistent signal
# ssvid = 309776000 # strong consistent signal
# ssvid = 309669000 # spotty signal
# ssvid = 311000143 # spotty signal
# ssvid = 251826940 # daily signal
# ssvid = 311000945 # daily signal
ssvids = [224805000, 309776000, 309669000, 311000143, 251826940, 311000945]
for ssvid in ssvids_all[:50]:
    df = gpd.read_file(f"/Users/jonathanraphael/git/ceruleanserver/local/temp/outputs/all FPSO _ais/ssvid_split/{ssvid}.geojson")
    df["time_diff"] = df["timestamp"].map(dateutil.parser.parse).diff().map(lambda o: o.total_seconds())/60/60
    log_diff = df["time_diff"].iloc[1:].map(log)
    
    # plt.hist(log_diff, bins=20, density=True)
    # mu, std = norm.fit(log_diff)
    # x = np.linspace(*plt.xlim(), 100)
    # p = norm.pdf(x, mu, std)
    # plt.plot(x, p, 'k', linewidth=2)
    # plt.title(f"Lognormal AIS Gaps ({ssvid}): mu = %.2f,  std = %.2f" % (mu, std))
    # plt.xlabel("Hours Between Broadcasts")
    # plt.show()

    out = probplot(log_diff, plot=plt, alpha=1, plot_fit=True) #XXX This line requires a modified version of "_morestats.py". Ask Jona for a copy, or delete extra kwargs alpha and plot_fit
    plt.title(f"Probability Plot: {ssvid} (r^2 = {out[1][2]**2:.2f})")
    # plt.xlim(-4,4)
    # plt.ylim(-4,4)
    plt.show()
    



# %%



# %%
# This code is used to parse the long-form outputs of Tatiana's IMO web scraper
import pandas as pd
fn = '/Users/jonathanraphael/Downloads/scraper_output_3.csv'
df = pd.read_csv(fn, encoding = 'unicode_escape', header=None)
column_names = ["Name", "IMO Number", "Flag", "Call sign", "MMSI", "Ship UN Sanction", "Owning/operating entity under UN Sanction", "Type", "Converted from", "Date of build", "Gross tonnage", "Registered owner"]
reshaped = pd.DataFrame(columns = column_names)

vessel_count = -1
for i, item in df[0].iteritems():
    idx = [key in item for key in column_names]
    if any(idx):
        col = column_names[idx.index(True)]
        if col == "Name":
            vessel_count+=1
        reshaped.at[vessel_count, col] = item[(len(col)+1):]        
reshaped.to_csv('/Users/jonathanraphael/Downloads/scraper_reshaped_3.csv')
# %%
