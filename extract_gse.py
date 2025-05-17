import cdflib
import pandas as pd
from cdflib.epochs import CDFepoch

# Read CDF data
cdf = cdflib.CDF("D:/jupyter_stuff/datasets/wi_ors_pre_1997.cdf")
epoch_raw = cdf.varget('Epoch')
gse_pos = cdf.varget('GSE_POS')

# Convert CDF Epoch to datetime
epoch_dt = CDFepoch.to_datetime(epoch_raw)

# Build DataFrame with time index
df = pd.DataFrame({
    'GSE_X [km]': gse_pos[:, 0],
    'GSE_Y [km]': gse_pos[:, 1],
    'GSE_Z [km]': gse_pos[:, 2]
}, index=pd.to_datetime(epoch_dt))
df.index.name = 'Time'  # (optional, but recommended)

# Save as HDF5
df.to_hdf('gse_position_1997.h5', key='gse_pos', mode='w')