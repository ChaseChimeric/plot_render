## Overview
This folder contains the pass renders I have completed so far.
 
There are quite a few passes where the data either didn't process correctly or didn't exist for the time period shown so they probably wont be that great
I think pass 4/5 are pretty definitive as to what a good pass/render should look like.

I'll try to keep the code I used updated at https://github.com/ChaseChimeric/plot_render.git

Below is an example of my current working file structure

jupyter_stuff/
├─ datasets/
│  ├─ QTN_merged_199X.h5
│  ├─ TNR_results_199X.h5
│  ├─ MFI_GSE_merged_one_sec.h5
│  ├─ BS_pass.h5
│  ├─ gse_position_199X.h5
│  ├─ wi_ors_pre_1996.cdf
│  ├─ extract_gse.py
├─ notebooks/
│  ├─ render.py
│  ├─ render_outbound.py
│  ├─ render_to_show.ipynb
│  ├─ pass1/
│  │  ├─ animated_plot0.mp4
│  │  ├─ animated_plot1.mp4
│  ├─ pass2/
│  ├─ pass.../

## datasets/
contains all of the larger hdf5/cdf databases that I work off of.
### QTN_merged_199X.h5
contains all the quasi-thermal noise data that I graph with the render script.
I tend to download all the data for a specific year and then render out all the passes for that year.
### TNR_results_199X.h5
Same story here, has all of the particle speed and density data
### MFI_GSE_merged_one_sec.h5
Contains the magnetif field intensity data for the entire timespan we consider, 1995-2004.
### BS_pass.h5
contains very rudimentary data in each pass, I use to find in/out timestamps
### gse_position_199X.h5
predicted gse location for each year. Have to download the corrensponding cdf fro cdaweb first and convert with one of the scripts I have.
### wi_ors_pre_1996.cdf
This file becomes gse_position_199X after processing. Can download from cdaweb (the downloaded file won't have this name, I changed it myself)
### extract_gse.py
I would actually recommend changing the target filename to the one you want and copy pasting this to the python interpreter, it might show errors better
This converts wi_ors_pre_1996.cdf to gse_position_1996.h5 for render.py to use

## notebooks/
I was up until now using a low of jupyter notebooks which are useful for looking at the data, but not as much for batch processing
Now contains the code for rendering passes
### render.py
main render script. this will try to render the inbound pass and then the outbound pass. I've had a couple issues with data not existing for 
certain periods while rendering but for the most part it should be there. The process to use this script is basically changing the file names 
manually to the year I'm currently working on, and then doing `python ./render.py -p {pass number}` for each pass that I need to render. There are
edge cases where passes will fail to render if they exist across years, for example from december 1995 to january 1996, didn't have time to fully fix those
### render_outbound.py
Same as above, but only for the outbound pass. I had a couple issues where the first pass worked, but the second didnt and I had to regenerate.
Don't have one for just inbound cause they haven't given me as many issues.
### render_to_show.ipynb
This is a jupyter notebook that I was using to view the MFI data and QTN plots. May or may not be useful.

## pass1/
folder that contains the rendered video clips
### animated_plot0/1.mp4
The actual render, I would expect most of these to be about 30-90 minutes long
