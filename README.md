# Body-inference-network-paper
Analysis code (incl. RSA) accompanying the Body Inference Network (BIN) paper

## Data location
We provide the Apptainer container (needed to run the scripts), neural data, RSMs [here](https://yaleedu-my.sharepoint.com/:f:/g/personal/hakan_yilmaz_yale_edu/EsBl2XSXw61NqLoJ3j6BL6cBE8__HD0YzdfvHce9wdxPCg?e=NGV0tq).

## Run scripts 
To run the scripts, download the container `cont.sif` and execute the sripts, for example, as such:
``` bash
./run.sh python scripts/make_rdms.py --glob_dir "SPIN/*" --plot --img-type svg
```

## Models 
The model code is available in the following repositories:
- [BIN and Class-Pose](https://github.com/CNCLgithub/SPIN)
- [SimCLR](https://github.com/CNCLgithub/FFCV-SSL)
- [MAE](https://github.com/CNCLgithub/SparK)

Please also find the synthesized monkey imageset in the BIN (and Class-Pose) repository.
