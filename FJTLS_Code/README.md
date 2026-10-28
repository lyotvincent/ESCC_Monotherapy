# Multiplex immunofluorescence processing and analysis
```text
data/
  slides/                             
  registered/<sample_id>/registered_slides/
    DAPI.ome.tiff
    <marker>.ome.tiff
  patches/<sample_id>.h5              # dataset named coords
  nucleiseg/                          # nuclear segmentation output
  wholecellseg/                       # whole-cell segmentation output
  cell_marker_matrix/<sample_id>/
    <sample_id>_cell_marker_matrix.h5ad
  clinical.csv                        # sample_id and response columns
```
## Workflow

```sh
# DAPI segmentation for all samples (requires the appropriate GPU environment).
sh run_segment_DAPI_all.sh

# Whole-cell region construction for one sample.
sh run_segment_whole_cell.sh SAMPLE_ID

# Cell-by-marker matrix for the same sample.
sh run_generate_cell_marker_matrix.sh SAMPLE_ID

# Marker and spatial analyses; repeat for the desired analysis script.
python analyze/CCL19/CCL19.py --matrix-root data/cell_marker_matrix --clinical-csv data/clinical.csv
python analyze/CCR7/CCR7.py --matrix-root data/cell_marker_matrix --clinical-csv data/clinical.csv
python analyze/CCL19_CCR7/CCL19_CCR7.py --matrix-root data/cell_marker_matrix --clinical-csv data/clinical.csv
python analyze/CD4_8_CCR7/CD4_8_CCR7.py --matrix-root data/cell_marker_matrix --clinical-csv data/clinical.csv
python analyze/CD8/CD8.py --matrix-root data/cell_marker_matrix --clinical-csv data/clinical.csv

# Run after CD4_8_CCR7.py has regenerated its response-metrics CSV.
python analyze/CD4_8_CCR7/generate_large_font_boxplot.py
```