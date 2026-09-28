# Data

This project uses two independent brain MRI datasets for cross-dataset evaluation:

- **BDNeuro-MRI** — a four-class brain MRI dataset containing glioma, meningioma, pituitary tumor, and no-tumor images.
- **PMRAM** — an independent brain MRI dataset containing the same four diagnostic categories.

The experimental dataset used in this project contains:

| Dataset | Images |
|---|---:|
| BDNeuro-MRI | 5,941 |
| PMRAM | 1,410 |
| **Total** | **7,351** |

The datasets were harmonized to the following four labels:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

## Dataset Usage

BDNeuro-MRI uses its predefined training, validation, and test partitions.

For PMRAM, the prepared experimental dataset was divided using a stratified split with `random_state=42`:

- Training: 987 images
- Validation: 211 images
- Test: 212 images

When PMRAM was used as the independent external dataset, all 1,410 images in the prepared experimental set were used for evaluation.

## Data Availability

Raw MRI images are not redistributed in this repository.

The original datasets are available from their respective sources:

- **BDNeuro-MRI:** Mendeley Data, DOI: `10.17632/zwr4ntf94j.8`
- **PMRAM:** Mendeley Data, DOI: `10.17632/m7w55sw88b.1`

The notebooks in this repository assume that the datasets have been downloaded separately and organized locally before execution.

