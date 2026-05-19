# Bone Segmentation in X-ray Images using U-Net

This project implements a U-Net based deep learning model for bone segmentation on X-ray images using PyTorch.

The goal of the project is to segment all bone regions in X-ray images by treating all bones as a single class and the background as another class.

## Dataset

The dataset was provided externally with X-ray images and JSON annotations.

Images and JSON labels:

https://drive.google.com/drive/folders/1b9hX52VOdfKuB-D4eFhWwCW7kM4wiwHL?usp=sharing

Numpy labels:

https://drive.google.com/drive/folders/19ghKdTq1Oh3GbwWhE2_jqMBhuzdd0wdo?usp=sharing

Dataset files are not included in this repository.

## Preprocessing

The original JSON annotation files contained multiple bone labels.  
For this project, all bone labels were merged into a single binary mask:

- Bone pixels → 1
- Background pixels → 0

Binary masks were generated using OpenCV polygon operations.

## Model

The segmentation model is based on the U-Net architecture implemented with PyTorch.

The project includes:

- Dataset preprocessing
- Binary mask generation
- U-Net implementation
- Training and validation
- Dice Score evaluation
- Hausdorff Distance evaluation
- 5-Fold Cross Validation
- Prediction visualization

## Evaluation

The model was evaluated using:

- Dice Score
- Hausdorff Distance

5-Fold Cross Validation was used during evaluation.

Final average results:

- Average Dice Score: 0.6986
- Average Hausdorff Distance: 38.7923

Detailed fold results are available in:

```text
reports/cross_validation_results.txt
```

## Prediction Output

Prediction examples are saved in the `outputs` folder.

Example output:

```text
outputs/prediction_result.png
```

![Prediction Example](outputs/prediction_result.png)

## Project Structure

```text
bone-segmentation/

├── outputs/
├── reports/
├── src/

├── README.md
├── requirements.txt
```

## Source Files

Main source files inside `src/`:

- create_masks.py
- cross_validation.py
- dataset.py
- dice_score.py
- hausdorff.py
- predict.py
- train.py
- train_val.py
- unet.py

## Requirements

Main libraries used in the project:

- PyTorch
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- SciPy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Generate binary masks:

```bash
python create_masks.py
```

Train the model:

```bash
python train_val.py
```

Run 5-Fold Cross Validation:

```bash
python cross_validation.py
```

Generate prediction outputs:

```bash
python predict.py
```

## Author

Gülsu Naz Koçak