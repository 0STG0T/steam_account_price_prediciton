# Steam Account Price Prediction

Docker-based deployment for Steam account price prediction model.

## Setup

### Prerequisites
- Docker
- Docker Compose

### Installation
1. Clone the repository:
```bash
git clone https://github.com/0STG0T/steam_account_price_prediciton.git
cd steam_account_price_prediciton
```

2. Build the Docker image:
```bash
docker-compose build
```

## Usage

### Training Mode
To start the Jupyter notebook server for training:
```bash
docker-compose up
```
Access Jupyter at http://localhost:8888

### Prediction Mode
To run predictions using a trained model:
```bash
docker-compose run --rm steam-price-predictor predict
```

## Data and Models
- Place your training data in the `data/` directory
- Trained models will be saved to the `models/` directory
- Both directories are mounted as volumes in the container

## Project Structure
```
.
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── project/
│   ├── main/
│   └── idea_notebooks/
├── data/
├── models/
└── docker-compose.yml
```

## Development
The Docker setup supports both training and inference workflows:
- Training mode launches Jupyter for interactive development
- Prediction mode runs the ONNX inference example

## Notes
- Jupyter server runs without authentication for development
- Consider adding security measures for production deployment
- Models are exported in ONNX format for efficient inference
