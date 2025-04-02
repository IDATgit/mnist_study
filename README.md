# MNIST Study Platform

## Project Overview

This project is an in-depth investigation of the MNIST dataset and Deep Neural Networks (DNNs). The primary goal is to explore and understand several key aspects of neural networks and their performance on handwritten digit recognition:

1. **Capabilities and Limitations**: What can current DNNs achieve on the MNIST dataset, and where do they fall short?
2. **Architectural Properties**: Which neural network properties are most relevant to performance in this specific case?
3. **Generalization and Real-World Applicability**: Is MNIST truly "solved"? What does this mean for real-world handwriting recognition? How well do these models generalize?

Through this investigation, we aim to gain deeper insights into the practical aspects of neural networks and their application to image recognition tasks.

## Project Structure

The project is organized into several key directories, each serving a specific purpose:

### `data/`
- Contains the MNIST dataset in its raw format
- Organized into training and testing sets
- Includes both images and labels

### `data_inspection/`
- Scripts for analyzing and visualizing the MNIST dataset
- Outputs include:
  - Sample digit visualizations
  - Label distribution analysis
  - Dataset statistics

### `models/`
- Contains two types of neural network implementations:
  - `parametric_models/`: Flexible architectures with configurable parameters
  - `specific_models/`: Fixed architectures optimized for MNIST
- Includes both fully connected and convolutional network implementations

### `model_interpretation/`
- Tools for analyzing model performance and behavior
- Includes:
  - Interactive drawing application for real-time testing
  - Basic interpretation scripts for model analysis
  - Confusion matrix visualizations

### `trainers/`
- Training infrastructure for different model types
- Includes:
  - Generic training utilities
  - Model-specific training scripts
  - Checkpoint management
  - TensorBoard logging

### `utils/`
- Common utilities and helper functions
- Data loading and preprocessing tools

## Study Insights

*(This section will be updated as the investigation progresses)*

### Current Findings

1. **Model Performance**
   - Both fully connected and convolutional networks achieve high accuracy (>98%) on the test set
   - Convolutional networks show better generalization and robustness to variations

2. **Architectural Considerations**
   - Network depth and width impact learning speed and final performance
   - Batch normalization significantly improves training stability
   - Dropout helps prevent overfitting, especially in deeper networks

3. **Generalization Challenges**
   - Models trained on MNIST struggle with:
     - Different writing styles
     - Rotated digits
     - Varying stroke widths
     - Noisy or distorted inputs

### Ongoing Investigations

1. **Robustness Testing**
   - Evaluating model performance on modified MNIST samples
   - Testing with real-world handwritten digits
   - Analyzing failure cases

2. **Architectural Experiments**
   - Comparing different network architectures
   - Investigating the role of various components
   - Exploring novel approaches

3. **Practical Applications**
   - Real-world deployment considerations
   - Performance optimization
   - Integration challenges

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the interactive drawing application: `python model_interpretation/interactive_draw.py`

## Contributing

This is a personal study project, but feedback and suggestions are welcome. Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 