# Expected Improvement for Contextual Bandits

This repository contains the code for the publication "Expected Improvement for Contextual Bandits" accepted at NeurIPS 2022. The project implements a novel approach for enhancing the performance of contextual bandit algorithms using the concept of Expected Improvement (EI).

## Files Description

- **bandit.py**: Defines the `ContextualBandit` class, which simulates a contextual bandit environment. Key dependencies include `numpy`, `itertools`, `random`, and `torch`.

- **ei.py**: Implements the `EI` class, an abstract base class for Expected Improvement calculation. This file leverages `numpy`, `abc` for abstract base classes, `tqdm` for progress bars, `utils` for utility functions like inverse Sherman-Morrison formula, and `scipy.stats` for statistical functions.

- **LinEI.py**: Contains utility functions such as `sample_x` for sampling contexts. It imports various `numpy` modules, `scipy.stats`, and `matplotlib.pyplot` for visualization.

- **main_neuralEI.py**: Serves as the main entry point for experiments, demonstrating the usage of the `ContextualBandit` and `NeuralEI` classes. Dependencies include `numpy`, `matplotlib.pyplot`, `seaborn`, and modules from `bandit.py` and `neural_ei.py`.

- **neural_ei.py**: Defines the `NeuralEI` class, an implementation of Expected Improvement using neural networks. This file uses `numpy`, `torch`, and classes from `ei.py` and `utils.py`.

- **utils.py**: Provides utility functions and classes like `Model` and `inv_sherman_morrison`. It depends on `numpy` and `torch.nn`.

## Usage

To run the main experiments, execute `main_neuralEI.py`. Ensure that all dependencies are installed. The scripts demonstrate how contextual bandits can be optimized using our Expected Improvement approach.

## Requirements

The project requires Python with packages such as `numpy`, `torch`, `matplotlib`, `seaborn`, `scipy`, and others. A `requirements.txt` file can be used to install all necessary packages.

## License

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

