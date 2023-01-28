# Install tensorflow on M1 machine

## Using pip

I recommend sticking to those version of tf to avoid errors.
```shell
python3 -m venv .env
source .env/bin/activate
pip install tensorflow-macos==2.9
```
To take advantage of the M1 GPU (optional)
```shell
pip install tensorflow-metal==0.5.0
```
Test the installation
```shell
echo "import tensorflow as tf

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)" >> test.py
```

```shell
python test.py
```

## Using miniconda

Install homebrew
```shell
/bin/bash -c “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Install xcode command line tools
```shell
xcode-select --install
```
Install miniforge
```shell
conda config --set auto_activate_base false
```
Create environment
```shell
conda create --name tf python=3.9
```
Activate environment
```shell
conda activate tf
```
Install tensorflow dependencies
```shell
conda install -c apple tensorflow-deps
```
Install base tensorflow
```shell
pip install tensorflow-macos
```
Install metal plugin
```shell
pip install tensorflow-metal
```
Install other packages if needed
```
conda install pip
pip install -r requirements.txt
```