# crnn

`crnn` implements the CRNN model described by the paper "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition" by Shi et al. with a slight variation in the BiLSTM layers.

## Getting Started

### Initialize environment
```shell
uv sync --extra <cu128|cpu>
```

### Usage

#### Train

Choose between available datasets or a custom dataset. Custom datasets require a label text file for every image where a `.txt` extension replaces the image extension. The path to the custom dataset should have `train` and `test` subdirectories. 

**Vocabulary**: Provide a string representing a character set for prediction with a **blank token** character at index 0, such as "-0123456789" for digits only

```shell
crnn train [--device cuda,cpu] --vocabulary <string> [--dataset MNIST|LABEL_FILE] [-o OUTPUT_DIR]
```

#### Infer

Run inference on an image using a trained model. 

**Vocabulary**: Provide a `vocab.txt` file in the model directory or specify using the `--vocabulary` flag

```shell
crnn infer [--device cuda,cpu]  [--vocabulary <string>] -f <image file> -m <path to model>
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

