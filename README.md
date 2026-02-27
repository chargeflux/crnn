# crnn

`crnn` implements the CRNN model described by the paper "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition" by Shi et al. with a slight variation in the BiLSTM layers.

## Getting Started

### Initialize environment
```shell
uv sync --extra <cu128|cpu>
```

### Usage

```shell
crnn train [--device cuda,cpu] [--vocabulary VOCABULARY] [--dataset mnist] [-o OUTPUT_DIR]

crnn infer [--device cuda,cpu] [--vocabulary VOCABULARY] -f FILE -m MODEL_PATH
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

