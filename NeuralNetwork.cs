public class NeuralNetwork
{
    private readonly Layer[] _layers;
    private readonly double _learningRate;

    public NeuralNetwork(int[] layerSizes, double learningRate)
    {
        if (layerSizes.Length < 2)
            throw new ArgumentException("Network must have at least 2 layers");

        _learningRate = learningRate;
        _layers = new Layer[layerSizes.Length - 1];

        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
        }
    }

    public double[] Predict(double[] input)
    {
        double[] output = input;
        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
        }
        return Softmax(output);
    }

    public void Train(double[] input, double[] target)
    {
        Predict(input);

        double[] errors = new double[_layers[^1].OutputSize];
        double[] output = _layers[^1].Outputs;

        for (int i = 0; i < errors.Length; i++)
        {
            errors[i] = output[i] - target[i];
        }

        for (int i = _layers.Length - 1; i >= 0; i--)
        {
            errors = _layers[i].Backward(errors, _learningRate);
        }
    }

    private static double[] Softmax(double[] values)
    {
        double[] exp = new double[values.Length];
        double sum = 0;

        for (int i = 0; i < values.Length; i++)
        {
            exp[i] = Math.Exp(values[i]);
            sum += exp[i];
        }

        for (int i = 0; i < exp.Length; i++)
        {
            exp[i] /= sum;
        }

        return exp;
    }
}