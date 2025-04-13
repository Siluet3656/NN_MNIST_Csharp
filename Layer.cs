public class Layer
{
    public int InputSize { get; }
    public int OutputSize { get; }
    public double[,] Weights { get; }
    public double[] Biases { get; }
    public double[] Outputs { get; private set; }
    public double[] Inputs { get; private set; }
    
    private readonly Random _random;

    public Layer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Weights = new double[outputSize, inputSize];
        Biases = new double[outputSize];
        Outputs = new double[outputSize];
        Inputs = new double[inputSize];
        _random = new Random();
        
        InitializeWeights();
    }

    private void InitializeWeights()
    {
        double scale = Math.Sqrt(2.0 / (InputSize + OutputSize));
        for (int i = 0; i < OutputSize; i++)
        {
            for (int j = 0; j < InputSize; j++)
            {
                Weights[i, j] = _random.NextDouble() * scale * 2 - scale;
            }
            Biases[i] = 0.01;
        }
    }

    public double[] Forward(double[] inputs)
    {
        if (inputs.Length != InputSize)
            throw new ArgumentException("Input size mismatch");

        Inputs = inputs;

        for (int i = 0; i < OutputSize; i++)
        {
            double sum = Biases[i];
            for (int j = 0; j < InputSize; j++)
            {
                sum += inputs[j] * Weights[i, j];
            }
            Outputs[i] = ReLU(sum);
        }

        return Outputs;
    }

    public double[] Backward(double[] errors, double learningRate)
    {
        double[] inputErrors = new double[InputSize];
        double[] outputErrors = new double[OutputSize];
        
        for (int j = 0; j < InputSize; j++)
        {
            double sum = 0;
            for (int i = 0; i < OutputSize; i++)
            {
                sum += errors[i] * ReLUDerivative(Outputs[i]) * Weights[i, j];
            }
            inputErrors[j] = sum;
        }
        
        for (int i = 0; i < OutputSize; i++)
        {
            double delta = errors[i] * ReLUDerivative(Outputs[i]);
            for (int j = 0; j < InputSize; j++)
            {
                Weights[i, j] -= learningRate * delta * Inputs[j];
            }
            Biases[i] -= learningRate * delta;
        }

        return inputErrors;
    }

    private static double ReLU(double x) => x > 0 ? x : 0;
    private static double ReLUDerivative(double x) => x > 0 ? 1 : 0;
}