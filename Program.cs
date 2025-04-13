using System.Diagnostics;
using Microsoft.ML;

class Program
{
    static void Main()
    {
        Console.WriteLine("Loading MNIST dataset...");
            
        (double[], int)[] trainData = MNISTReader.LoadDataSet(
            "train-images-idx3-ubyte", 
            "train-labels-idx1-ubyte", 
            60000);
            
        (double[], int)[] testData = MNISTReader.LoadDataSet(
            "t10k-images-idx3-ubyte", 
            "t10k-labels-idx1-ubyte", 
            10000);
            
        Console.WriteLine($"Loaded {trainData.Length} training samples");
        Console.WriteLine($"Loaded {testData.Length} test samples");
        Console.WriteLine($"First 5 training label: {trainData[0].Item2}, {trainData[1].Item2}, {trainData[2].Item2}, {trainData[3].Item2}, {trainData[4].Item2}");
        
        var nn = new NeuralNetwork(new[] {28*28, 128, 128, 10}, 0.001);

        while (true)
        {
            Console.WriteLine("=== MNIST Neural Network v1.1 (28*28/128/128/10) ===");
            Console.WriteLine("1. Training");
            Console.WriteLine("2. Testing");
            Console.WriteLine("3. Load model from file");
            Console.WriteLine("4. Test one image");
            Console.WriteLine("5. Exit");
            Console.Write("Choose (1-4): ");

            string input = Console.ReadLine();

            switch (input)
            {
                case "1":
                    Train(nn, trainData);
                    break;
                case "2":
                    Test(nn, testData);
                    break;
                case "3":
                    int numOfEpoches = ModelIO.GetNumOfEpoches();
                    if (numOfEpoches > 0)
                    {
                        Console.WriteLine("Choose file: ");
                        for (int i = 0; i < numOfEpoches; i++)
                        {
                            Console.WriteLine("MNIST_Model_e"+ i +".json");
                        }
                        input = Console.ReadLine();
                        nn = ModelIO.LoadModel("MNIST_Model_e"+ input +".json");
                    }
                    else
                    {
                        Console.WriteLine("No files found.");
                    }
                    break;
                case "4":
                    Console.WriteLine("???");
                    break;
                case "5":
                    Console.WriteLine("Exiting...");
                    return;
                default:
                    Console.WriteLine("Invalid input! Try again.");
                    break;
            }

            Console.WriteLine("\nPress any key to continue...");
            Console.ReadKey();
            Console.Clear();
        }
    }

    private static double CrossEntropyLoss(double[] prediction, double[] target)
    {
        double loss = 0;
        for (int i = 0; i < prediction.Length; i++)
        {
            loss += target[i] * Math.Log(prediction[i] + 1e-10);
        }
        return -loss;
    }

    private static void Train(NeuralNetwork nn,(double[], int)[] trainData)
    {
        Console.WriteLine("Training started...");
        int epochs = 10;
        int batchSize = 32;

        var stopwatch = Stopwatch.StartNew();
        
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}");
            double error = 0;
            int correct = 0;
            
            var shuffledData = trainData.OrderBy(x => Guid.NewGuid()).ToArray();

            for (int i = 0; i < shuffledData.Length; i += batchSize)
            {
                var batch = shuffledData.Skip(i).Take(batchSize).ToArray();
                
                foreach (var (image, label) in batch)
                {
                    double[] target = new double[10];
                    target[label] = 1;
                    
                    nn.Train(image, target);
                    
                    var prediction = nn.Predict(image);
                    error += CrossEntropyLoss(prediction, target);
                    if (prediction.ArgMax() == label) correct++;
                }

                if (i % (batchSize * 100) == 0)
                {
                    Console.WriteLine($"Progress: {i * 100.0 / shuffledData.Length:F1}%");
                }
            }
            
            double accuracy = correct * 100.0 / shuffledData.Length;
            double avgError = error / shuffledData.Length;
            Console.WriteLine($"Epoch {epoch + 1}: Error = {avgError:F4}, Accuracy = {accuracy:F2}%");
            ModelIO.SaveModel(nn, "MNIST_Model_e" + epoch +".json");
        }

        stopwatch.Stop();
        Console.WriteLine($"Training completed in {stopwatch.Elapsed.TotalSeconds:F2} seconds");
    }

    private static void Test(NeuralNetwork nn, (double[], int)[] testData)
    {
        Console.WriteLine("Testing...");
        int testCorrect = 0;
        foreach (var (image, label) in testData)
        {
            var prediction = nn.Predict(image);
            if (prediction.ArgMax() == label) testCorrect++;
        }

        double testAccuracy = testCorrect * 100.0 / testData.Length;
        Console.WriteLine($"Test Accuracy: {testAccuracy:F2}%");
    }
}

public static class ArrayExtensions
{
    public static int ArgMax(this double[] array)
    {
        int maxIndex = 0;
        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > array[maxIndex]) maxIndex = i;
        }
        return maxIndex;
    }
}