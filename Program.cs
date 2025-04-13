using System.Diagnostics;
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
        Console.WriteLine($"First training label: {trainData[0].Item2}");
        
        // Создаем нейросеть: 784 входа, 2 скрытых слоя по 128 нейронов, 10 выходов
        var nn = new NeuralNetwork(new[] {784, 128, 128, 10}, 0.001);

        // Обучение
        Console.WriteLine("Training started...");
        int epochs = 10;
        int batchSize = 32;

        var stopwatch = Stopwatch.StartNew();
        
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}");
            double error = 0;
            int correct = 0;

            // Перемешиваем данные
            var shuffledData = trainData.OrderBy(x => Guid.NewGuid()).ToArray();

            for (int i = 0; i < shuffledData.Length; i += batchSize)
            {
                var batch = shuffledData.Skip(i).Take(batchSize).ToArray();
                
                foreach (var (image, label) in batch)
                {
                    // Преобразуем метку в one-hot вектор
                    double[] target = new double[10];
                    target[label] = 1;

                    // Обучаем на текущем примере
                    nn.Train(image, target);

                    // Вычисляем ошибку и точность
                    var prediction = nn.Predict(image);
                    error += CrossEntropyLoss(prediction, target);
                    if (prediction.ArgMax() == label) correct++;
                }

                if (i % (batchSize * 100) == 0)
                {
                    Console.WriteLine($"Progress: {i * 100.0 / shuffledData.Length:F1}%");
                }
            }

            // Выводим статистику после эпохи
            double accuracy = correct * 100.0 / shuffledData.Length;
            double avgError = error / shuffledData.Length;
            Console.WriteLine($"Epoch {epoch + 1}: Error = {avgError:F4}, Accuracy = {accuracy:F2}%");
        }

        stopwatch.Stop();
        Console.WriteLine($"Training completed in {stopwatch.Elapsed.TotalSeconds:F2} seconds");

        // Тестирование
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

    private static double CrossEntropyLoss(double[] prediction, double[] target)
    {
        double loss = 0;
        for (int i = 0; i < prediction.Length; i++)
        {
            loss += target[i] * Math.Log(prediction[i] + 1e-10);
        }
        return -loss;
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