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
    }
}