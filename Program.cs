// Program.cs
using System;

class Program
{
    static void Main()
    {
        try
        {
            Console.WriteLine("Loading MNIST dataset...");
            
            var trainData = MNISTReader.LoadDataSet(
                "train-images-idx3-ubyte", 
                "train-labels-idx1-ubyte", 
                60000);
            
            var testData = MNISTReader.LoadDataSet(
                "t10k-images-idx3-ubyte", 
                "t10k-labels-idx1-ubyte", 
                10000);

            Console.WriteLine($"Loaded {trainData.Length} training samples");
            Console.WriteLine($"Loaded {testData.Length} test samples");
            Console.WriteLine($"First training label: {trainData[0].Item2}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine("Make sure you have MNIST files in the 'Data' folder:");
            Console.WriteLine("- train-images-idx3-ubyte");
            Console.WriteLine("- train-labels-idx1-ubyte");
            Console.WriteLine("- t10k-images-idx3-ubyte");
            Console.WriteLine("- t10k-labels-idx1-ubyte");
            //Console.WriteLine($"Current data path: {MNISTReader.BaseDataPath}");
        }
    }
}