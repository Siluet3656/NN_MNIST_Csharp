using System.Text.Json;
using System.Text.Json.Serialization;

public static class ModelIO
{
    private static string BaseSavesPath => Path.Combine(Directory.GetCurrentDirectory(), "Saves");
    public static void SaveModel(NeuralNetwork network, string filePath)
    {
        try
        {
            if (!Directory.Exists(BaseSavesPath))
            {
                Directory.CreateDirectory(BaseSavesPath);
            }
            
            string savePath = Path.Combine(BaseSavesPath, filePath);
            
            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                Converters = { new DoubleArrayConverter(), new Double2DArrayConverter() }
            };

            var data = new NetworkData
            {
                LayerSizes = GetLayerSizes(network),
                Weights = GetAllWeights(network),
                Biases = GetAllBiases(network),
                LearningRate = network.GetLearningRate()
            };

            string json = JsonSerializer.Serialize(data, options);
            File.WriteAllText(savePath, json);
            
            Console.WriteLine($"Model successfully saved to {savePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error saving model: {ex.Message}");
        }
    }

    public static NeuralNetwork? LoadModel(string filePath)
    {
        try
        {
            string loadPath = Path.Combine(BaseSavesPath, filePath);
            string json = File.ReadAllText(loadPath);
            
            var options = new JsonSerializerOptions
            {
                Converters = { new DoubleArrayConverter(), new Double2DArrayConverter() }
            };
            
            var data = JsonSerializer.Deserialize<NetworkData>(json, options);
            
            if (data == null)
            {
                Console.WriteLine("Failed to deserialize model data");
                return null;
            }

            var network = new NeuralNetwork(data.LayerSizes, data.LearningRate);
            SetAllWeights(network, data.Weights);
            SetAllBiases(network, data.Biases);

            Console.WriteLine($"Model successfully loaded from {loadPath}");
            return network;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading model: {ex.Message}");
            return null;
        }
    }

    public static int GetNumOfEpoches()
    {
        if (Directory.Exists(BaseSavesPath))
        {
            string searchPattern = "*.json";
            string[] files = Directory.GetFiles(BaseSavesPath, searchPattern);
            return files.Length;
        }
        else
        {
            return 0;
        }
    }
    
    private static int[] GetLayerSizes(NeuralNetwork network)
    {
        var layerSizes = new int[network.GetLayers().Length + 1];
        layerSizes[0] = network.GetLayers()[0].InputSize;
        
        for (int i = 0; i < network.GetLayers().Length; i++)
        {
            layerSizes[i + 1] = network.GetLayers()[i].OutputSize;
        }
        
        return layerSizes;
    }

    private static double[][,] GetAllWeights(NeuralNetwork network)
    {
        var layers = network.GetLayers();
        var weights = new double[layers.Length][,];
        
        for (int i = 0; i < layers.Length; i++)
        {
            weights[i] = layers[i].Weights;
        }
        
        return weights;
    }

    private static double[][] GetAllBiases(NeuralNetwork network)
    {
        var layers = network.GetLayers();
        var biases = new double[layers.Length][];
        
        for (int i = 0; i < layers.Length; i++)
        {
            biases[i] = layers[i].Biases;
        }
        
        return biases;
    }

    private static void SetAllWeights(NeuralNetwork network, double[][,] weights)
    {
        var layers = network.GetLayers();
        
        for (int i = 0; i < layers.Length; i++)
        {
            Array.Copy(weights[i], layers[i].Weights, weights[i].Length);
        }
    }

    private static void SetAllBiases(NeuralNetwork network, double[][] biases)
    {
        var layers = network.GetLayers();
        
        for (int i = 0; i < layers.Length; i++)
        {
            Array.Copy(biases[i], layers[i].Biases, biases[i].Length);
        }
    }

    private class NetworkData
    {
        public int[] LayerSizes { get; set; } = Array.Empty<int>();
        public double[][,] Weights { get; set; } = Array.Empty<double[,]>();
        public double[][] Biases { get; set; } = Array.Empty<double[]>();
        public double LearningRate { get; set; }
    }
    
    private class DoubleArrayConverter : JsonConverter<double[]>
    {
        public override double[] Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            using JsonDocument doc = JsonDocument.ParseValue(ref reader);
            return doc.RootElement.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        }

        public override void Write(Utf8JsonWriter writer, double[] value, JsonSerializerOptions options)
        {
            writer.WriteStartArray();
            foreach (var item in value) writer.WriteNumberValue(item);
            writer.WriteEndArray();
        }
    }

    private class Double2DArrayConverter : JsonConverter<double[,]>
    {
        public override double[,] Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            using JsonDocument doc = JsonDocument.ParseValue(ref reader);
            var rows = doc.RootElement.EnumerateArray().ToArray();
            int rowCount = rows.Length;
            int colCount = rows[0].GetArrayLength();
            
            var result = new double[rowCount, colCount];
            
            for (int i = 0; i < rowCount; i++)
            {
                var cols = rows[i].EnumerateArray().ToArray();
                for (int j = 0; j < colCount; j++)
                {
                    result[i, j] = cols[j].GetDouble();
                }
            }
            
            return result;
        }

        public override void Write(Utf8JsonWriter writer, double[,] value, JsonSerializerOptions options)
        {
            writer.WriteStartArray();
            
            for (int i = 0; i < value.GetLength(0); i++)
            {
                writer.WriteStartArray();
                for (int j = 0; j < value.GetLength(1); j++)
                {
                    writer.WriteNumberValue(value[i, j]);
                }
                writer.WriteEndArray();
            }
            
            writer.WriteEndArray();
        }
    }
}